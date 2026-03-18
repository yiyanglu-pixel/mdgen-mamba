# MDGen-Mamba

Fork of [MDGen](https://arxiv.org/abs/2409.17808) (Generative Modeling of Molecular Dynamics Trajectories) by Bowen Jing\*, Hannes Stark\*, Tommi Jaakkola, and Bonnie Berger — with **Mamba (Selective State Space Model) integration** for linear-complexity temporal modeling.

We introduce generative modeling of molecular trajectories as a paradigm for learning flexible multi-task surrogate models of MD from data. By conditioning on appropriately chosen frames of the trajectory, such generative models can be adapted to diverse tasks such as forward simulation, transition path sampling, and trajectory upsampling. By alternatively conditioning on part of the molecular system and inpainting the rest, we also demonstrate the first steps towards dynamics-conditioned molecular design. We validate these capabilities on tetrapeptide simulations and show initial steps towards learning trajectories of protein monomers. Methodological details and further evaluations can be found in the paper. Please feel free to reach out to us at bjing@mit.edu, hstark@mit.edu with any questions.

**Mamba integration:** This fork replaces the O(n²) self-attention in the temporal dimension (`mha_t`) with Mamba's O(n) selective state space model, enabling scalable generation of long MD trajectories (1,000–10,000+ frames). The spatial attention (`mha_l`) and IPA modules are preserved unchanged, maintaining MDGen's physical geometric priors.

**Note:** This repository is provided for research reproducibility and is not intended for usage in application workflows.

![mdgen.png](mdgen.png)

## Installation

```
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
pip install wandb dm-tree einops torchdiffeq fair-esm pyEMMA
pip install matplotlib==3.7.2 numpy==1.21.2
```

### Mamba dependencies (optional, required for `--mamba` flag)

```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0.0
```

> **Note:** `mamba-ssm` requires CUDA and a compatible GPU. See [mamba-ssm installation guide](https://github.com/state-spaces/mamba) for details.

## Datasets

1. Download the tetrapeptide MD datasets:
```
mkdir -p data/4AA_sims data/4AA_sims_implicit
gsutil -m rsync -r gs://mdgen-public/4AA_sims data/4AA_sims
gsutil -m rsync -r gs://mdgen-public/4AA_sims_implicit data/4AA_sims_implicit
```
**Update: we are temporarily unable to publicly host the MD dataset. Please contact us for access.**


2. Download the ATLAS simulations via https://github.com/bjing2016/alphaflow/blob/master/scripts/download_atlas.sh to `data/atlas_sims`.
3. Preprocess the tetrapeptide simulations
```
# Forward simulation and TPS, prep with interval 100 * 100fs = 10ps
python -m scripts.prep_sims --splits splits/4AA.csv --sim_dir data/4AA_sims --outdir data/4AA_data --num_workers [N] --suffix _i100 --stride 100

# Upsampling, prep with interval 100fs
python -m scripts.prep_sims --splits splits/4AA_implicit.csv --sim_dir data/4AA_sims_implicit --outdir data/4AA_data_implicit --num_workers [N]

# Inpainting, prep with interval 100fs
python -m scripts.prep_sims --splits splits/4AA.csv --sim_dir data/4AA_sims --outdir data/4AA_data --num_workers [N]
```
4. Preprocess the ATLAS simulations
```
# Prep with interval 40 * 10 ps = 400 ps
python -m scripts.prep_sims --splits splits/atlas.csv --sim_dir data/atlas_sims --outdir data/atlas_data --num_workers [N] --suffix _i40 --stride 40
```

## Mamba Integration

### Architecture overview

MDGen's core data tensor has shape `(B, T, L, C)` where B=batch, T=frames, L=residues, C=features. The `LatentMDGenLayer` alternates between:

1. **Spatial attention (`mha_l`)** — operates along the residue dimension L (unchanged)
2. **Temporal attention (`mha_t`)** — operates along the frame dimension T (**replaced by Mamba**)
3. **FFN** — pointwise feed-forward (unchanged)

The Mamba integration replaces only `mha_t`, reducing temporal modeling complexity from O(T²) to O(T). This is the primary bottleneck when scaling to long trajectories (T > 1,000 frames).

```
LatentMDGenLayer forward:
  x: (B, T, L, C)
    │
    ├─ [optional IPA] ──── Invariant Point Attention (unchanged)
    │
    ├─ mha_l ───────────── Spatial attention over residues (unchanged)
    │                       reshape to (B*T, L, C), O(L²)
    │
    ├─ mha_t ───────────── Temporal modeling over frames (★ REPLACED)
    │                       reshape to (B*L, T, C)
    │                       Attention: O(T²)  →  Mamba: O(T)
    │
    └─ FFN ─────────────── Feed-forward network (unchanged)
```

### New modules (`mdgen/model/mamba_operators.py`)

| Class | Description |
|-------|-------------|
| `MambaOperator` | Unidirectional Mamba-1 wrapper. Input/output: `(B, T, C)`. Linear complexity O(T). |
| `BiMambaOperator` | Bidirectional Mamba — runs forward + backward, combines via concat projection (default), averaging, or learned gating. Essential for MD where causality is not enforced. |
| `MambaTemporalBlock` | Drop-in replacement for `AttentionWithRoPE`. Wraps `BiMambaOperator` (default) or `MambaOperator` based on `bidirectional` flag. |

### CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mamba` | flag | `False` | Enable Mamba for temporal attention (`mha_t`) |
| `--bi_mamba` | flag | `False` | Use bidirectional Mamba (requires `--mamba`) |
| `--mamba_d_state` | int | `64` | SSM state dimension. Larger = more memory, better long-range recall |
| `--mamba_d_conv` | int | `4` | Local convolution kernel size inside Mamba |
| `--mamba_expand` | int | `2` | Expansion factor for Mamba's inner dimension |

### Modified files

| File | Changes |
|------|---------|
| `mdgen/model/mamba_operators.py` | **New file.** `MambaOperator`, `BiMambaOperator`, `MambaTemporalBlock` |
| `mdgen/model/latent_model.py` | `LatentMDGenLayer.__init__`: Mamba branch in `_init_submodules`; `forward`: Mamba branch for temporal dim; `initialize_weights`: skip Mamba internal modules to preserve SSM initialization |
| `mdgen/wrapper.py` | `NewMDGenWrapper`: pass Mamba parameters to model |
| `mdgen/parsing.py` | Add `--mamba`, `--bi_mamba`, `--mamba_d_state`, `--mamba_d_conv`, `--mamba_expand` arguments |

### Key design decisions & bug fixes

1. **Preserving Mamba's Δ initialization**: `initialize_weights()` now skips Mamba internal modules (`mamba_forward`, `mamba_backward`, `.mamba.mamba`). Mamba's `dt_proj.bias` uses carefully designed log-space initialization critical for SSM dynamics — the original `self.apply(_basic_init)` would overwrite this with zeros.

2. **Mask handling**: Padding positions are zeroed out before entering Mamba (`x = x * mask.unsqueeze(-1).float()`) to prevent SSM state pollution from padding tokens.

3. **Bidirectional combine mode**: Default changed from `'add'` (simple averaging) to `'concat'` (concatenate + linear projection) to preserve directional information from forward and backward Mamba passes.

4. **Unified d_state defaults**: All Mamba classes use `d_state=64` as default, consistent with CLI argument defaults.

5. **AdaLN compatibility**: The existing AdaLN (Adaptive Layer Normalization) mechanism is fully preserved — Mamba receives AdaLN-modulated input and the gated residual connection remains unchanged.

## Training

### Original MDGen training (without Mamba)

Commands similar to these were used to train the models presented in the paper.
```
# Forward simulation
python train.py --sim_condition --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data/ --num_frames 1000 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 --wandb --run_name [NAME]

# Interpolation / TPS
python train.py --tps_condition --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data/ --num_frames 100 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 --wandb --run_name [NAME]

# Upsampling
python train.py --sim_condition --train_split splits/4AA_implicit_train.csv --val_split splits/4AA_implicit_val.csv --data_dir data/4AA_data_implicit/ --num_frames 1000 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 20 --val_repeat 25 --cond_interval 100 --batch_size 8 --epochs 10000 --wandb --run_name [NAME]

# Inpainting / design
python train.py --inpainting --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv --data_dir data/4AA_data --num_frames 100 --prepend_ipa --abs_pos_emb --crop 4 --ckpt_freq 100 --val_repeat 25 --batch_size 32 --design --sampling_method euler --epochs 10000 --frame_interval 10 --no_aa_emb --no_torsion --wandb --run_name [NAME]

# ATLAS
python train.py --sim_condition --train_split splits/atlas_train.csv --val_split splits/atlas_val.csv --data_dir share/data_atlas/ --num_frames 250 --batch_size 1 --prepend_ipa --crop 256 --val_repeat 25 --epochs 10000 --atlas --ckpt_freq 10 --suffix _i40 --wandb --run_name [NAME]
```

### Training with Mamba

Add `--mamba` (and optionally `--bi_mamba`) to any training command to replace temporal attention with Mamba. All other arguments remain the same.

```
# Forward simulation with bidirectional Mamba (recommended)
python train.py --sim_condition --mamba --bi_mamba \
    --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
    --data_dir data/4AA_data/ --num_frames 1000 --prepend_ipa --abs_pos_emb \
    --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 \
    --wandb --run_name sim_mamba_bidir

# Forward simulation with unidirectional Mamba
python train.py --sim_condition --mamba \
    --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
    --data_dir data/4AA_data/ --num_frames 1000 --prepend_ipa --abs_pos_emb \
    --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 \
    --wandb --run_name sim_mamba_unidir

# TPS / Interpolation with Mamba
python train.py --tps_condition --mamba --bi_mamba \
    --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
    --data_dir data/4AA_data/ --num_frames 100 --prepend_ipa --abs_pos_emb \
    --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 \
    --wandb --run_name tps_mamba

# Custom Mamba hyperparameters
python train.py --sim_condition --mamba --bi_mamba \
    --mamba_d_state 128 --mamba_d_conv 8 --mamba_expand 4 \
    --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
    --data_dir data/4AA_data/ --num_frames 1000 --prepend_ipa --abs_pos_emb \
    --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 \
    --wandb --run_name sim_mamba_large_state

# Long trajectory (5000 frames) — where Mamba's O(n) complexity matters most
python train.py --sim_condition --mamba --bi_mamba \
    --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
    --data_dir data/4AA_data/ --num_frames 5000 --prepend_ipa --abs_pos_emb \
    --crop 4 --ckpt_freq 40 --val_repeat 25 --suffix _i100 --epochs 10000 \
    --wandb --run_name sim_mamba_long_traj
```

## Model weights

The model weights used in the paper may be downloaded here:
```
wget https://storage.googleapis.com/mdgen-public/weights/forward_sim.ckpt
wget https://storage.googleapis.com/mdgen-public/weights/interpolation.ckpt
wget https://storage.googleapis.com/mdgen-public/weights/upsampling.ckpt
wget https://storage.googleapis.com/mdgen-public/weights/inpainting.ckpt
wget https://storage.googleapis.com/mdgen-public/weights/atlas.ckpt
```

> **Note:** These are the original MDGen weights (without Mamba). Mamba-enabled models need to be trained from scratch using the commands above.

## Inference

Commands similar to these were used to obtain the samples analyzed in the paper.
```
# Forward simulation
python sim_inference.py --sim_ckpt forward_sim.ckpt --data_dir share/4AA_sims --split splits/4AA_test.csv --num_rollouts 10 --num_frames 1000 --xtc --out_dir [DIR]

# Interpolation / TPS
python tps_inference.py --sim_ckpt interpolation.ckpt --data_dir share/4AA_sims --split splits/4AA_test.csv --num_frames 100 --suffix _i100 --mddir data/4AA_sims  --out_dir /data/cb/scratch/share/results/0506_tps_1ns

# Upsampling
python upsampling_inference.py --ckpt upsampling.ckpt --split splits/4AA_implicit_test.csv --out_dir outpdb/0505_100ps_upsampling_3139 --batch_size 10 --xtc --out_dir [DIR]

# Inpainting / design for high flux transitions
python design_inference.py --sim_ckpt inpainting.ckpt --split splits/4AA_test.csv --data_dir data/4AA_data/ --num_frames 100 --mddir data/4AA_sims --random_start_idx --out_dir [DIR]

# Inpainting / design for random transitions
python design_inference.py --sim_ckpt inpainting.ckpt --split splits/4AA_test.csv --data_dir data/4AA_data/ --num_frames 100 --mddir data/4AA_sims --out_dir [DIR]

# ATLAS forward simulation # note no --xtc here!
python sim_inference.py --sim_ckpt atlas.ckpt --data_dir share/data_atlas/ --num_frames 250 --num_rollouts 1 --split splits/atlas_test.csv --suffix _R1 --out_dir [DIR]
```

## Analysis

We run analysis scripts that produce a pickle file in each sample directory.
```
# Forward simulation
python -m scripts.analyze_peptide_sim --mddir data/4AA_sims --pdbdir [DIR] --plot --save --num_workers 1

# Interpolation / TPS
python -m scripts.analyze_peptide_tps --mddir data/4AA_sims --data_dir data/4AA_sims  --pdbdir [DIR] --plot --save --num_workers 1 --outdir [DIR]

# Upsampling
python -m scripts.analyze_upsampling --mddir data/4AA_sims_implicit --pdbdir [DIR] --plot --save --num_workers 1

# Inpainting / design
python -m scripts.analyze_peptide_design --mddir data/4AA_sims --data_dir data/4AA_data --pdbdir [DIR]
```
To analyze the ATLAS rollouts, follow the instructions at https://github.com/bjing2016/alphaflow?tab=readme-ov-file#Evaluation-scripts.

Tables and figures in the paper are extracted from these pickle files.

## Roadmap

### Completed (current)
- **Method A — Temporal Mamba**: Replace `mha_t` with bidirectional Mamba-1, preserving `mha_l` and IPA unchanged
- Bug fixes: weight initialization protection, mask handling, d_state defaults, combine mode

### Planned
- **P2 — Upgrade to Mamba-2**: Replace `mamba_ssm.Mamba` with `mamba_ssm.Mamba2` for chunk-wise parallel SSD, larger native d_state (64-128), and `seq_idx` support for variable-length sequences
- **P3 — Method B — Spatial Mamba**: Add optional Mamba replacement for `mha_l` (spatial attention) to scale to proteins with L > 100 residues
- **P3 — Method C — Hybrid architecture**: Mix Mamba and Transformer layers (e.g., 3:1 ratio) to combine Mamba's linear complexity with Transformer's global information mixing
- **Ablation studies**: Systematic comparison of combine modes (concat vs gate vs add), d_state values, and Mamba vs Attention across trajectory lengths and residue counts

## License

MIT. Additional licenses may apply for third-party source code noted in file headers.

## Citation
```
@misc{jing2024generativemodelingmoleculardynamics,
      title={Generative Modeling of Molecular Dynamics Trajectories},
      author={Bowen Jing and Hannes Stärk and Tommi Jaakkola and Bonnie Berger},
      year={2024},
      eprint={2409.17808},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2409.17808},
}
```
