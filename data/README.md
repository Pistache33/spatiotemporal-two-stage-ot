# Data provenance

Raw data and large intermediate arrays are **not** committed to this
repository. This file records where each input came from and how it can be
regenerated.

## Input H5ADs (NBVAE-30 latents)

| Pair | Stage | File (original machine path) | Status |
|---|---|---|---|
| Pair C | E10.5 source | `mosta_e105_e115_nbvae30/data/E10.5_E2S1.nbvae30.h5ad` | not committed |
| Pair C | E11.5 target | `mosta_e105_e115_nbvae30/data/E11.5_E1S1.nbvae30.h5ad` | not committed |
| Pair A | E9.5/E10.5 joint | `mosta_flatvi_fgw/data/E9.5__E10.5.nbvae30_latents.h5ad` | not committed |
| Pair A | source/target split | `domain_concentration_repro/input/PairA_source_E9.5_E1S1.h5ad` + `..._target_E10.5_E2S1.h5ad` | **path no longer exists** — regenerate from the joint file above |

The `X_nbvae_30` obsm key holds 30-dimensional NBVAE latents; `spatial` holds
2-D spatial coordinates; `annotation` holds cell-type annotations.

## Upstream data

MOSTA mouse-embryo spatial transcriptomics:
https://db.cngb.org/stomics/mosta/ (or the paper's supplementary tables).
This repository does **not** redistribute MOSTA count matrices.

## Regeneration outline

1. Obtain the MOSTA sections for E9.5 / E10.5 / E11.5.
2. Extract NBVAE-30 latents (`X_nbvae_30`) — the baseline grid runner expects
   `obsm["X_nbvae_30"]`, `obsm["spatial"]`, and `obs["annotation"]`.
3. For Pair A, split the joint `E9.5__E10.5.nbvae30_latents.h5ad` into
   `E9.5_E1S1` (source) and `E10.5_E2S1` (target).

## Large outputs (regenerate, do not commit)

- `hard_support.npz` — produced by `src/stage1/run.py` (Stage 1).
- `*_transport_sparse.npz`, `*_top1_target_index.npy` — produced by
  `src/stage2/constrained_ot.py` (Stage 2).
- `transport.npy` (dense baselines) — produced by
  `baselines/moscot/05_run_fgw_grid_point.py` and the SOCS caller.

Only small summaries (CSV/JSON) and PNGs are committed under `results/` and
`figures/`.
