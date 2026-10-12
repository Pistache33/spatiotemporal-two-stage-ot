# spatiotemporal-two-stage-ot

Learning biologically plausible correspondences in time-series spatial
transcriptomics via a two-stage, constrained optimal-transport pipeline.

This repository accompanies a computational study of cell-state correspondence
between consecutive time points of the mouse embryo (MOSTA):

- **Pair A**: E9.5_E1S1 → E10.5_E2S1
- **Pair C**: E10.5_E2S1 → E11.5_E1S1

It contains (1) the two-stage OT method, (2) the MOSCOT and SOCS baselines,
and (3) the diagnostics used to interpret where correspondence inference
succeeds and where it fails.

> **Positioning.** This is a study of **constrained correspondence inference
> and its diagnostics**, not a state-of-the-art accuracy benchmark. The
> two-stage method is not claimed to outperform MOSCOT/SOCS on overall
> accuracy (see [Results](#results)); its value is in making the feasible-set
> constraint explicit and in diagnosing *why* unconstrained baselines leak
> mass across organs.

---

## Method overview

**Stage 1** builds a hard feasible set (which target spots each source spot may
map to) without using annotation labels:

1. **Spatial alignment (spatial only)** — contour/orientation alignment +
   bidirectional thin-plate-spline (TPS) warping between source and target
   slices.
2. **Domain clustering (expression only)** — independent NBVAE-30 Leiden
   clustering of source and target.
3. **Pareto domain routing (expression + spatial, no weighted sum)** — each
   source Leiden cluster selects candidate target Leiden clusters on the 2-D
   Pareto front of (expression-centroid distance, geometry distance).
4. **Feasible set** — for each selected target Leiden, its spatially connected
   components are decomposed; components touched by nearest-neighbour seeds
   from the source cluster are kept. Their union is the hard feasible set.

**Stage 2** solves an exact-hard **unbalanced fused Gromov–Wasserstein (FGW)**
problem restricted to the feasible set: a sparse log-Sinkhorn solver with the
GW linearization from OTT, expression cost from NBVAE-30 Euclidean distances,
and spatial GW geometry on standardized raw coordinates.

### Information usage

| Information | Used in construction | Used in evaluation |
|---|---|---|
| Spatial (contour / TPS / components / NN seeds) | ✅ Stage 1 | — |
| Expression (NBVAE-30 Leiden / centroid distance) | ✅ Stage 1 + Stage 2 cost | — |
| Pareto selection (expression + geometry, no weighted sum) | ✅ Stage 1 | — |
| Annotation labels | ❌ | ✅ post-hoc |
| Curated transitions table | ❌ | ✅ post-hoc reference only |

The curated-transition table (`configs/curated_transitions.tsv`) is **only**
used as an evaluation reference. It does not participate in model
construction or parameter selection.

---

## Repository layout

```
configs/            grids, curated transitions, SOCS run configs
src/
  stage1/           spatial_alignment / domain_clustering / pareto_routing /
                    feasible_set / run  (extracted from notebooks)
  stage2/           constrained_ot (+ domain_concentration_fgw dependency)
  evaluation/       accuracy / support_diagnostics
baselines/
  moscot/           grid generator + runner + evaluation (copied verbatim)
  socs/             official SOCS caller + resume entry
notebooks/          reference notebooks (absolute paths preserved)
results/            small CSVs/JSONs (weighted-best summaries, per-organ tables)
figures/            representative PNGs
data/               data provenance (no raw data committed)
```

---

## Results

Curated-transition accuracy (weighted / macro), selected runs:

| Pair | Method | Weighted | Macro | Selection |
|---|---|---|---|---|
| PairA | Two-stage OT | **0.494** | 0.487 | src1_tgt0p6 (mass) |
| PairA | SOCS | 0.452 | 0.455 | task 6, α_moscot 0.4→α_socs 0.6, τ=(0.05,1) |
| PairA | MOSCOT | 0.432 | 0.394 | task 5, α=0.4, τ=(1.0,0.05) |
| PairC | MOSCOT | **0.433** | 0.423 | task 11, α=0.6, τ=(0.01,1.0) |
| PairC | SOCS | 0.394 | 0.383 | task 0, α_moscot 0.2→α_socs 0.8, τ=(1.0,0.01) |
| PairC | Two-stage OT | 0.384 | 0.428 | src1_tgt0p6 (mass) |

Honest reading: the two-stage method is the best *on Pair A* but is below the
MOSCOT baseline *on Pair C* on weighted accuracy. Per-organ results
(`results/*/stage1_stage2/stage2_by_annotation.csv`) show strong performance
for spatially concentrated, fate-restricted organs (Heart, Dermomyotome, Liver,
Brain, DRG) and weak performance for diffuse / multi-lineage mesenchyme.

---

## Baselines

### MOSCOT

- Actual run entry (no fabricated wrappers):
  ```bash
  python baselines/moscot/09_make_nbvae30_fgw_grid.py ...   # build grid TSV
  python baselines/moscot/06_run_fgw_local.py \
      --grid configs/moscot_grid_pairC.tsv \
      --transitions configs/curated_transitions.tsv \
      --output-root <out>
  ```
  (`06_run_fgw_local.py` loops over tasks, calling `05_run_fgw_grid_point.py`.)
- Embedding: **`X_nbvae_30`** (NBVAE-30 latents); Pair C slices
  E10.5_E2S1 → E11.5_E1S1, Pair A slices E9.5_E1S1 → E10.5_E2S1.
- Grid: α ∈ {0.2, 0.4, 0.6} × τ ∈ {(1,.01),(1,.05),(.05,1),(.01,1)}, ε=1e-4,
  no growth marginals.
- Evaluation: curated-transition weighted / macro accuracy (self + known
  descendants).
- **Best-run selection rule**: maximum weighted accuracy; tie-break by task id.
- MOSCOT `alpha` is the **GW (quadratic) weight**.

### SOCS

- Actual run entry (official implementation call + resume):
  ```bash
  python baselines/socs/run_official.py --pairs PairA ...   # Pair A
  python baselines/socs/resume_pairc.py                      # Pair C resume
  ```
- Third-party implementation: [SOCS-main](https://github.com/algo-bio-lab/SOCS-main),
  MIT License (c) 2025 algo-bio-lab. Reproduced here via the caller script; the
  SOCS source itself is **not** vendored.
- **Alpha convention**: SOCS `alpha` is the *expression* weight, the opposite
  of MOSCOT. The caller maps `alpha_socs = 1 − alpha_moscot`
  (`--alpha-mode match_moscot`). τ is converted as `rho = ε·τ/(1−τ)`
  (τ=1 ⇒ ρ=∞).
- Embedding: `X_nbvae_30`; same slices and resolutions as MOSCOT.
- Best-run selection rule: maximum weighted accuracy.

### Third-party attribution

- **MOSCOT** (BSD-3): https://github.com/ComputationalCellBiology/moscot
- **OTT-JAX** (Apache-2.0): https://github.com/ott-jax/ott
- **SOCS** (MIT): https://github.com/algo-bio-lab/SOCS-main
- **Scanpy / AnnData / JAX**: see `environment.yml`.

---


## Data

See `data/README.md`. Raw H5ADs and large coupling arrays (`*.npy`, `*.npz`) are
**not** committed. Data sources and regeneration steps are documented there.

---

## Known limitations

- Single Leiden seed (`seed=0`); seed-robustness not characterized.
- **Pair C over-concentrated coupling (Brain / DRG).** On Pair C the Stage-2
  hard-feasible coupling for **Brain** and **dorsal root ganglion (DRG)**
  deposits most of its transported mass on a small number of target spots
  (coupling collapse / over-concentration), even though per-organ top-1
  curated accuracy remains high (Brain 0.87, DRG 0.87 in
  `results/PairC/stage1_stage2/stage2_by_annotation.csv`). This
  over-concentration is a known failure mode of the exact-hard unbalanced FGW
  on these organs and is reported here rather than hidden.
- Pair A Stage-1 input path (`domain_concentration_repro/input/`) no longer
  exists on the original machine; the NBVAE-30 latent file
  `mosta_flatvi_fgw/data/E9.5__E10.5.nbvae30_latents.h5ad` is the canonical
  input and the source/target split must be regenerated from it.
- Notebooks contain absolute paths (kept for fidelity); update before running.
- `src1_tgt1p4` variants are not presented as full Stage-2 results (see
  `results/*/stage1_stage2/` note).

---

## AI assistance

- `figures/method_overview.png` is an **AI-generated schematic** used for
  illustration only; all quantitative results elsewhere in this repository are
  produced by the code and data contained here.
- Parts of the code in `src/` and `baselines/` were **drafted with the help of
  an AI coding assistant. The method design,
  parameters, and scientific conclusions are the author's own.

---

## License

See `LICENSE`. Third-party components retain their own licenses.
