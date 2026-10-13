# MOSCOT baseline

MOSCOT is run as a full-rank fused Gromov–Wasserstein (FGW) baseline on the
NBVAE-30 embeddings. This directory contains the grid generator, runner, and
evaluation scripts; MOSCOT itself is installed from PyPI.

## How it is run

1. **Build the grid** — `09_make_nbvae30_fgw_grid.py` writes the 12-point grid
   TSV (`configs/moscot_grid_pairA.tsv` / `configs/moscot_grid_pairC.tsv`).
2. **Run the grid** — `06_run_fgw_local.py` loops over grid tasks and calls
   `05_run_fgw_grid_point.py` for each point, solving one FGW problem and
   evaluating the dense coupling.
3. **Summarize** — `07_summarize_flatvi_fgw.py` aggregates the grid results;
   the weighted-best run is selected.

```bash
python baselines/moscot/06_run_fgw_local.py \
    --grid configs/moscot_grid_pairC.tsv \
    --transitions configs/curated_transitions.tsv \
    --output-root <out>
```

## Files

| File | Role |
|---|---|
| `09_make_nbvae30_fgw_grid.py` | build the NBVAE-30 grid TSV |
| `06_run_fgw_local.py` | loop over grid tasks, calling the per-point runner |
| `05_run_fgw_grid_point.py` | run + evaluate one grid point |
| `fgw_eval_common.py` | dense-coupling evaluation helpers |
| `07_summarize_flatvi_fgw.py` | summarize grid results |

## Parameters

- Embedding: `X_nbvae_30`; cost `euclidean`, `scale_cost=max_cost`.
- Grid: α ∈ {0.2, 0.4, 0.6} × τ ∈ {(1,.01),(1,.05),(.05,1),(.01,1)}.
- ε = 1e-4, rank = -1 (full rank), no growth marginals.
- `alpha` = GW (quadratic) weight in MOSCOT.

## Best-run selection rule

Maximum `weighted_accuracy` over successful tasks; tie-break by task id.

## License

MOSCOT is BSD-3-Clause. This directory only contains our runner/evaluation
scripts; MOSCOT is installed from PyPI.
