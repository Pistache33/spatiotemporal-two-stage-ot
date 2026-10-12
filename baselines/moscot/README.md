# MOSCOT baseline

Copied verbatim from the original run directory
`baseline_scripts/mosta_flatvi_fgw/`. No run entry points were invented.

## Files

| File | Role |
|---|---|
| `09_make_nbvae30_fgw_grid.py` | build the NBVAE-30 grid TSV |
| `06_run_fgw_local.py` | loop over grid tasks, calling the per-point runner |
| `05_run_fgw_grid_point.py` | run + evaluate one grid point |
| `fgw_eval_common.py` | dense-coupling evaluation helpers |
| `07_summarize_flatvi_fgw.py` | summarize grid results |

## Actual run (Pair C)

```bash
python baselines/moscot/09_make_nbvae30_fgw_grid.py ...
python baselines/moscot/06_run_fgw_local.py \
    --grid configs/moscot_grid_pairC.tsv \
    --transitions configs/curated_transitions.tsv \
    --output-root <out>
```

Pair A used the same runner with
`configs/moscot_grid_pairA.tsv` (grid `grids/nbvae30.tsv` from the
`mosta_growth_marginal_ablation/no_growth_score` run).

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
