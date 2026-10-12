# Equivalence of extracted source vs. reference notebooks

## What was extracted

The `src/` modules were extracted **programmatically** from the notebooks that
actually ran the experiments (function bodies copied verbatim, not rewritten):

| Module | Source notebook (cell) |
|---|---|
| `src/stage1/spatial_alignment.py` | `pairA&C_stage1&2.ipynb` cell 0 |
| `src/stage1/domain_clustering.py` | `pairA&C_stage1&2.ipynb` cell 0 |
| `src/stage1/pareto_routing.py` | `pairA&C_stage1&2.ipynb` cell 0 |
| `src/stage1/feasible_set.py` | `pairA&C_stage1&2.ipynb` cell 0 |
| `src/stage1/run.py` | `pairA&C_stage1&2.ipynb` cell 0 |
| `src/stage2/constrained_ot.py` | `pairA&C_stage1&2.ipynb` cell 3 |
| `src/evaluation/accuracy.py` | `cell_type_evaluation.ipynb` cells 0–2 |
| `src/evaluation/support_diagnostics.py` | re-export of `evaluate_support` |

## Code-level check (performed)

For every extracted function, the body was compared byte-for-byte (after
stripping trailing blank lines) against the original notebook cell:

- **38 / 38 Stage-1 + evaluation functions: source-identical** (AST-based
  `ast.get_source_segment` comparison, exact match after stripping trailing
  whitespace).
- **Stage 2 (`constrained_ot.py`): identical (669 lines)** to notebook cell 3,
  with exactly one change: `DCR_SCRIPT` repointed from
  `/mnt/disk2/lirui/baseline_scripts/run_domain_concentration_fgw.py` to the
  bundled `src/stage2/domain_concentration_fgw.py` (the same file, copied
  verbatim).
- Only structural changes made: module headers (docstrings), import
  statements, and the addition of the original module-level constants
  (`ROOT`, `PAIR_CONFIG`, `CURATED_PATH`, etc.). **No function body, parameter,
  objective, or constraint was modified.**
- All `src/` files pass `python -m py_compile`.

## What was NOT done

- **No end-to-end numerical re-run** was performed. Equivalence here means
  *code-level* identity, **not** that the extracted modules have been executed
  and reproduced the saved `hard_support.npz`, couplings, or evaluations.
- The `src/stage1/run.py` and `src/stage2/constrained_ot.py` orchestrators
  retain original absolute paths; they will not run out-of-the-box without
  (a) the input H5ADs and (b) path updates.

Therefore: **"code-identical extraction" is claimed; "full reproduction
verified" is not.** Update this file if/when a clean-environment re-run is
completed and the saved artifacts are reproduced.
