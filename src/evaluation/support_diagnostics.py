"""Support-diagnostics: feasible-set coverage / purity evaluation.

`evaluate_support` is defined verbatim in `src/stage1/feasible_set.py`; this
module re-exports it for a clean evaluation namespace. It computes per-annotation
feasible-set coverage and purity post hoc (annotation/curated used for
evaluation only, never for construction).
"""
from ..stage1.feasible_set import evaluate_support  # noqa: F401

__all__ = ["evaluate_support"]
