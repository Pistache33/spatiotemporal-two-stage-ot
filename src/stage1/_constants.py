"""Shared Stage-1 constants (verbatim values from the source notebook)."""

REP_KEY, SPATIAL_KEY, ANNOTATION_KEY = "X_nbvae_30", "spatial", "annotation"
LEIDEN_K, LEIDEN_SEED, LEIDEN_N_ITERATIONS = 15, 0, 2
SPATIAL_CC_FACTOR, N_BOUNDARY_POINTS = 1.5, 256
EVAL_DROP_LABELS = {"Cavity"}

from pathlib import Path
# Original curated-transitions location used in the source notebooks.
CURATED_PATH = Path(__file__).resolve().parents[2] / "configs" / "curated_transitions.tsv"
