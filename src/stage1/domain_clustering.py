"""Stage-1 domain clustering: NBVAE30 Leiden + target spatial components.

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 0).
"""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ._constants import (REP_KEY, SPATIAL_KEY, LEIDEN_K, LEIDEN_SEED,
                         LEIDEN_N_ITERATIONS)


def get_xy(adata):
    if SPATIAL_KEY not in adata.obsm: raise KeyError(f"{SPATIAL_KEY!r} missing")
    X = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2: raise ValueError(f"bad spatial shape={X.shape}")
    return X[:, :2]


def get_z(adata):
    if REP_KEY not in adata.obsm: raise KeyError(f"{REP_KEY!r} missing")
    Z = np.asarray(adata.obsm[REP_KEY], dtype=np.float32)
    if not np.isfinite(Z).all(): raise ValueError(f"non-finite {REP_KEY}")
    return Z


def run_leiden(adata, resolution):
    Z = get_z(adata)
    tmp = ad.AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32))
    tmp.obsm["X_use"] = Z
    sc.pp.neighbors(tmp, use_rep="X_use", n_neighbors=min(LEIDEN_K, max(2, adata.n_obs - 1)), random_state=LEIDEN_SEED)
    sc.tl.leiden(tmp, resolution=float(resolution), key_added="leiden", random_state=LEIDEN_SEED, n_iterations=LEIDEN_N_ITERATIONS)
    return tmp.obs["leiden"].astype(str).to_numpy()


def median_1nn(xy):
    d, _ = cKDTree(xy).query(xy, k=2, workers=-1)
    d = d[:, 1]
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0: raise ValueError("cannot estimate median 1NN")
    return float(np.median(d))


def query_pairs_array(tree, radius):
    try:
        pairs = tree.query_pairs(radius, output_type="ndarray")
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    except TypeError:
        pairs = list(tree.query_pairs(radius))
        return np.empty((0, 2), dtype=np.int64) if not pairs else np.asarray(pairs, dtype=np.int64)


def leiden_spatial_components(labels, xy, radius):
    labels = np.asarray(labels, dtype=str)
    piece_of_spot = np.full(len(labels), -1, dtype=np.int32)
    pieces, piece_leiden = [], []
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) == 1:
            cid = len(pieces); pieces.append(idx.astype(np.int32)); piece_leiden.append(str(lab)); piece_of_spot[idx] = cid; continue
        pairs = query_pairs_array(cKDTree(xy[idx]), radius)
        if len(pairs):
            rows, cols = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]
            graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(idx), len(idx)))
        else:
            graph = csr_matrix((len(idx), len(idx)), dtype=np.uint8)
        n_comp, cc = connected_components(graph, directed=False, return_labels=True)
        for local_c in range(n_comp):
            spots = idx[cc == local_c]
            cid = len(pieces); pieces.append(spots.astype(np.int32)); piece_leiden.append(str(lab)); piece_of_spot[spots] = cid
    if np.any(piece_of_spot < 0): raise RuntimeError("some spots lack component assignment")
    return {"pieces": pieces, "piece_of_spot": piece_of_spot, "piece_leiden": np.asarray(piece_leiden, dtype=str)}
