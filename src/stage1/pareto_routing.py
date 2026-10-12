"""Stage-1 Pareto domain routing (expression vs geometry distance).

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 0).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def domain_geometry_cost(X, Y):
    d_xy, _ = cKDTree(Y).query(X, k=1, workers=-1)
    d_yx, _ = cKDTree(X).query(Y, k=1, workers=-1)
    return float(0.5 * (np.median(d_xy) + np.median(d_yx)))


def pareto_front_2d(E, G):
    front = np.ones(E.shape[0], dtype=bool)
    for j in range(E.shape[0]):
        dominated = (E <= E[j]) & (G <= G[j]) & ((E < E[j]) | (G < G[j]))
        if np.any(dominated): front[j] = False
    return front


def build_pareto_supports(warped_source, zs, zt, source_leiden, target_leiden, target_aligned, target_components, target_piece_of_spot):
    src_labels, tgt_labels = pd.unique(source_leiden), pd.unique(target_leiden)
    src_members = [np.flatnonzero(source_leiden == a) for a in src_labels]
    tgt_members = [np.flatnonzero(target_leiden == b) for b in tgt_labels]
    src_centroids = np.vstack([zs[idx].mean(axis=0) for idx in src_members])
    tgt_centroids = np.vstack([zt[idx].mean(axis=0) for idx in tgt_members])
    E = np.linalg.norm(src_centroids[:, None, :] - tgt_centroids[None, :, :], axis=2)
    G = np.empty_like(E, dtype=np.float64)
    for ai, sidx in enumerate(src_members):
        X = warped_source[sidx]
        for bi, tidx in enumerate(tgt_members):
            G[ai, bi] = domain_geometry_cost(X, target_aligned[tidx])
    support_by_domain, rows = [], []
    for ai, sidx in enumerate(src_members):
        front = pareto_front_2d(E[ai], G[ai])
        selected = np.flatnonzero(front)
        touched_components = set()
        for bi in selected:
            tidx = tgt_members[bi]
            tree = cKDTree(target_aligned[tidx])
            _, local_idx = tree.query(warped_source[sidx], k=1, workers=-1)
            seeds = tidx[np.asarray(local_idx, dtype=np.int32)]
            touched_components.update(np.asarray(target_piece_of_spot[seeds], dtype=np.int32).tolist())
        blocks = [target_components[c] for c in sorted(touched_components)]
        support = np.unique(np.concatenate(blocks)).astype(np.int32) if blocks else np.empty(0, dtype=np.int32)
        support_by_domain.append(support)
        for bi in range(len(tgt_labels)):
            rows.append({
                "source_domain": str(src_labels[ai]), "target_domain": str(tgt_labels[bi]),
                "expression_distance": float(E[ai, bi]), "geometry_distance": float(G[ai, bi]),
                "pareto_selected": bool(front[bi]),
            })
    return {
        "source_labels": np.asarray(src_labels, dtype=str), "target_labels": np.asarray(tgt_labels, dtype=str),
        "source_members": src_members, "target_members": tgt_members, "expression_distance": E, "geometry_distance": G,
        "support_by_domain": support_by_domain, "routing_table": pd.DataFrame(rows),
    }
