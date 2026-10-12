"""Stage-1 spatial alignment: contour alignment + thin-plate spline.

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 0). Algorithm, parameters,
objective and constraints are unchanged.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import RBFInterpolator
from ._constants import N_BOUNDARY_POINTS


def isotropic_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    center = X.mean(axis=0)
    Xc = X - center
    scale = np.sqrt(np.mean(np.sum(Xc ** 2, axis=1)))
    if not np.isfinite(scale) or scale <= 0: raise ValueError("degenerate spatial scale")
    return Xc / scale, {"center": center, "scale": scale}


def pca_frame(X):
    cov = np.cov(np.asarray(X, dtype=np.float64).T)
    eigval, eigvec = np.linalg.eigh(cov)
    eigvec = eigvec[:, np.argsort(eigval)[::-1]]
    return np.asarray(X) @ eigvec, eigvec


def signed_permutation_matrices():
    mats = []
    for perm in [(0, 1), (1, 0)]:
        P = np.eye(2)[list(perm)]
        for sx, sy in product([-1.0, 1.0], repeat=2):
            mats.append(np.diag([sx, sy]) @ P)
    return mats


def convex_boundary(X):
    return np.asarray(X[ConvexHull(X).vertices], dtype=np.float64)


def resample_closed_curve(polygon, n_points):
    polygon = np.asarray(polygon, dtype=np.float64)
    closed = np.vstack([polygon, polygon[0]])
    delta = np.diff(closed, axis=0)
    seglen = np.linalg.norm(delta, axis=1)
    good = seglen > 1e-12
    if not np.all(good):
        closed = closed[np.r_[good, True]]
        delta = np.diff(closed, axis=0)
        seglen = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0.0, np.cumsum(seglen)]
    total = cumulative[-1]
    query = np.linspace(0.0, total, int(n_points), endpoint=False)
    out = []
    for s in query:
        j = np.searchsorted(cumulative, s, side="right") - 1
        j = min(max(j, 0), len(seglen) - 1)
        u = (s - cumulative[j]) / seglen[j]
        out.append((1.0 - u) * closed[j] + u * closed[j + 1])
    return np.asarray(out)


def boundary_sample(X, n_points=N_BOUNDARY_POINTS):
    return resample_closed_curve(convex_boundary(X), n_points)


def symmetric_chamfer(A, B):
    tree_A, tree_B = cKDTree(A), cKDTree(B)
    d_AB, _ = tree_B.query(A, k=1, workers=-1)
    d_BA, _ = tree_A.query(B, k=1, workers=-1)
    return float(0.5 * (np.mean(d_AB ** 2) + np.mean(d_BA ** 2)))


def choose_orientation(src_pca, tgt_pca):
    tgt_boundary = boundary_sample(tgt_pca)
    best = None
    for M in signed_permutation_matrices():
        src_try = src_pca @ M.T
        src_boundary = boundary_sample(src_try)
        score = symmetric_chamfer(src_boundary, tgt_boundary)
        if best is None or score < best["score"]:
            best = {"score": score, "matrix": M.copy(), "src_aligned": src_try, "src_boundary": src_boundary}
    return best, tgt_boundary


def cyclic_boundary_match(source_boundary, target_boundary):
    if len(source_boundary) != len(target_boundary): raise ValueError("boundary sample sizes differ")
    best_score, best_shift, best_target = np.inf, None, None
    for shift in range(len(source_boundary)):
        Y = np.roll(target_boundary, shift, axis=0)
        score = float(np.mean(np.sum((source_boundary - Y) ** 2, axis=1)))
        if score < best_score:
            best_score, best_shift, best_target = score, shift, Y
    return best_target, int(best_shift), float(best_score)


def fit_bidirectional_tps(source_boundary, target_boundary):
    forward = RBFInterpolator(source_boundary, target_boundary, kernel="thin_plate_spline", smoothing=0.0, degree=1)
    backward = RBFInterpolator(target_boundary, source_boundary, kernel="thin_plate_spline", smoothing=0.0, degree=1)
    return forward, backward


def build_global_deformation(src_xy_raw, tgt_xy_raw):
    src_iso, src_iso_meta = isotropic_normalize(src_xy_raw)
    tgt_iso, tgt_iso_meta = isotropic_normalize(tgt_xy_raw)
    src_pca, src_basis = pca_frame(src_iso)
    tgt_pca, tgt_basis = pca_frame(tgt_iso)
    orientation, tgt_boundary = choose_orientation(src_pca, tgt_pca)
    src_aligned, src_boundary = orientation["src_aligned"], orientation["src_boundary"]
    tgt_boundary_matched, cyclic_shift, cyclic_mse = cyclic_boundary_match(src_boundary, tgt_boundary)
    forward_tps, backward_tps = fit_bidirectional_tps(src_boundary, tgt_boundary_matched)
    warped_source = np.asarray(forward_tps(src_aligned), dtype=np.float64)
    reconstructed_source = np.asarray(backward_tps(warped_source), dtype=np.float64)
    cycle_error = np.linalg.norm(reconstructed_source - src_aligned, axis=1)
    return {
        "source_aligned": src_aligned, "target_aligned": tgt_pca, "warped_source": warped_source,
        "cycle_error": cycle_error, "orientation_matrix": orientation["matrix"],
        "orientation_score": orientation["score"], "cyclic_shift": cyclic_shift, "cyclic_mse": cyclic_mse,
        "source_boundary": src_boundary, "target_boundary": tgt_boundary_matched,
        "source_iso_meta": src_iso_meta, "target_iso_meta": tgt_iso_meta,
        "source_pca_basis": src_basis, "target_pca_basis": tgt_basis,
    }
