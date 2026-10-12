#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-step domain-concentration refinement for moscot FGW.

Pipeline
--------
1. Load source/target h5ad.
2. Use obsm[feature_key] directly (preprocess=False by default).
3. Build Euclidean cross-feature cost C0.
4. Obtain baseline mouse-off FGW coupling P0 (solve or load existing .npy).
5. Build Leiden domains independently in source/target.
6. Aggregate P0 to domain transport M and row-normalize to p.
7. Compute H_ab = 2 * (max_b p_ab - p_ab).
8. Lift H to cell pairs: C1_ij = C0_ij + lambda * H[g(i), h(j)].
9. Re-solve FGW once with exactly the same solver parameters.
10. Optionally evaluate weighted/macro curated accuracy.

Important
---------
- Mouse growth prior is explicitly OFF: prepare(..., a=False, b=False).
- Annotation is never used to construct the refinement.
- One run = one FGW parameter setting + one refinement configuration.
- Dense cost matrices are temporary memmaps and deleted by default.
- Dense transports are saved only when explicitly requested.

Current successful defaults
---------------------------
FGW (Pair A): alpha=.4, epsilon=1e-4, tau_a=1, tau_b=.05, rank=-1.
Domain: source resolution=.725, target resolution=.95, n_neighbors=15.
Refinement: lambda=.05.

Exact Pair-A ~48.10% weighted run used:
  --source-resolutions 0.725 0.875
  --target-resolutions 0.95
  --lambda-concentration 0.05
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def log(msg: str) -> None:
    print(msg, flush=True)


def jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path: Path, obj: Mapping) -> None:
    path.write_text(json.dumps(jsonable(dict(obj)), indent=2, ensure_ascii=False), encoding="utf-8")


def supported_kwargs(fn, requested: Mapping) -> Tuple[dict, dict]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(requested), {}
    keep = {k: v for k, v in requested.items() if k in sig.parameters}
    drop = {k: v for k, v in requested.items() if k not in sig.parameters}
    return keep, drop


def rtag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-step label-free domain-concentration refinement of moscot FGW.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Input / output")
    g.add_argument("--source-h5ad", type=Path, required=True)
    g.add_argument("--target-h5ad", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--feature-key", default="X_nbvae_30")
    g.add_argument("--spatial-key", default="spatial")
    g.add_argument("--annotation-key", default="annotation", help="Post-hoc evaluation only.")

    g = p.add_argument_group("Optional preprocessing")
    g.add_argument("--preprocess", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--preprocess-n-pcs", type=int, default=30)
    g.add_argument("--preprocess-n-hvg", type=int, default=3000, help="<=0 disables HVG selection.")
    g.add_argument("--preprocess-target-sum", type=float, default=1e4)

    g = p.add_argument_group("Leiden domains")
    g.add_argument("--source-resolutions", type=float, nargs="+", default=[0.725])
    g.add_argument("--target-resolutions", type=float, nargs="+", default=[0.95])
    g.add_argument("--domain-n-neighbors", type=int, default=15)
    g.add_argument("--leiden-seed", type=int, default=0)
    g.add_argument("--leiden-n-iterations", type=int, default=2)

    g = p.add_argument_group("Concentration refinement")
    g.add_argument("--lambda-concentration", type=float, default=0.05)
    g.add_argument("--cost-chunk-rows", type=int, default=128)

    g = p.add_argument_group("Baseline transport reuse")
    g.add_argument("--baseline-transport", type=Path, default=None)
    g.add_argument("--baseline-source-names", type=Path, default=None)
    g.add_argument("--baseline-target-names", type=Path, default=None)
    g.add_argument("--save-baseline-transport", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--save-refined-transport", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument(
        "--baseline-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Solve/evaluate only P0. Useful for storage-light alpha/tau grid searches.",
    )

    g = p.add_argument_group("moscot FGW parameters")
    g.add_argument("--alpha", type=float, default=0.4)
    g.add_argument("--epsilon", type=float, default=1e-4)
    g.add_argument("--tau-a", type=float, default=1.0)
    g.add_argument("--tau-b", type=float, default=0.05)
    g.add_argument("--rank", type=int, default=-1)
    g.add_argument("--scale-cost", default="max_cost")
    g.add_argument("--min-iterations", type=int, default=50)
    g.add_argument("--max-iterations", type=int, default=1500)
    g.add_argument("--threshold", type=float, default=0.01)
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--inner-min-iterations", type=int, default=20)
    g.add_argument("--inner-max-iterations", type=int, default=500)
    g.add_argument("--inner-threshold", type=float, default=0.01)
    g.add_argument("--inner-iterations", type=int, default=10)

    g = p.add_argument_group("Optional curated evaluation")
    g.add_argument("--curated-transitions", type=Path, default=None)
    g.add_argument("--curated-source-column", default="cell_type")
    g.add_argument("--curated-descendants-column", default="known_descendants")
    g.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)

    g = p.add_argument_group("Temporary files")
    g.add_argument("--temp-dir", type=Path, default=None)
    g.add_argument("--keep-temporary-costs", action=argparse.BooleanOptionalAction, default=False)

    return p


# =============================================================================
# Data / preprocessing
# =============================================================================


def read_inputs(source_path: Path, target_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    log(f"[load] source: {source_path}")
    source = sc.read_h5ad(source_path)
    log(f"[load] target: {target_path}")
    target = sc.read_h5ad(target_path)
    if source.obs_names.has_duplicates or target.obs_names.has_duplicates:
        raise ValueError("source/target obs_names must be unique")
    return source, target


def joint_pca_preprocess(source, target, *, output_key, n_pcs, n_hvg, target_sum):
    """Optional shared PCA. OFF by default."""
    common = source.var_names.intersection(target.var_names)
    if len(common) < 2:
        raise ValueError(f"Only {len(common)} common genes")
    a = source[:, common].copy()
    b = target[:, common].copy()
    ns = a.n_obs
    joint = ad.concat([a, b], join="inner", merge="same", index_unique="__")
    sc.pp.filter_genes(joint, min_cells=3)
    sc.pp.normalize_total(joint, target_sum=float(target_sum))
    sc.pp.log1p(joint)
    if 0 < int(n_hvg) < joint.n_vars:
        sc.pp.highly_variable_genes(joint, n_top_genes=int(n_hvg), flavor="seurat", subset=True)
    n_eff = min(int(n_pcs), joint.n_obs - 1, joint.n_vars - 1)
    if n_eff < 2:
        raise ValueError(f"Cannot run PCA: n_components={n_eff}")
    sc.pp.pca(joint, n_comps=n_eff, zero_center=True, svd_solver="arpack")
    z = np.asarray(joint.obsm["X_pca"], dtype=np.float32)
    source.obsm[output_key] = z[:ns].copy()
    target.obsm[output_key] = z[ns:].copy()
    log(f"[preprocess] joint PCA -> obsm[{output_key!r}], dim={n_eff}")


def validate_arrays(source, target, *, feature_key, spatial_key):
    for side, x in (("source", source), ("target", target)):
        if feature_key not in x.obsm:
            raise KeyError(f"{side}.obsm lacks {feature_key!r}; available={list(x.obsm.keys())}")
        if spatial_key not in x.obsm:
            raise KeyError(f"{side}.obsm lacks {spatial_key!r}; available={list(x.obsm.keys())}")
    zs = np.asarray(source.obsm[feature_key], dtype=np.float32)
    zt = np.asarray(target.obsm[feature_key], dtype=np.float32)
    ss = np.asarray(source.obsm[spatial_key], dtype=np.float32)
    st = np.asarray(target.obsm[spatial_key], dtype=np.float32)
    if zs.ndim != 2 or zt.ndim != 2 or zs.shape[1] != zt.shape[1]:
        raise ValueError(f"Feature shapes incompatible: {zs.shape} vs {zt.shape}")
    for name, arr in (("zs", zs), ("zt", zt), ("ss", ss), ("st", st)):
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {name}: shape={arr.shape}")
    return zs, zt, ss, st


def make_pair_adata(source, target, zs, zt, ss, st, *, feature_key, spatial_key):
    """Minimal AnnData: no shared gene matrix is required when preprocess=False."""
    ns, nt = source.n_obs, target.n_obs
    pair = ad.AnnData(X=np.zeros((ns + nt, 1), dtype=np.float32))
    src_internal = np.asarray([f"source::{x}" for x in source.obs_names.astype(str)])
    tgt_internal = np.asarray([f"target::{x}" for x in target.obs_names.astype(str)])
    pair.obs_names = np.concatenate([src_internal, tgt_internal])
    pair.obs["_dcr_time"] = np.concatenate([
        np.zeros(ns, dtype=np.float32),
        np.ones(nt, dtype=np.float32),
    ])
    pair.obsm[feature_key] = np.vstack([zs, zt]).astype(np.float32)
    pair.obsm[spatial_key] = np.vstack([ss, st]).astype(np.float32)
    return pair, src_internal, tgt_internal


# =============================================================================
# Leiden
# =============================================================================


def run_leiden(data, *, feature_key, resolution, n_neighbors, seed, n_iterations, graph_key, label_key):
    if graph_key not in data.uns:
        sc.pp.neighbors(data, n_neighbors=int(n_neighbors), use_rep=feature_key, key_added=graph_key)
    try:
        sc.tl.leiden(
            data,
            resolution=float(resolution),
            neighbors_key=graph_key,
            key_added=label_key,
            random_state=int(seed),
            flavor="igraph",
            n_iterations=int(n_iterations),
            directed=False,
        )
    except Exception:
        sc.tl.leiden(
            data,
            resolution=float(resolution),
            neighbors_key=graph_key,
            key_added=label_key,
            random_state=int(seed),
        )
    return data.obs[label_key].astype(str).to_numpy()


def build_partitions(source, target, args):
    src_parts: Dict[float, np.ndarray] = {}
    tgt_parts: Dict[float, np.ndarray] = {}
    graph_key = "_dcr_graph"
    for r in args.source_resolutions:
        labels = run_leiden(
            source, feature_key=args.feature_key, resolution=r,
            n_neighbors=args.domain_n_neighbors, seed=args.leiden_seed,
            n_iterations=args.leiden_n_iterations, graph_key=graph_key,
            label_key=f"_dcr_src_r{rtag(r)}",
        )
        src_parts[float(r)] = labels
        log(f"[domains] source resolution={r:g}: K={len(np.unique(labels))}")
    for r in args.target_resolutions:
        labels = run_leiden(
            target, feature_key=args.feature_key, resolution=r,
            n_neighbors=args.domain_n_neighbors, seed=args.leiden_seed,
            n_iterations=args.leiden_n_iterations, graph_key=graph_key,
            label_key=f"_dcr_tgt_r{rtag(r)}",
        )
        tgt_parts[float(r)] = labels
        log(f"[domains] target resolution={r:g}: K={len(np.unique(labels))}")
    return src_parts, tgt_parts


def save_partitions(outdir, source, target, src_parts, tgt_parts):
    s = pd.DataFrame({"obs_name": source.obs_names.astype(str)})
    t = pd.DataFrame({"obs_name": target.obs_names.astype(str)})
    for r, labels in src_parts.items():
        s[f"leiden_r{rtag(r)}"] = labels
    for r, labels in tgt_parts.items():
        t[f"leiden_r{rtag(r)}"] = labels
    s.to_csv(outdir / "source_domains.csv", index=False)
    t.to_csv(outdir / "target_domains.csv", index=False)


# =============================================================================
# Dense Euclidean C0
# =============================================================================


def build_c0(path: Path, zs: np.ndarray, zt: np.ndarray, chunk_rows: int):
    ns, nt = zs.shape[0], zt.shape[0]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(ns, nt))
    zs2 = np.sum(zs * zs, axis=1)
    zt2 = np.sum(zt * zt, axis=1)
    raw_max = 0.0
    raw_min = math.inf
    for start in range(0, ns, int(chunk_rows)):
        stop = min(start + int(chunk_rows), ns)
        d2 = zs2[start:stop, None] + zt2[None, :] - 2.0 * (zs[start:stop] @ zt.T)
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        out[start:stop] = d2.astype(np.float32, copy=False)
        raw_max = max(raw_max, float(d2.max()))
        raw_min = min(raw_min, float(d2.min()))
        if start == 0 or stop == ns or start % 1024 == 0:
            log(f"[cost] rows {start}:{stop}/{ns}")
    if not np.isfinite(raw_max) or raw_max <= 0:
        raise ValueError(f"Invalid C0 maximum={raw_max}")
    for start in range(0, ns, int(chunk_rows)):
        stop = min(start + int(chunk_rows), ns)
        out[start:stop] /= raw_max
    out.flush()
    stats = {
        "shape": [ns, nt],
        "raw_min": raw_min,
        "raw_max": raw_max,
        "normalized_min": raw_min / raw_max,
        "normalized_max": 1.0,
    }
    log(f"[cost] C0 normalized: min={stats['normalized_min']:.6f}, max=1.000000")
    return out, stats


# =============================================================================
# moscot
# =============================================================================


def get_subproblem(problem):
    for key in ((0.0, 1.0), ("0.0", "1.0")):
        try:
            return key, problem[key]
        except Exception:
            pass
    keys = list(problem.problems.keys())
    if len(keys) == 1:
        key = keys[0]
        return key, problem.problems[key]
    raise KeyError(f"Expected one source->target subproblem; found {keys}")


def get_solution(problem, key):
    if key in problem.solutions:
        return problem.solutions[key]
    if len(problem.solutions) == 1:
        return next(iter(problem.solutions.values()))
    raise KeyError(f"Cannot find solution for {key}")


def extract_transport(solution):
    x = getattr(solution, "transport_matrix", None)
    if callable(x):
        x = x()
    if x is None:
        x = getattr(solution, "matrix", None)
        if callable(x):
            x = x()
    if x is None:
        raise AttributeError("moscot solution has no transport matrix")
    p = np.asarray(x, dtype=np.float32)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError(f"Invalid transport: shape={p.shape}")
    return p


def solve_fgw(pair, xy_cost, src_internal, tgt_internal, args):
    from moscot.problems.spatiotemporal import SpatioTemporalProblem

    problem = SpatioTemporalProblem(adata=pair)
    prepare_requested = {
        "time_key": "_dcr_time",
        "spatial_key": args.spatial_key,
        "joint_attr": args.feature_key,
        "cost": "euclidean",
        "normalize_spatial": True,
        "policy": "sequential",
        "a": False,
        "b": False,
    }
    prepare_kwargs, prepare_dropped = supported_kwargs(problem.prepare, prepare_requested)
    if "a" in prepare_dropped or "b" in prepare_dropped:
        raise TypeError("Installed moscot cannot explicitly accept a=False,b=False; refusing ambiguous mouse-off run")
    problem = problem.prepare(**prepare_kwargs)
    sub_key, sub = get_subproblem(problem)

    actual_src = sub.adata_src.obs_names.astype(str).to_numpy()
    actual_tgt = sub.adata_tgt.obs_names.astype(str).to_numpy()
    if not np.array_equal(actual_src, src_internal):
        raise ValueError("moscot source order changed")
    if not np.array_equal(actual_tgt, tgt_internal):
        raise ValueError("moscot target order changed")

    sub.set_xy(
        pd.DataFrame(xy_cost, index=src_internal, columns=tgt_internal, copy=False),
        tag="cost_matrix",
    )

    linear_solver_kwargs = {
        "min_iterations": int(args.inner_min_iterations),
        "max_iterations": int(args.inner_max_iterations),
        "threshold": float(args.inner_threshold),
        "inner_iterations": int(args.inner_iterations),
    }
    requested = {
        "alpha": float(args.alpha),
        "epsilon": float(args.epsilon),
        "tau_a": float(args.tau_a),
        "tau_b": float(args.tau_b),
        "rank": int(args.rank),
        "scale_cost": str(args.scale_cost),
        "min_iterations": int(args.min_iterations),
        "max_iterations": int(args.max_iterations),
        "threshold": float(args.threshold),
        "batch_size": int(args.batch_size),
        "linear_solver_kwargs": linear_solver_kwargs,
    }
    solve_kwargs, solve_dropped = supported_kwargs(problem.solve, requested)
    essential = {"alpha", "epsilon", "tau_a", "tau_b", "rank", "scale_cost", "min_iterations", "max_iterations", "threshold"}
    bad = essential.intersection(solve_dropped)
    if bad:
        raise TypeError(f"moscot dropped essential solve kwargs: {sorted(bad)}")

    t0 = time.time()
    problem = problem.solve(**solve_kwargs)
    elapsed = time.time() - t0
    solution = get_solution(problem, sub_key)
    transport = extract_transport(solution)
    if transport.shape != (len(src_internal), len(tgt_internal)):
        raise ValueError(f"Unexpected transport shape={transport.shape}")
    info = {
        "elapsed_seconds": elapsed,
        "converged": bool(getattr(solution, "converged", False)),
        "solver_cost": float(getattr(solution, "cost")) if getattr(solution, "cost", None) is not None else None,
        "prepare_kwargs_applied": prepare_kwargs,
        "prepare_kwargs_dropped": prepare_dropped,
        "solve_kwargs_applied": solve_kwargs,
        "solve_kwargs_dropped": solve_dropped,
    }
    return transport, info


# =============================================================================
# Existing baseline P0
# =============================================================================


def read_names(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=str)).astype(str)


def load_existing_p0(path, source, target, src_names_path=None, tgt_names_path=None):
    if not path.exists():
        raise FileNotFoundError(path)
    p = np.load(path, mmap_mode="r")
    if src_names_path is None and tgt_names_path is None:
        if p.shape != (source.n_obs, target.n_obs):
            raise ValueError(f"P0 shape={p.shape} != {(source.n_obs, target.n_obs)}")
        log("[baseline] no name files supplied; assuming P0 already follows h5ad row order")
        return np.asarray(p, dtype=np.float32)
    if src_names_path is None or tgt_names_path is None:
        raise ValueError("Supply both baseline name files or neither")
    saved_s = read_names(src_names_path)
    saved_t = read_names(tgt_names_path)
    if p.shape != (len(saved_s), len(saved_t)):
        raise ValueError("P0 shape does not match saved name files")
    def match_current_names(data, saved_names):
        current = data.obs_names.astype(str)
        idx = pd.Index(saved_names).get_indexer(current)
        if np.all(idx >= 0):
            return idx
        if "original_obs_name" in data.obs:
            original = data.obs["original_obs_name"].astype(str).to_numpy()
            idx = pd.Index(saved_names).get_indexer(original)
            if np.all(idx >= 0):
                return idx
        return idx

    si = match_current_names(source, saved_s)
    ti = match_current_names(target, saved_t)
    if np.any(si < 0) or np.any(ti < 0):
        raise KeyError("Current h5ad names are not fully represented in saved P0 name files")
    log("[baseline] reordering P0 to current h5ad order using saved names")
    return np.asarray(p[np.ix_(si, ti)], dtype=np.float32)


# =============================================================================
# Domain concentration
# =============================================================================


def domain_penalty(p0, src_labels, tgt_labels):
    src_names, src_codes = np.unique(np.asarray(src_labels).astype(str), return_inverse=True)
    tgt_names, tgt_codes = np.unique(np.asarray(tgt_labels).astype(str), return_inverse=True)
    m = np.zeros((len(src_names), len(tgt_names)), dtype=np.float64)
    for a in range(len(src_names)):
        ii = np.flatnonzero(src_codes == a)
        target_mass = np.asarray(p0[ii, :], dtype=np.float64).sum(axis=0)
        m[a] = np.bincount(tgt_codes, weights=target_mass, minlength=len(tgt_names))
    row_mass = m.sum(axis=1, keepdims=True)
    prob = np.divide(m, row_mass, out=np.zeros_like(m), where=row_mass > 0)
    h = 2.0 * (prob.max(axis=1, keepdims=True) - prob)
    return {
        "source_names": src_names,
        "target_names": tgt_names,
        "source_codes": src_codes.astype(np.int32),
        "target_codes": tgt_codes.astype(np.int32),
        "probability": prob,
        "penalty": h.astype(np.float32),
    }


def save_penalty(outdir, sr, tr, obj):
    tag = f"source_r{rtag(sr)}__target_r{rtag(tr)}"
    pd.DataFrame(obj["probability"], index=obj["source_names"], columns=obj["target_names"]).to_csv(
        outdir / f"domain_probability__{tag}.csv"
    )
    pd.DataFrame(obj["penalty"], index=obj["source_names"], columns=obj["target_names"]).to_csv(
        outdir / f"domain_penalty__{tag}.csv"
    )


def build_refined_cost(path, c0, penalties, *, lam, chunk_rows):
    ns, nt = c0.shape
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(ns, nt))
    mn, mx = math.inf, -math.inf
    for start in range(0, ns, int(chunk_rows)):
        stop = min(start + int(chunk_rows), ns)
        hcell = np.zeros((stop - start, nt), dtype=np.float32)
        for obj in penalties:
            hcell += obj["penalty"][
                obj["source_codes"][start:stop, None],
                obj["target_codes"][None, :],
            ]
        hcell /= float(len(penalties))
        block = np.asarray(c0[start:stop], dtype=np.float32) + float(lam) * hcell
        out[start:stop] = block
        mn = min(mn, float(block.min()))
        mx = max(mx, float(block.max()))
    out.flush()
    log(f"[refine] C1 range=[{mn:.6f}, {mx:.6f}], lambda={lam:g}, components={len(penalties)}")
    return out, {"min": mn, "max": mx, "lambda": float(lam), "n_components": len(penalties)}


# =============================================================================
# Curated evaluation (optional)
# =============================================================================


def split_descendants(value):
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "/":
        return []
    return [x.strip() for x in re.split(r"[,;|\n]+", text) if x.strip() and x.strip() != "/"]


def load_allowed(path, source_col, descendants_col, include_self):
    table = pd.read_csv(path, sep=None, engine="python")
    if source_col not in table.columns or descendants_col not in table.columns:
        raise KeyError(f"Curated columns={list(table.columns)}")
    allowed = {}
    for _, row in table.iterrows():
        src = str(row[source_col]).strip()
        if not src or src.lower() == "nan":
            continue
        targets = set(split_descendants(row[descendants_col]))
        if include_self:
            targets.add(src)
        allowed[src] = targets
    return allowed


def evaluate_curated(p, source_annotation, target_annotation, allowed):
    s_names, s_codes = np.unique(np.asarray(source_annotation).astype(str), return_inverse=True)
    t_names, t_codes = np.unique(np.asarray(target_annotation).astype(str), return_inverse=True)
    mass = np.zeros((len(s_names), len(t_names)), dtype=np.float64)
    for a in range(len(s_names)):
        ii = np.flatnonzero(s_codes == a)
        target_mass = np.asarray(p[ii, :], dtype=np.float64).sum(axis=0)
        mass[a] = np.bincount(t_codes, weights=target_mass, minlength=len(t_names))
    rows = []
    correct_total = 0.0
    total = float(mass.sum())
    for a, src in enumerate(s_names):
        row_total = float(mass[a].sum())
        permitted = allowed.get(str(src), set())
        correct = sum(float(mass[a, b]) for b, tgt in enumerate(t_names) if str(tgt) in permitted)
        correct_total += correct
        rows.append({
            "source_annotation": str(src),
            "total_mass": row_total,
            "correct_mass": correct,
            "accuracy": correct / row_total if row_total > 0 else np.nan,
        })
    per_source = pd.DataFrame(rows)
    return {
        "weighted_accuracy": correct_total / total if total > 0 else np.nan,
        "macro_accuracy": float(per_source["accuracy"].dropna().mean()),
        "per_source": per_source,
        "transition_mass": pd.DataFrame(mass, index=s_names, columns=t_names),
    }


# =============================================================================
# Temporary cost workspace
# =============================================================================


class CostWorkspace:
    def __init__(self, outdir: Path, keep: bool, parent: Optional[Path]):
        self._tmp = None
        if keep:
            self.path = outdir / "temporary_costs"
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            self._tmp = tempfile.TemporaryDirectory(
                prefix="domain_concentration_",
                dir=str(parent) if parent is not None else None,
            )
            self.path = Path(self._tmp.name)

    def close(self):
        if self._tmp is not None:
            self._tmp.cleanup()


# =============================================================================
# Main
# =============================================================================


def main():
    args = build_parser().parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "run_config.json", vars(args))

    source, target = read_inputs(args.source_h5ad, args.target_h5ad)
    if args.preprocess:
        joint_pca_preprocess(
            source, target,
            output_key=args.feature_key,
            n_pcs=args.preprocess_n_pcs,
            n_hvg=args.preprocess_n_hvg,
            target_sum=args.preprocess_target_sum,
        )

    zs, zt, ss, st = validate_arrays(
        source, target,
        feature_key=args.feature_key,
        spatial_key=args.spatial_key,
    )
    log(f"[input] source={source.n_obs}, target={target.n_obs}, feature_dim={zs.shape[1]}")

    # Domain construction is unnecessary for a baseline-only grid search.
    # In a full refinement run, domains are still built before annotation is touched.
    if args.baseline_only:
        src_parts, tgt_parts = {}, {}
    else:
        src_parts, tgt_parts = build_partitions(source, target, args)
        save_partitions(outdir, source, target, src_parts, tgt_parts)

    pair, src_internal, tgt_internal = make_pair_adata(
        source, target, zs, zt, ss, st,
        feature_key=args.feature_key,
        spatial_key=args.spatial_key,
    )

    metrics = {
        "status": "started",
        "source_h5ad": args.source_h5ad,
        "target_h5ad": args.target_h5ad,
        "feature_key": args.feature_key,
        "spatial_key": args.spatial_key,
        "n_source": source.n_obs,
        "n_target": target.n_obs,
        "preprocess": args.preprocess,
        "strict_mouse_off": True,
        "growth_marginals": "skipped",
        "source_resolutions": args.source_resolutions,
        "target_resolutions": args.target_resolutions,
        "source_actual_K": {str(r): len(np.unique(x)) for r, x in src_parts.items()},
        "target_actual_K": {str(r): len(np.unique(x)) for r, x in tgt_parts.items()},
        "domain_n_neighbors": args.domain_n_neighbors,
        "leiden_seed": args.leiden_seed,
        "leiden_n_iterations": args.leiden_n_iterations,
        "lambda_concentration": args.lambda_concentration,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "tau_a": args.tau_a,
        "tau_b": args.tau_b,
        "rank": args.rank,
        "scale_cost": args.scale_cost,
        "min_iterations": args.min_iterations,
        "max_iterations": args.max_iterations,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "inner_min_iterations": args.inner_min_iterations,
        "inner_max_iterations": args.inner_max_iterations,
        "inner_threshold": args.inner_threshold,
        "inner_iterations": args.inner_iterations,
    }

    workspace = CostWorkspace(outdir, args.keep_temporary_costs, args.temp_dir)

    try:
        c0, c0_stats = build_c0(workspace.path / "baseline_xy_cost.npy", zs, zt, args.cost_chunk_rows)
        metrics["baseline_cost"] = c0_stats

        if args.baseline_transport is None:
            log("[baseline] solving strict mouse-off FGW (a=False,b=False)")
            p0, baseline_info = solve_fgw(pair, c0, src_internal, tgt_internal, args)
        else:
            log(f"[baseline] loading P0: {args.baseline_transport}")
            p0 = load_existing_p0(
                args.baseline_transport, source, target,
                args.baseline_source_names, args.baseline_target_names,
            )
            baseline_info = {"skipped": True, "loaded_from": args.baseline_transport}

        metrics["baseline_solver"] = baseline_info
        metrics["baseline_transport_sum"] = float(p0.sum())

        if args.save_baseline_transport:
            np.save(outdir / "baseline_transport.npy", p0)
            np.savetxt(outdir / "baseline_source_obs_names.txt", source.obs_names.astype(str), fmt="%s")
            np.savetxt(outdir / "baseline_target_obs_names.txt", target.obs_names.astype(str), fmt="%s")

        # Storage-light grid mode: solve/evaluate P0 only, save no dense matrix
        # unless --save-baseline-transport was explicitly requested.
        if args.baseline_only:
            if args.curated_transitions is not None:
                if args.annotation_key not in source.obs or args.annotation_key not in target.obs:
                    raise KeyError(f"annotation_key={args.annotation_key!r} missing from source or target")
                allowed = load_allowed(
                    args.curated_transitions,
                    args.curated_source_column,
                    args.curated_descendants_column,
                    args.include_self,
                )
                ya = source.obs[args.annotation_key].astype(str).to_numpy()
                yb = target.obs[args.annotation_key].astype(str).to_numpy()
                e0 = evaluate_curated(p0, ya, yb, allowed)
                e0["per_source"].to_csv(outdir / "baseline_per_source_accuracy.csv", index=False)
                e0["transition_mass"].to_csv(outdir / "baseline_transition_mass.csv")
                metrics["evaluation"] = {
                    "baseline_weighted_accuracy": e0["weighted_accuracy"],
                    "baseline_macro_accuracy": e0["macro_accuracy"],
                }
                log("\n" + "=" * 88)
                log("BASELINE-ONLY CURATED RESULT")
                log("=" * 88)
                log(f"baseline weighted = {e0['weighted_accuracy']:.6f}")
                log(f"baseline macro    = {e0['macro_accuracy']:.6f}")
            metrics["status"] = "success"
            metrics["baseline_only"] = True
            write_json(outdir / "metrics.json", metrics)
            log(f"[done baseline-only] {outdir / 'metrics.json'}")
            return

        penalties = []
        for sr, slabels in src_parts.items():
            for tr, tlabels in tgt_parts.items():
                log(f"[penalty] source_r={sr:g}, target_r={tr:g}")
                obj = domain_penalty(p0, slabels, tlabels)
                penalties.append(obj)
                save_penalty(outdir, sr, tr, obj)

        c1, c1_stats = build_refined_cost(
            workspace.path / "refined_xy_cost.npy",
            c0,
            penalties,
            lam=args.lambda_concentration,
            chunk_rows=args.cost_chunk_rows,
        )
        metrics["refined_cost"] = c1_stats

        log("[refined] solving one strict mouse-off FGW refinement")
        p1, refined_info = solve_fgw(pair, c1, src_internal, tgt_internal, args)
        metrics["refined_solver"] = refined_info
        metrics["refined_transport_sum"] = float(p1.sum())

        if args.save_refined_transport:
            np.save(outdir / "refined_transport.npy", p1)

        if args.curated_transitions is not None:
            if args.annotation_key not in source.obs or args.annotation_key not in target.obs:
                raise KeyError(f"annotation_key={args.annotation_key!r} missing from source or target")
            allowed = load_allowed(
                args.curated_transitions,
                args.curated_source_column,
                args.curated_descendants_column,
                args.include_self,
            )
            ya = source.obs[args.annotation_key].astype(str).to_numpy()
            yb = target.obs[args.annotation_key].astype(str).to_numpy()
            e0 = evaluate_curated(p0, ya, yb, allowed)
            e1 = evaluate_curated(p1, ya, yb, allowed)
            e0["per_source"].to_csv(outdir / "baseline_per_source_accuracy.csv", index=False)
            e1["per_source"].to_csv(outdir / "refined_per_source_accuracy.csv", index=False)
            e0["transition_mass"].to_csv(outdir / "baseline_transition_mass.csv")
            e1["transition_mass"].to_csv(outdir / "refined_transition_mass.csv")
            metrics["evaluation"] = {
                "baseline_weighted_accuracy": e0["weighted_accuracy"],
                "baseline_macro_accuracy": e0["macro_accuracy"],
                "refined_weighted_accuracy": e1["weighted_accuracy"],
                "refined_macro_accuracy": e1["macro_accuracy"],
                "delta_weighted": e1["weighted_accuracy"] - e0["weighted_accuracy"],
                "delta_macro": e1["macro_accuracy"] - e0["macro_accuracy"],
            }
            ev = metrics["evaluation"]
            log("\n" + "=" * 88)
            log("CURATED RESULT")
            log("=" * 88)
            log(f"baseline weighted = {ev['baseline_weighted_accuracy']:.6f}")
            log(f"baseline macro    = {ev['baseline_macro_accuracy']:.6f}")
            log(f"refined weighted  = {ev['refined_weighted_accuracy']:.6f}")
            log(f"refined macro     = {ev['refined_macro_accuracy']:.6f}")
            log(f"delta weighted    = {ev['delta_weighted']:+.6f}")
            log(f"delta macro       = {ev['delta_macro']:+.6f}")
        else:
            log("[evaluation] no curated table supplied; skipping annotation-based evaluation")

        metrics["status"] = "success"
        write_json(outdir / "metrics.json", metrics)
        log(f"[done] {outdir / 'metrics.json'}")

    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error_type"] = type(exc).__name__
        metrics["error"] = str(exc)
        write_json(outdir / "metrics.json", metrics)
        raise
    finally:
        workspace.close()
        gc.collect()


if __name__ == "__main__":
    main()
