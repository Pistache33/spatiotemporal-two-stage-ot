#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official SOCS on MOSTA Pair A / Pair B with unsupervised Leiden structures.

Method construction uses NO curated annotation labels:
  1) load X_nbvae_30 + spatial only from source/target h5ad (backed mode),
  2) independently build Leiden structures on X_nbvae_30 kNN graphs,
  3) run the official SOCS CPU solver from the user's SOCS-main checkout,
  4) evaluate weighted / macro curated-transition accuracy post hoc.

Default grid matches the user's original 12-point moscot FGW grid:
  alpha_moscot in {0.2, 0.4, 0.6}
  (tau_a,tau_b) in {(1,.01),(1,.05),(.05,1),(.01,1)}
  epsilon = 1e-4

SOCS uses the opposite alpha convention:
  SOCS alpha = expression weight, while moscot alpha = quadratic/GW weight.
Therefore default --alpha-mode match_moscot sets alpha_socs = 1-alpha_moscot.
Use --alpha-mode same_numeric only if you explicitly want identical numeric alpha.

The official SOCS CPU solver internally uses
  tau = rho / (rho + epsilon)
so the matched conversion is
  rho = epsilon * tau / (1-tau), with tau=1 -> rho=inf.

Outputs:
  <out>/<Pair>/grid_results.csv
  <out>/<Pair>/best_by_weighted.json
  <out>/<Pair>/best_transport.npy
  <out>/<Pair>/source_domains.csv
  <out>/<Pair>/target_domains.csv
  per-task metrics.json and per_source_accuracy.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


PAIR_CONFIG = {
    "PairA": {
        "name": "E9.5_E1S1 -> E10.5_E2S1",
        "source": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_source_E9.5_E1S1.h5ad"),
        "target": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_target_E10.5_E2S1.h5ad"),
        "source_resolution": 1.0,
        "target_resolution": 1.4,
    },
    "PairB": {
        "name": "E9.5_E2S1 -> E10.5_E1S1",
        "source": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairB_source_E9.5_E2S1.h5ad"),
        "target": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairB_target_E10.5_E1S1.h5ad"),
        "source_resolution": 1.4,
        "target_resolution": 1.375,
    },
}

DEFAULT_CURATED = Path(
    "/mnt/disk2/lirui/baseline_scripts/mosta_flatvi_fgw/MOSTA_curated_transitions.tsv"
)
DEFAULT_SOCS_REPO = Path("/mnt/disk2/lirui/baseline_scripts/SOCS-main")
DEFAULT_OUT = Path("/mnt/disk2/lirui/mosta_socs_official_unsupervised")


def log(msg: str = "") -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(dict(obj)), indent=2, ensure_ascii=False), encoding="utf-8")


def split_descendants(value) -> List[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "/":
        return []
    return [x.strip() for x in re.split(r"[,;|\n]+", text) if x.strip() and x.strip() != "/"]


def load_allowed(path: Path, source_col: str, descendants_col: str, include_self: bool) -> Dict[str, set]:
    table = pd.read_csv(path, sep=None, engine="python")
    if source_col not in table.columns or descendants_col not in table.columns:
        raise KeyError(f"Curated columns={list(table.columns)}")
    allowed: Dict[str, set] = {}
    for _, row in table.iterrows():
        src = str(row[source_col]).strip()
        if not src or src.lower() == "nan":
            continue
        targets = set(split_descendants(row[descendants_col]))
        if include_self:
            targets.add(src)
        allowed[src] = targets
    return allowed


def evaluate_curated(
    p: np.ndarray,
    source_annotation: np.ndarray,
    target_annotation: np.ndarray,
    allowed: Mapping[str, set],
):
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


def load_lean(path: Path, feature_key: str, spatial_key: str, annotation_key: str):
    log(f"[load] {path}")
    x = ad.read_h5ad(path, backed="r")
    try:
        if feature_key not in x.obsm:
            raise KeyError(f"{path.name}: missing obsm[{feature_key!r}]")
        if spatial_key not in x.obsm:
            raise KeyError(f"{path.name}: missing obsm[{spatial_key!r}]")
        if annotation_key not in x.obs:
            raise KeyError(f"{path.name}: missing obs[{annotation_key!r}]")
        names = x.obs_names.astype(str).to_numpy()
        z = np.asarray(x.obsm[feature_key], dtype=np.float64).copy()
        spatial = np.asarray(x.obsm[spatial_key], dtype=np.float64).copy()
        annotation = x.obs[annotation_key].astype(str).to_numpy()
    finally:
        x.file.close()
    return names, z, spatial, annotation


def leiden_labels(z: np.ndarray, resolution: float, n_neighbors: int, seed: int, n_iterations: int) -> np.ndarray:
    a = ad.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
    a.obsm["X_rep"] = np.asarray(z, dtype=np.float32)
    sc.pp.neighbors(a, n_neighbors=int(n_neighbors), use_rep="X_rep", key_added="_graph")
    try:
        sc.tl.leiden(
            a,
            resolution=float(resolution),
            neighbors_key="_graph",
            key_added="struct",
            random_state=int(seed),
            flavor="igraph",
            n_iterations=int(n_iterations),
            directed=False,
        )
    except Exception:
        sc.tl.leiden(
            a,
            resolution=float(resolution),
            neighbors_key="_graph",
            key_added="struct",
            random_state=int(seed),
        )
    return a.obs["struct"].astype(str).to_numpy()


def tau_to_rho(tau: float, eps: float) -> float:
    tau = float(tau)
    if tau >= 1.0:
        return float("inf")
    if tau <= 0.0:
        return 0.0
    return float(eps) * tau / (1.0 - tau)


def make_grid(alpha_values: Iterable[float], eps: float, alpha_mode: str):
    tau_pairs = [
        (1.0, 0.01),
        (1.0, 0.05),
        (0.05, 1.0),
        (0.01, 1.0),
    ]
    tasks = []
    task_id = 0
    for alpha_moscot in alpha_values:
        alpha_moscot = float(alpha_moscot)
        if alpha_mode == "match_moscot":
            alpha_socs = 1.0 - alpha_moscot
        elif alpha_mode == "same_numeric":
            alpha_socs = alpha_moscot
        else:
            raise ValueError(alpha_mode)
        for tau_a, tau_b in tau_pairs:
            tasks.append({
                "task_id": task_id,
                "alpha_moscot": alpha_moscot,
                "alpha_socs": alpha_socs,
                "tau_a": tau_a,
                "tau_b": tau_b,
                "rho1": tau_to_rho(tau_a, eps),
                "rho2": tau_to_rho(tau_b, eps),
                "epsilon": float(eps),
            })
            task_id += 1
    return tasks


def task_tag(task: Mapping) -> str:
    def f(x):
        if math.isinf(float(x)):
            return "inf"
        return f"{float(x):g}".replace(".", "p")
    return (
        f"task_{int(task['task_id']):03d}"
        f"__am_{f(task['alpha_moscot'])}"
        f"__as_{f(task['alpha_socs'])}"
        f"__taua_{f(task['tau_a'])}"
        f"__taub_{f(task['tau_b'])}"
    )


def build_pair_adata(
    src_names: np.ndarray,
    tgt_names: np.ndarray,
    zs: np.ndarray,
    zt: np.ndarray,
    ss: np.ndarray,
    st: np.ndarray,
    src_ann: np.ndarray,
    tgt_ann: np.ndarray,
    src_struct: np.ndarray,
    tgt_struct: np.ndarray,
    feature_key: str,
    spatial_key: str,
    annotation_key: str,
):
    ns, nt = len(src_names), len(tgt_names)
    pair = ad.AnnData(X=np.zeros((ns + nt, 1), dtype=np.float32))
    pair.obs_names = np.concatenate([
        np.asarray([f"source::{x}" for x in src_names]),
        np.asarray([f"target::{x}" for x in tgt_names]),
    ])
    pair.obs["time"] = np.concatenate([
        np.asarray(["t0"] * ns, dtype=object),
        np.asarray(["t1"] * nt, dtype=object),
    ])
    pair.obs["struct"] = np.concatenate([src_struct.astype(str), tgt_struct.astype(str)])
    pair.obs[annotation_key] = np.concatenate([src_ann.astype(str), tgt_ann.astype(str)])
    pair.obsm[feature_key] = np.vstack([zs, zt]).astype(np.float64, copy=False)
    pair.obsm[spatial_key] = np.vstack([ss, st]).astype(np.float64, copy=False)
    return pair


def load_existing_successes(pair_out: Path) -> List[dict]:
    rows = []
    for p in sorted((pair_out / "tasks").glob("task_*/metrics.json")) if (pair_out / "tasks").exists() else []:
        try:
            x = json.loads(p.read_text())
        except Exception:
            continue
        if x.get("status") == "success":
            rows.append(x)
    return rows


def run_pair(pair_name: str, args, SOCSModel) -> None:
    cfg = PAIR_CONFIG[pair_name]
    pair_out = args.output_root / pair_name
    task_root = pair_out / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 100)
    log(f"{pair_name}: {cfg['name']}")
    log(f"source resolution={cfg['source_resolution']}, target resolution={cfg['target_resolution']}")
    log("=" * 100)

    src_names, zs, ss, src_ann = load_lean(cfg["source"], args.feature_key, args.spatial_key, args.annotation_key)
    tgt_names, zt, st, tgt_ann = load_lean(cfg["target"], args.feature_key, args.spatial_key, args.annotation_key)
    ns, nt = len(src_names), len(tgt_names)
    log(f"[input] source={ns}, target={nt}, feature_dim={zs.shape[1]}")

    src_struct = leiden_labels(zs, cfg["source_resolution"], args.n_neighbors, args.leiden_seed, args.leiden_iterations)
    tgt_struct = leiden_labels(zt, cfg["target_resolution"], args.n_neighbors, args.leiden_seed, args.leiden_iterations)
    ks, kt = len(np.unique(src_struct)), len(np.unique(tgt_struct))
    log(f"[domains] source K={ks}, target K={kt}")

    pd.DataFrame({"obs_name": src_names, "struct": src_struct}).to_csv(pair_out / "source_domains.csv", index=False)
    pd.DataFrame({"obs_name": tgt_names, "struct": tgt_struct}).to_csv(pair_out / "target_domains.csv", index=False)

    allowed = load_allowed(args.curated, args.curated_source_col, args.curated_desc_col, args.include_self)

    tasks = make_grid(args.alpha_values, args.epsilon, args.alpha_mode)
    existing = {int(x["task_id"]): x for x in load_existing_successes(pair_out)} if args.resume else {}
    pending = [t for t in tasks if int(t["task_id"]) not in existing]

    log(f"[grid] total={len(tasks)}, existing_success={len(existing)}, pending={len(pending)}")
    if not pending:
        all_rows = list(existing.values())
        grid_df = pd.DataFrame(all_rows).sort_values("task_id")
        grid_df.to_csv(pair_out / "grid_results.csv", index=False)
        best = grid_df.sort_values(["weighted_accuracy", "macro_accuracy"], ascending=False).iloc[0].to_dict()
        write_json(pair_out / "best_by_weighted.json", best)
        log(grid_df[["task_id","alpha_moscot","alpha_socs","tau_a","tau_b","weighted_accuracy","macro_accuracy"]].to_string(index=False))
        return

    pair = build_pair_adata(
        src_names, tgt_names, zs, zt, ss, st, src_ann, tgt_ann,
        src_struct, tgt_struct, args.feature_key, args.spatial_key, args.annotation_key,
    )

    model = SOCSModel(
        pair,
        time_key="time",
        spatial_key=args.spatial_key,
        expr_key=args.feature_key,
        struct_key="struct",
        gr_key=None,
    )

    # Exact official SOCS distance construction, computed once per pair.
    log("[SOCS] computing D0, D1, D01 once")
    t_geom = time.time()
    D0, D1, D01 = model.compute_distance_matrices("t0", "t1", verbose=True)
    d0max, d1max, d01max = float(np.max(D0)), float(np.max(D1)), float(np.max(D01))
    if d0max <= 0 or d1max <= 0 or d01max <= 0:
        raise ValueError(f"Invalid distance maxima: D0={d0max}, D1={d1max}, D01={d01max}")

    # Official defaults: feedback equals max raw spatial distance.
    fb0 = d0max if args.fb0 is None else float(args.fb0)
    fb1 = d1max if args.fb1 is None else float(args.fb1)

    # compute_struct_matrices reads model.ot_config['struct_excl'].
    model.ot_config = {"struct_excl": list(args.struct_excl)}
    S0, S1 = model.compute_struct_matrices("t0", "t1", fb0, fb1, verbose=True)
    D0 += S0
    D1 += S1
    del S0, S1
    gc.collect()

    # Exact official default geometry normalizers use maxima BEFORE adding structure.
    f0 = (d0max / d01max) ** 2 if args.f0 is None else float(args.f0)
    f1 = (d1max / d01max) ** 2 if args.f1 is None else float(args.f1)

    geom_meta = {
        "source_n": ns,
        "target_n": nt,
        "source_K": ks,
        "target_K": kt,
        "source_resolution": cfg["source_resolution"],
        "target_resolution": cfg["target_resolution"],
        "D0_max_raw": d0max,
        "D1_max_raw": d1max,
        "D01_max": d01max,
        "fb0": fb0,
        "fb1": fb1,
        "f0": f0,
        "f1": f1,
        "geometry_seconds": time.time() - t_geom,
        "feature_key": args.feature_key,
        "spatial_key": args.spatial_key,
        "n_neighbors": args.n_neighbors,
        "leiden_seed": args.leiden_seed,
        "leiden_iterations": args.leiden_iterations,
        "struct_excl": list(args.struct_excl),
    }
    write_json(pair_out / "geometry_metadata.json", geom_meta)
    log(f"[SOCS] geometry ready in {geom_meta['geometry_seconds']:.1f}s; fb0={fb0:.6g}, fb1={fb1:.6g}, f0={f0:.6g}, f1={f1:.6g}")

    # Existing rows matter when resuming and deciding whether a new coupling is best.
    rows = list(existing.values())
    best_weighted = max([float(x["weighted_accuracy"]) for x in rows], default=-np.inf)

    for task in pending:
        tag = task_tag(task)
        out = task_root / tag
        out.mkdir(parents=True, exist_ok=True)
        log("\n" + "-" * 100)
        log(
            f"[{pair_name}] {tag} | moscot-alpha={task['alpha_moscot']:.3g} -> "
            f"SOCS-alpha={task['alpha_socs']:.3g} | tau=({task['tau_a']:.3g},{task['tau_b']:.3g}) | "
            f"rho=({task['rho1']},{task['rho2']})"
        )
        log("-" * 100)

        metrics = {
            "status": "started",
            "pair": pair_name,
            **task,
            "alpha_mode": args.alpha_mode,
            "method": args.method,
            "max_iters": args.max_iters,
            **geom_meta,
        }
        write_json(out / "metrics.json", metrics)

        t0 = time.time()
        try:
            # Same official compute_transport_map and official CPU UGW solver.
            model.ot_config = {
                "eps": float(task["epsilon"]),
                "alpha": float(task["alpha_socs"]),
                "rho": float(task["rho1"]),
                "rho2": float(task["rho2"]),
                "nIters": int(args.max_iters),
                "struct_excl": list(args.struct_excl),
            }
            P = model.compute_transport_map(
                D0, D1, D01, f0, f1,
                "t0", "t1",
                method=args.method,
                verbose=True,
            )
            P = np.asarray(P, dtype=np.float64)
            if P.shape != (ns, nt) or not np.isfinite(P).all() or np.any(P < 0):
                raise ValueError(f"Invalid transport shape/values: {P.shape}")

            ev = evaluate_curated(P, src_ann, tgt_ann, allowed)
            elapsed = time.time() - t0
            metrics.update({
                "status": "success",
                "elapsed_seconds": elapsed,
                "transport_sum": float(P.sum()),
                "weighted_accuracy": float(ev["weighted_accuracy"]),
                "macro_accuracy": float(ev["macro_accuracy"]),
            })
            ev["per_source"].to_csv(out / "per_source_accuracy.csv", index=False)
            ev["transition_mass"].to_csv(out / "transition_mass.csv")
            write_json(out / "metrics.json", metrics)
            rows.append(metrics)

            log(f"weighted = {metrics['weighted_accuracy']:.6f}")
            log(f"macro    = {metrics['macro_accuracy']:.6f}")
            log(f"seconds  = {elapsed:.1f}")

            if args.save_all_transports:
                np.save(out / "transport.npy", P.astype(np.float32))
                np.savetxt(out / "source_obs_names.txt", src_names, fmt="%s")
                np.savetxt(out / "target_obs_names.txt", tgt_names, fmt="%s")

            if metrics["weighted_accuracy"] > best_weighted:
                best_weighted = metrics["weighted_accuracy"]
                np.save(pair_out / "best_transport.npy", P.astype(np.float32))
                np.savetxt(pair_out / "source_obs_names.txt", src_names, fmt="%s")
                np.savetxt(pair_out / "target_obs_names.txt", tgt_names, fmt="%s")
                write_json(pair_out / "best_by_weighted.json", metrics)
                log(f"[best] new best weighted={best_weighted:.6f}")

            del P
            gc.collect()

        except Exception as exc:
            metrics.update({
                "status": "failed",
                "elapsed_seconds": time.time() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            write_json(out / "metrics.json", metrics)
            log(f"[FAILED] {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

        success_rows = [x for x in rows if x.get("status") == "success"]
        if success_rows:
            pd.DataFrame(success_rows).sort_values("task_id").to_csv(pair_out / "grid_results.csv", index=False)

    success_rows = [x for x in rows if x.get("status") == "success"]
    if not success_rows:
        raise RuntimeError(f"No successful SOCS tasks for {pair_name}")

    grid_df = pd.DataFrame(success_rows).sort_values("task_id").reset_index(drop=True)
    grid_df.to_csv(pair_out / "grid_results.csv", index=False)
    best_row = grid_df.sort_values(["weighted_accuracy", "macro_accuracy"], ascending=False).iloc[0].to_dict()
    write_json(pair_out / "best_by_weighted.json", best_row)

    log("\n" + "=" * 100)
    log(f"{pair_name} GRID RESULT")
    log("=" * 100)
    show = grid_df[[
        "task_id", "alpha_moscot", "alpha_socs", "tau_a", "tau_b",
        "rho1", "rho2", "weighted_accuracy", "macro_accuracy", "elapsed_seconds"
    ]]
    log(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    log("\nBEST BY WEIGHTED")
    log(
        f"task={int(best_row['task_id'])} | weighted={best_row['weighted_accuracy']:.6f} | "
        f"macro={best_row['macro_accuracy']:.6f} | alpha_moscot={best_row['alpha_moscot']} | "
        f"alpha_socs={best_row['alpha_socs']} | tau=({best_row['tau_a']},{best_row['tau_b']})"
    )

    del D0, D1, D01, pair, model
    gc.collect()


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pairs", nargs="+", choices=list(PAIR_CONFIG), default=["PairA", "PairB"])
    p.add_argument("--socs-repo", type=Path, default=DEFAULT_SOCS_REPO)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    p.add_argument("--curated-source-col", default="Cell_type")
    p.add_argument("--curated-desc-col", default="Known_descendants")
    p.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--feature-key", default="X_nbvae_30")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--annotation-key", default="annotation")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--leiden-seed", type=int, default=0)
    p.add_argument("--leiden-iterations", type=int, default=2)

    p.add_argument("--alpha-values", type=float, nargs="+", default=[0.2, 0.4, 0.6], help="Original moscot alpha grid values.")
    p.add_argument(
        "--alpha-mode", choices=["match_moscot", "same_numeric"], default="match_moscot",
        help="match_moscot: SOCS alpha=1-moscot alpha because the conventions are opposite; same_numeric: reuse the number literally.",
    )
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--method", choices=["fugw_cpu"], default="fugw_cpu", help="CPU is required to preserve separate rho1/rho2 and match the asymmetric tau grid.")

    p.add_argument("--fb0", type=float, default=None, help="None = official SOCS default max(D0).")
    p.add_argument("--fb1", type=float, default=None, help="None = official SOCS default max(D1).")
    p.add_argument("--f0", type=float, default=None, help="None = official SOCS default (max(D0)/max(D01))^2.")
    p.add_argument("--f1", type=float, default=None, help="None = official SOCS default (max(D1)/max(D01))^2.")
    p.add_argument("--struct-excl", nargs="*", default=[], help="For unsupervised Leiden all clusters are structural, so default is empty.")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-all-transports", action=argparse.BooleanOptionalAction, default=False)
    return p


def main():
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    repo = args.socs_repo.resolve()
    if not (repo / "socs").is_dir():
        raise FileNotFoundError(f"Cannot find SOCS package under {repo / 'socs'}")
    sys.path.insert(0, str(repo))

    # Import only after prepending the user's official checkout.
    from socs.ot.socs_model import SOCSModel

    run_meta = {
        "socs_repo": repo,
        "pairs": args.pairs,
        "feature_key": args.feature_key,
        "alpha_values_moscot": args.alpha_values,
        "alpha_mode": args.alpha_mode,
        "epsilon": args.epsilon,
        "max_iters": args.max_iters,
        "method": args.method,
        "grid_tau_pairs": [(1.0,0.01),(1.0,0.05),(0.05,1.0),(0.01,1.0)],
        "tau_to_rho_formula": "rho = epsilon * tau / (1-tau); tau=1 -> inf",
    }
    write_json(args.output_root / "run_config.json", run_meta)

    log(f"Python: {sys.executable}")
    log(f"SOCS repo: {repo}")
    log(f"Output: {args.output_root}")
    log(f"Grid: 3 alpha x 4 tau = {3*4} tasks/pair")
    log(f"alpha mode: {args.alpha_mode}")
    log(f"epsilon: {args.epsilon:g}, max_iters: {args.max_iters}")

    for pair_name in args.pairs:
        run_pair(pair_name, args, SOCSModel)

    # Combined summary of best weighted setting per pair.
    bests = []
    for pair_name in args.pairs:
        p = args.output_root / pair_name / "best_by_weighted.json"
        if p.exists():
            x = json.loads(p.read_text())
            bests.append({
                "pair": pair_name,
                "task_id": x.get("task_id"),
                "alpha_moscot": x.get("alpha_moscot"),
                "alpha_socs": x.get("alpha_socs"),
                "tau_a": x.get("tau_a"),
                "tau_b": x.get("tau_b"),
                "rho1": x.get("rho1"),
                "rho2": x.get("rho2"),
                "weighted_accuracy": x.get("weighted_accuracy"),
                "macro_accuracy": x.get("macro_accuracy"),
            })
    if bests:
        df = pd.DataFrame(bests)
        df.to_csv(args.output_root / "best_summary.csv", index=False)
        log("\n" + "=" * 100)
        log("BEST SOCS RESULTS")
        log("=" * 100)
        log(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
