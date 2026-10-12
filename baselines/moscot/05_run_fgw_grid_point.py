#!/usr/bin/env python3
"""Run one full-rank embedding FGW task and evaluate the dense coupling."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from fgw_eval_common import (
    aggregate_dense_transport,
    allowed_edges_from_table,
    coupling_diagnostics,
    evaluate_transition_mass,
    json_dump,
    load_grid_row,
    macro_accuracy,
    per_source_accuracy,
    read_name_file,
    row_normalize,
    safe_float,
    write_name_file,
)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def digest_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def supported_kwargs(callable_obj, requested: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(requested), {}
    accepted = {key: value for key, value in requested.items() if key in signature.parameters}
    dropped = {key: value for key, value in requested.items() if key not in signature.parameters}
    return accepted, dropped


def get_subproblem(problem, source: float, target: float):
    for key in ((source, target), (str(source), str(target))):
        try:
            return key, problem[key]
        except Exception:
            continue
    keys = list(problem.problems.keys())
    if len(keys) == 1:
        return keys[0], problem[keys[0]]
    raise KeyError(f"Could not locate subproblem {(source, target)}; available keys={keys}")


def get_solution(problem, key):
    if key in problem.solutions:
        return problem.solutions[key]
    if len(problem.solutions) == 1:
        return next(iter(problem.solutions.values()))
    raise KeyError(f"Could not locate solution for {key}; available={list(problem.solutions)}")


def extract_transport(solution) -> np.ndarray:
    candidate = getattr(solution, "transport_matrix", None)
    if callable(candidate):
        candidate = candidate()
    if candidate is None:
        candidate = getattr(solution, "matrix", None)
        if callable(candidate):
            candidate = candidate()
    if candidate is None:
        raise AttributeError("Solution exposes neither transport_matrix nor matrix")
    transport = np.asarray(candidate, dtype=np.float32)
    if transport.ndim != 2:
        raise ValueError(f"Expected 2-D transport matrix, got shape={transport.shape}")
    if not np.isfinite(transport).all() or np.any(transport < 0):
        raise ValueError("Transport contains non-finite or negative values")
    return transport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/disk2/lirui/mosta_flatvi_fgw/results"),
    )
    parser.add_argument("--skip-growth-marginals", action="store_true")
    parser.add_argument("--uniform-on-growth-score-failure", action="store_true")
    parser.add_argument("--skip-coupling-diagnostics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    row = load_grid_row(args.grid, args.task_id)
    jax_enable_x64 = parse_bool(row.get("jax_enable_x64", "false"))
    try:
        from jax import config as jax_config
        jax_config.update("jax_enable_x64", jax_enable_x64)
    except Exception:
        pass

    # Import moscot only after setting JAX precision.
    from moscot.problems.spatiotemporal import SpatioTemporalProblem

    pair = row["pair"]
    representation = row["representation"]
    alpha = float(row["alpha"])
    tau_a = float(row["tau_a"])
    tau_b = float(row["tau_b"])
    epsilon = float(row["epsilon"])
    rank = int(row["rank"])
    gamma_requested = float(row["gamma_requested"])
    source_time = float(row["source_time"])
    target_time = float(row["target_time"])

    slug = f"task_{args.task_id:03d}__alpha_{alpha:g}__taua_{tau_a:g}__taub_{tau_b:g}"
    out_dir = args.output_root / pair / representation / slug
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"[skip] {metrics_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = {
        "task_id": args.task_id,
        "pair": pair,
        "pair_h5ad": row["pair_h5ad"],
        "source_time": source_time,
        "target_time": target_time,
        "representation": representation,
        "feature_cost": "embedding_euclidean",
        "time_key": row["time_key"],
        "spatial_key": row["spatial_key"],
        "annotation_key": row["annotation_key"],
        "cost": row["cost"],
        "scale_cost": row["scale_cost"],
        "alpha": alpha,
        "epsilon": epsilon,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "rank": rank,
        "gamma_requested": gamma_requested,
        "gamma_applied": False,
        "min_iterations": int(row["min_iterations"]),
        "max_iterations": int(row["max_iterations"]),
        "threshold": float(row["threshold"]),
        "batch_size": int(row["batch_size"]),
        "inner_min_iterations": int(row["inner_min_iterations"]),
        "inner_max_iterations": int(row["inner_max_iterations"]),
        "inner_threshold": float(row["inner_threshold"]),
        "inner_iterations": int(row["inner_iterations"]),
        "jax_enable_x64": jax_enable_x64,
        "software_versions": {
            name: package_version(name) for name in ("moscot", "ott", "jax", "scanpy", "anndata", "numpy", "pandas")
        },
    }

    t0 = time.time()
    transport_saved = False
    try:
        adata = sc.read_h5ad(row["pair_h5ad"])
        time_key = row["time_key"]
        annotation_key = row["annotation_key"]
        if representation not in adata.obsm:
            raise KeyError(f"Representation {representation!r} missing from obsm={list(adata.obsm)}")
        embedding = np.asarray(adata.obsm[representation])
        if embedding.ndim != 2 or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding {representation}: shape={embedding.shape}")
        observed = sorted(map(float, adata.obs[time_key].unique()))
        if observed != [source_time, target_time]:
            raise ValueError(f"Observed times {observed} != expected {[source_time, target_time]}")

        problem = SpatioTemporalProblem(adata=adata)
        growth_status = "skipped"
        if not args.skip_growth_marginals:
            try:
                problem = problem.score_genes_for_marginals(
                    gene_set_proliferation="mouse", gene_set_apoptosis="mouse",
                )
                growth_status = "mouse_scored"
            except Exception as exc:
                if not args.uniform_on_growth_score_failure:
                    raise
                growth_status = f"failed_fell_back_to_uniform:{type(exc).__name__}:{exc}"

        problem = problem.prepare(
            time_key=time_key, spatial_key=row["spatial_key"], joint_attr=representation, cost=row["cost"],
        )
        subproblem_key, subproblem = get_subproblem(problem, source_time, target_time)
        source_names = np.asarray(subproblem.adata_src.obs_names.astype(str))
        target_names = np.asarray(subproblem.adata_tgt.obs_names.astype(str))

        linear_solver_kwargs = {
            "min_iterations": int(row["inner_min_iterations"]),
            "max_iterations": int(row["inner_max_iterations"]),
            "threshold": float(row["inner_threshold"]),
            "inner_iterations": int(row["inner_iterations"]),
        }
        requested_solve_kwargs = {
            "alpha": alpha, "epsilon": epsilon, "tau_a": tau_a, "tau_b": tau_b, "rank": rank,
            "scale_cost": row["scale_cost"], "min_iterations": int(row["min_iterations"]),
            "max_iterations": int(row["max_iterations"]), "threshold": float(row["threshold"]),
            "batch_size": int(row["batch_size"]), "linear_solver_kwargs": linear_solver_kwargs,
        }
        solve_kwargs, dropped = supported_kwargs(problem.solve, requested_solve_kwargs)
        essential = {"alpha", "epsilon", "tau_a", "tau_b", "rank", "scale_cost", "min_iterations", "max_iterations", "threshold"}
        missing_essential = essential.intersection(dropped)
        if missing_essential:
            raise TypeError(f"moscot solve() rejected essential parameters: {sorted(missing_essential)}; signature={inspect.signature(problem.solve)}")
        problem = problem.solve(**solve_kwargs)
        solution = get_solution(problem, subproblem_key)
        transport = extract_transport(solution)
        if transport.shape != (len(source_names), len(target_names)):
            raise ValueError(f"Transport shape {transport.shape} does not match source/target sizes")

        transport_path = out_dir / "transport.npy"
        np.save(transport_path, transport); transport_saved = True
        write_name_file(source_names, out_dir / "source_obs_names.txt")
        write_name_file(target_names, out_dir / "target_obs_names.txt")
        source_labels = subproblem.adata_src.obs[annotation_key].astype(str).to_numpy()
        target_labels = subproblem.adata_tgt.obs[annotation_key].astype(str).to_numpy()
        transition_mass = aggregate_dense_transport(transport, source_labels, target_labels)
        transition_prob = row_normalize(transition_mass)
        transition_mass.to_csv(out_dir / "cell_type_transition_mass.csv")
        transition_prob.to_csv(out_dir / "cell_type_transition.csv")
        allowed = allowed_edges_from_table(args.transitions, include_self=True)
        source_table = per_source_accuracy(transition_mass, allowed)
        source_table.to_csv(out_dir / "per_source_accuracy.csv", index=False)
        score = evaluate_transition_mass(transition_mass, allowed)
        diagnostics = {} if args.skip_coupling_diagnostics else coupling_diagnostics(transport)
        solver_rank = getattr(solution, "rank", -1); solver_rank = solver_rank() if callable(solver_rank) else solver_rank
        raw_low_rank = getattr(solution, "is_low_rank", False); raw_low_rank = raw_low_rank() if callable(raw_low_rank) else raw_low_rank
        metrics = {
            **base_metrics, "status": "success", "growth_marginals": growth_status,
            "solver_kwargs_applied": solve_kwargs, "solver_kwargs_dropped": dropped,
            "solver_converged": bool(getattr(solution, "converged", False)),
            "solver_cost": safe_float(getattr(solution, "cost", math.nan)),
            "solution_rank": int(solver_rank) if solver_rank is not None else -1,
            "solution_is_low_rank": bool(raw_low_rank), "elapsed_seconds": time.time() - t0,
            "n_source_cells": int(transport.shape[0]), "n_target_cells": int(transport.shape[1]),
            "n_source_labels": int(transition_mass.shape[0]), "n_target_labels": int(transition_mass.shape[1]),
            "macro_curated_accuracy": macro_accuracy(source_table),
            "transport_path": str(transport_path),
            "source_obs_names_path": str(out_dir / "source_obs_names.txt"),
            "target_obs_names_path": str(out_dir / "target_obs_names.txt"),
            "cell_type_transition_mass_path": str(out_dir / "cell_type_transition_mass.csv"),
            "cell_type_transition_path": str(out_dir / "cell_type_transition.csv"),
            "per_source_accuracy_path": str(out_dir / "per_source_accuracy.csv"),
            "transport_min": float(transport.min()), "transport_max": float(transport.max()),
            **score, **diagnostics,
        }
        json_dump(metrics, metrics_path)
        print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True))
        del problem, subproblem, solution, transport, adata
        gc.collect()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass

    except Exception as exc:
        failure = {
            **base_metrics,
            "status": "failed",
            "transport_saved_before_failure": transport_saved,
            "elapsed_seconds": time.time() - t0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        json_dump(failure, out_dir / "failure.json")
        print(json.dumps(failure, indent=2, ensure_ascii=False, allow_nan=True))
        raise


if __name__ == "__main__":
    main()
