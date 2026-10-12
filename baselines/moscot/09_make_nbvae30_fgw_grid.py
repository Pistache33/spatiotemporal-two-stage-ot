#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


GRID_CONFIGS = [
    (alpha, tau_a, tau_b)
    for alpha in (0.2, 0.4, 0.6)
    for tau_a, tau_b in (
        (1.0, 0.01),
        (1.0, 0.05),
        (0.05, 1.0),
        (0.01, 1.0),
    )
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create the 12-point NB-VAE-30 full-rank FGW grid."
    )
    p.add_argument(
        "--pair-h5ad",
        type=Path,
        default=Path(
            "/mnt/disk2/lirui/mosta_flatvi_fgw/data/"
            "E9.5__E10.5.nbvae30_latents.h5ad"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/disk2/lirui/mosta_flatvi_fgw/grid_nbvae30.tsv"),
    )
    p.add_argument("--representation", default="X_nbvae_30")
    args = p.parse_args()

    rows = []
    for task_id, (alpha, tau_a, tau_b) in enumerate(GRID_CONFIGS):
        rows.append(
            {
                "task_id": task_id,
                "pair": "E9.5__E10.5",
                "pair_h5ad": str(args.pair_h5ad),
                "source_time": 9.5,
                "target_time": 10.5,
                "representation": args.representation,
                "time_key": "time",
                "spatial_key": "spatial",
                "annotation_key": "annotation",
                "cost": "euclidean",
                "scale_cost": "max_cost",
                "alpha": alpha,
                "epsilon": 0.0001,
                "tau_a": tau_a,
                "tau_b": tau_b,
                "rank": -1,
                "gamma_requested": 100,
                "min_iterations": 50,
                "max_iterations": 1500,
                "threshold": 0.01,
                "batch_size": 1024,
                "inner_min_iterations": 20,
                "inner_max_iterations": 500,
                "inner_threshold": 0.01,
                "inner_iterations": 10,
                "jax_enable_x64": "false",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} NB-VAE-30 FGW tasks to {args.output}")
    for row in rows:
        print(
            f"task={row['task_id']:02d} "
            f"alpha={row['alpha']} tau_a={row['tau_a']} tau_b={row['tau_b']}"
        )


if __name__ == "__main__":
    main()
