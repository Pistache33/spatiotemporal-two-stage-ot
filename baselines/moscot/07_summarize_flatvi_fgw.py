#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_metrics(root: Path) -> pd.DataFrame:
    rows=[]
    for path in root.glob("**/metrics.json"):
        obj=json.loads(path.read_text(encoding="utf-8")); obj["metrics_path"]=str(path); rows.append(obj)
    return pd.DataFrame(rows)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("/mnt/disk2/lirui/mosta_flatvi_fgw/results"))
    p.add_argument("--output-dir", type=Path, default=Path("/mnt/disk2/lirui/mosta_flatvi_fgw/comparison"))
    p.add_argument("--pca30-results-root", type=Path, default=Path("/mnt/disk2/lirui/mosta_moscot_fast_fullrank/results"))
    args=p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    current=read_metrics(args.results_root)
    current=current[current.get("status", "success")=="success"].copy()
    if current.empty: raise ValueError("No successful FlatVI/NBVAE FGW metrics found")
    current.to_csv(args.output_dir/"all_flatvi_fgw_results.csv", index=False)
    key=["alpha","epsilon","tau_a","tau_b","rank","scale_cost"]
    pivot=current.pivot_table(index=key, columns="representation", values="accuracy", aggfunc="max").reset_index()
    pivot.to_csv(args.output_dir/"matched_accuracy.csv", index=False)
    macro=current.pivot_table(index=key, columns="representation", values="macro_curated_accuracy", aggfunc="max").reset_index()
    macro.to_csv(args.output_dir/"matched_macro_accuracy.csv", index=False)
    best=current.loc[current.groupby("representation")["accuracy"].idxmax()].sort_values("accuracy", ascending=False)
    best.to_csv(args.output_dir/"best_by_representation.csv", index=False)
    ref=current[np.isclose(current.alpha.astype(float),.2)&np.isclose(current.tau_a.astype(float),1)&np.isclose(current.tau_b.astype(float),.01)]
    ref.sort_values("accuracy",ascending=False).to_csv(args.output_dir/"reference_alpha0.2_taua1_taub0.01.csv", index=False)
    lines=["# FlatVI / NB-VAE / PCA-10 FGW", "", "## Reference configuration", "", ref[["representation","accuracy","macro_curated_accuracy","col_sum_max","row_entropy_norm_mean","solver_converged"]].sort_values("accuracy",ascending=False).to_markdown(index=False), "", "## Best within evaluated grid", "", best[["representation","alpha","tau_a","tau_b","accuracy","macro_curated_accuracy","col_sum_max"]].to_markdown(index=False)]
    (args.output_dir/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))


if __name__=="__main__": main()
