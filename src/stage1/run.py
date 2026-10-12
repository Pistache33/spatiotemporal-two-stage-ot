"""Stage-1 orchestrator: run a pair from scratch and save outputs.

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 0). This is the actual
Stage-1 run path; it writes the same files as the original notebook.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
from ._constants import (LEIDEN_K, LEIDEN_SEED, LEIDEN_N_ITERATIONS,
                         N_BOUNDARY_POINTS, SPATIAL_CC_FACTOR)
from .spatial_alignment import build_global_deformation
from .domain_clustering import (get_xy, get_z, run_leiden, median_1nn,
                                leiden_spatial_components)
from .pareto_routing import build_pareto_supports
from .feasible_set import evaluate_support

# Original absolute paths from the source notebook (cell 0), preserved verbatim.
ROOT = Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/pareto")
PAIR_CONFIG = {
    "PairA": {
        "name": "E9.5_E1S1 -> E10.5_E2S1",
        "source": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_source_E9.5_E1S1.h5ad"),
        "target": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_target_E10.5_E2S1.h5ad"),
    },
    "PairC": {
        "name": "E10.5_E2S1 -> E11.5_E1S1",
        "source": Path("/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/E10.5_E2S1.nbvae30.h5ad"),
        "target": Path("/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/E11.5_E1S1.nbvae30.h5ad"),
    },
}
SOURCE_RES, TARGET_RES = 1.0, 0.6


def resolution_tag(x):
    return f"{float(x):g}".replace(".", "p")


def rotate_for_display(X, center=None):
    X = np.asarray(X, dtype=np.float64)
    center = X.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    span = np.ptp(X, axis=0)
    if span[0] >= span[1]:
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        angle = 90
    else:
        R = np.eye(2)
        angle = 0
    return (X - center) @ R.T, angle


def plot_tps_overlay(pair_name, cfg, deformation, outdir):
    figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    target_xy = deformation["target_aligned"]
    warped_xy = deformation["warped_source"]
    center = target_xy.mean(axis=0)
    target_d, angle = rotate_for_display(target_xy, center)
    warped_d, _ = rotate_for_display(warped_xy, center)
    fig, ax = plt.subplots(figsize=(7.2, 10.5))
    ax.scatter(target_d[:, 0], target_d[:, 1], s=7, c="#4C78A8", alpha=0.28, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(warped_d[:, 0], warped_d[:, 1], s=7, facecolors="none", edgecolors="#E67E22", linewidths=0.45, alpha=0.78, rasterized=True, zorder=2)
    source_stage, target_stage = cfg["name"].split(" -> ")
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#4C78A8", markeredgecolor="none", markersize=7, alpha=0.7, label=f"target: {target_stage}"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="#E67E22", markersize=7, label=f"warped source: {source_stage} → target frame"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=11)
    ax.set_title(f"{pair_name} | contour TPS alignment", fontsize=20, pad=12)
    ax.text(0.01, 0.99, f"orientation={np.asarray(deformation['orientation_matrix']).tolist()} | shift={deformation['cyclic_shift']} | display rotation={angle}°", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_linewidth(1.0)
    fig.tight_layout()
    path = figdir / f"{pair_name}_src{resolution_tag(SOURCE_RES)}_tgt{resolution_tag(TARGET_RES)}_02_tps_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return path


def save_stage1_outputs(pair_name, cfg, source, target, src_leiden, tgt_leiden, deformation, components, pareto, source_domain_code, outdir, summary, table):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "source_leiden.npy", src_leiden)
    np.save(outdir / "target_leiden.npy", tgt_leiden)
    np.savez_compressed(outdir / "deformation.npz",
        source_aligned=deformation["source_aligned"], target_aligned=deformation["target_aligned"],
        warped_source=deformation["warped_source"], cycle_error=deformation["cycle_error"],
        source_boundary=deformation["source_boundary"], target_boundary=deformation["target_boundary"],
        orientation_matrix=deformation["orientation_matrix"])
    comp_indptr = np.r_[0, np.cumsum([len(x) for x in components["pieces"]], dtype=np.int64)]
    comp_indices = np.concatenate(components["pieces"]).astype(np.int32)
    np.savez_compressed(outdir / "target_components.npz", indptr=comp_indptr, indices=comp_indices, piece_of_spot=components["piece_of_spot"], piece_leiden=components["piece_leiden"])
    sup = pareto["support_by_domain"]
    sup_indptr = np.r_[0, np.cumsum([len(x) for x in sup], dtype=np.int64)]
    sup_indices = np.concatenate(sup).astype(np.int32) if sup else np.empty(0, dtype=np.int32)
    np.savez_compressed(outdir / "hard_support.npz", source_domain_code=source_domain_code, support_indptr=sup_indptr, support_indices=sup_indices, source_domain_labels=pareto["source_labels"], target_domain_labels=pareto["target_labels"])
    pareto["routing_table"].to_csv(outdir / "pareto_domain_routing.csv", index=False)
    pd.DataFrame({"source_domain": pareto["source_labels"], "n_source_spots": [len(x) for x in pareto["source_members"]], "n_support_targets": [len(x) for x in sup], "support_fraction_target": [len(x) / target.n_obs for x in sup]}).to_csv(outdir / "support_by_source_domain.csv", index=False)
    table.to_csv(outdir / "support_posthoc_by_annotation.csv", index=False)
    pd.DataFrame([summary]).to_csv(outdir / "support_summary.csv", index=False)
    meta = {
        "pair": pair_name, "name": cfg["name"], "source_resolution": float(SOURCE_RES), "target_resolution": float(TARGET_RES),
        "leiden_k": LEIDEN_K, "leiden_seed": LEIDEN_SEED, "leiden_n_iterations": LEIDEN_N_ITERATIONS,
        "boundary_points": N_BOUNDARY_POINTS, "spatial_cc_factor": SPATIAL_CC_FACTOR,
        "construction": "spatial contour/TPS + independent NBVAE30 Leiden + Pareto domain routing + touched target-Leiden-local components",
        "cycle_error_saved_only_for_diagnostic": True, "cycle_error_used_in_support": False,
        "annotation_used_in_construction": False, "curated_used_in_construction": False,
        "cavity_in_construction": True, "cavity_dropped_only_in_posthoc_evaluation": True,
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)


def run_pair_from_scratch(pair_name, cfg, source_resolution=1.0, target_resolution=0.6):
    global SOURCE_RES, TARGET_RES
    SOURCE_RES, TARGET_RES = float(source_resolution), float(target_resolution)
    outdir = ROOT / pair_name / f"src{resolution_tag(SOURCE_RES)}_tgt{resolution_tag(TARGET_RES)}"
    print("=" * 100); print(f"{pair_name} | {cfg['name']} | source res={SOURCE_RES} | target res={TARGET_RES}"); print(f"output: {outdir}"); print("=" * 100)
    source, target = sc.read_h5ad(cfg["source"]), sc.read_h5ad(cfg["target"])
    src_xy, tgt_xy = get_xy(source), get_xy(target)
    print(f"source: {source.shape} | target: {target.shape}")
    print("\n[1] CONTOUR ALIGNMENT")
    deformation = build_global_deformation(src_xy, tgt_xy)
    print(f"orientation: {int(np.ravel(deformation['orientation_matrix'])[0] == np.ravel(deformation['orientation_matrix'])[0]) and np.nan if False else 'selected'} | chamfer: {deformation['orientation_score']}")
    print(f"orientation matrix:\n{deformation['orientation_matrix']}")
    print(f"cyclic shift: {deformation['cyclic_shift']} | cyclic MSE: {deformation['cyclic_mse']}")
    print(f"cycle error: median={np.median(deformation['cycle_error']):.6f} | q95={np.quantile(deformation['cycle_error'], .95):.6f} (diagnostic only)")
    plot_path = plot_tps_overlay(pair_name, cfg, deformation, outdir)
    print(f"saved: {plot_path}")
    print("\n[2] INDEPENDENT NBVAE30 LEIDEN")
    src_leiden, tgt_leiden = run_leiden(source, SOURCE_RES), run_leiden(target, TARGET_RES)
    print(f"source Leiden: {len(np.unique(src_leiden))} | target Leiden: {len(np.unique(tgt_leiden))}")
    print("\n[3] TARGET LEIDEN -> LOCAL SPATIAL COMPONENTS")
    tgt_med = median_1nn(deformation["target_aligned"]); tgt_radius = SPATIAL_CC_FACTOR * tgt_med
    components = leiden_spatial_components(tgt_leiden, deformation["target_aligned"], tgt_radius)
    print(f"target median 1NN: {tgt_med} | CC radius: {tgt_radius} | components: {len(components['pieces'])}")
    print("\n[4] PARETO DOMAIN ROUTING + TOUCHED COMPONENTS")
    zs, zt = get_z(source), get_z(target)
    pareto = build_pareto_supports(deformation["warped_source"], zs, zt, src_leiden, tgt_leiden, deformation["target_aligned"], components["pieces"], components["piece_of_spot"])
    src_labels = pareto["source_labels"]; src_map = {str(x): i for i, x in enumerate(src_labels)}
    source_domain_code = np.asarray([src_map[str(x)] for x in src_leiden], dtype=np.int32)
    selected_counts = pareto["routing_table"].groupby("source_domain")["pareto_selected"].sum()
    print(f"source domains: {len(src_labels)} | target domains: {len(pareto['target_labels'])}")
    print(f"Pareto-selected target domains per source: {selected_counts.describe().to_dict()}")
    print("\n[5] POST-HOC SUPPORT EVALUATION")
    summary, table = evaluate_support(pair_name, source, target, source_domain_code, pareto["support_by_domain"])
    display(pd.DataFrame([summary]).style.format({"support_fraction":"{:.2%}", "weighted_coverage":"{:.2%}", "macro_coverage":"{:.2%}", "weighted_purity":"{:.2%}", "macro_purity":"{:.2%}"}))
    save_stage1_outputs(pair_name, cfg, source, target, src_leiden, tgt_leiden, deformation, components, pareto, source_domain_code, outdir, summary, table)
    print(f"\n[done] {outdir}")
    return {"source": source, "target": target, "source_leiden": src_leiden, "target_leiden": tgt_leiden, "deformation": deformation, "components": components, "pareto": pareto, "source_domain_code": source_domain_code, "summary": summary, "table": table, "outdir": outdir}
