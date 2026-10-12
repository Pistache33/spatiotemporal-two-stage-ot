"""Evaluation: method selection + per-cell-type metrics (final 3-metric version).

Extracted verbatim from `cell_type_evaluation.ipynb`:
  - cell 0: `best_moscot`, `best_socs`, `ours_summary` (weighted-best selection)
  - cell 1: `split_desc`, `load_allowed` (curated viable map)
  - cell 2: `read_names`, `align_to_adata`, `load_dense_transport_stats`,
    `load_sparse_ours_stats`, `per_type_metrics` (active_fraction /
    all-source curated-valid top1 / curated-valid transport-mass)

The curated-transition table is used ONLY as an evaluation reference; it does
not participate in model construction or parameter selection.

Original absolute paths (CURATED / MOSCOT_ROOT / SOCS_ROOT / OURS_ROOT) preserved.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import anndata as ad

CURATED = Path(__file__).resolve().parents[2] / "configs" / "curated_transitions.tsv"
# Original absolute path: /mnt/disk2/lirui/baseline_scripts/mosta_flatvi_fgw/MOSTA_curated_transitions.tsv

MOSCOT_ROOT = {
    "PairA": Path("/mnt/disk2/lirui/mosta_growth_marginal_ablation/no_growth_score/E9.5_E1S1__E10.5_E2S1/results/E9.5_E1S1__E10.5_E2S1/X_nbvae_30"),
    "PairC": Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/moscot/E10.5__E11.5/X_nbvae_30"),
}
SOCS_ROOT = {
    "PairA": Path("/mnt/disk2/lirui/mosta_socs_official_unsupervised/PairA"),
    "PairC": Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/socs/PairC"),
}
OURS_ROOT = {
    "PairA": Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/pareto/PairA/src1_tgt0p6/stage2"),
    "PairC": Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/pareto/PairC/src1_tgt0p6/stage2"),
}

ALLOWED = None  # populated at the bottom after load_allowed is defined


def best_moscot(root):
    rows = []
    for mp in sorted(root.glob("task_*/metrics.json")):
        m = json.loads(mp.read_text())
        if m.get("status") != "success":
            continue
        weighted = m.get("accuracy", m.get("weighted_accuracy"))
        macro = m.get("macro_curated_accuracy", m.get("macro_accuracy"))
        if weighted is None or macro is None:
            continue
        rows.append({
            "weighted": float(weighted), "macro": float(macro),
            "task_id": int(m["task_id"]), "task_dir": mp.parent,
        })
    if not rows:
        raise RuntimeError(f"No successful MOSCOT metrics under {root}")
    return pd.DataFrame(rows).sort_values(
        ["weighted","task_id"], ascending=[False,True]
    ).iloc[0]


def best_socs(root):
    grid = pd.read_csv(root / "grid_results.csv")
    row = grid.loc[grid["weighted_accuracy"].astype(float).idxmax()]
    return {
        "weighted": float(row["weighted_accuracy"]),
        "macro": float(row["macro_accuracy"]),
        "task_id": int(row["task_id"]),
        "task_dir": None,
    }


def ours_summary(root):
    hits = sorted(root.glob("*_summary.csv"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one summary under {root}, found {hits}")
    r = pd.read_csv(hits[0]).iloc[0]
    return {
        "weighted": float(r["mass_weighted"]),
        "macro": float(r["mass_macro"]),
    }


def split_desc(x):
    x = str(x)
    if x.lower()=="nan": return []
    return [z.strip() for z in re.split(r"[,;|\n]+",x) if z.strip() and z.strip()!="/"]


def load_allowed():
    df = pd.read_csv(CURATED,sep=None,engine="python")
    out = {}
    for _,r in df.iterrows():
        s = str(r["Cell_type"]).strip()
        if not s or s.lower()=="nan": continue
        out[s] = {s}|set(split_desc(r["Known_descendants"]))
    return out


def read_names(path):
    return np.atleast_1d(
        np.loadtxt(path, dtype=str)
    ).astype(str)


def align_to_adata(adata, saved_names):
    """
    Return CURRENT adata row indices corresponding to saved_names.
    """

    current = adata.obs_names.astype(str)

    idx = pd.Index(current).get_indexer(
        saved_names
    )

    if (
        np.any(idx < 0)
        and "original_obs_name" in adata.obs.columns
    ):
        original = (
            adata.obs["original_obs_name"]
            .astype(str)
            .to_numpy()
        )

        idx = pd.Index(original).get_indexer(
            saved_names
        )

    if np.any(idx < 0):
        bad = saved_names[idx < 0][:5]
        raise KeyError(
            f"Cannot align saved names, examples: {bad}"
        )

    return idx.astype(np.int64)


def load_dense_transport_stats(
    P_path,
    source,
    target,
    source_names_path,
    target_names_path,
    block_size=512,
):

    P = np.load(
        P_path,
        mmap_mode="r",
    )

    saved_src = read_names(
        source_names_path
    )

    saved_tgt = read_names(
        target_names_path
    )

    if P.shape != (
        len(saved_src),
        len(saved_tgt),
    ):
        raise ValueError(
            f"{P_path}\n"
            f"P shape={P.shape}, "
            f"names={(len(saved_src), len(saved_tgt))}"
        )

    # current H5AD indices
    src_current_idx = align_to_adata(
        source,
        saved_src,
    )

    tgt_current_idx = align_to_adata(
        target,
        saved_tgt,
    )

    src_ann_current = (
        source.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    tgt_ann_current = (
        target.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    # annotations in SAVED transport order
    src_ann_saved = src_ann_current[
        src_current_idx
    ]

    tgt_ann_saved = tgt_ann_current[
        tgt_current_idx
    ]

    ns = source.n_obs

    top1 = np.full(
        ns,
        -1,
        dtype=np.int32,
    )

    row_mass_current = np.zeros(
        ns,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # top1
    # --------------------------------------------------------

    for lo in range(
        0,
        P.shape[0],
        block_size,
    ):

        hi = min(
            lo + block_size,
            P.shape[0],
        )

        B = np.asarray(
            P[lo:hi],
            dtype=np.float64,
        )

        rm = B.sum(axis=1)

        current_rows = src_current_idx[
            lo:hi
        ]

        row_mass_current[
            current_rows
        ] = rm

        active = (
            np.isfinite(rm)
            &
            (rm > 0)
        )

        if np.any(active):

            local_rows = np.flatnonzero(
                active
            )

            saved_target_col = np.argmax(
                B[local_rows],
                axis=1,
            )

            current_target = tgt_current_idx[
                saved_target_col
            ]

            top1[
                current_rows[local_rows]
            ] = current_target

    # --------------------------------------------------------
    # annotation-level transported mass
    # --------------------------------------------------------

    s_types = np.unique(
        src_ann_saved
    )

    t_types = np.unique(
        tgt_ann_saved
    )

    mass_table = pd.DataFrame(
        0.0,
        index=s_types,
        columns=t_types,
    )

    for s in s_types:

        ii = np.flatnonzero(
            src_ann_saved == s
        )

        target_mass = np.asarray(
            P[ii, :],
            dtype=np.float64,
        ).sum(axis=0)

        for t in t_types:

            mass_table.loc[
                s, t
            ] = target_mass[
                tgt_ann_saved == t
            ].sum()

    return {
        "top1": top1,
        "row_mass": row_mass_current,
        "mass_table": mass_table,
    }


def load_sparse_ours_stats(
    stage2_dir,
    source,
    target,
):

    hits = sorted(
        Path(stage2_dir).glob(
            "*_transport_sparse.npz"
        )
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one sparse transport under "
            f"{stage2_dir}\nFound: {hits}"
        )

    S = np.load(
        hits[0]
    )

    offsets = np.asarray(
        S["row_offsets"],
        dtype=np.int64,
    )

    target_indices = np.asarray(
        S["target_indices"],
        dtype=np.int32,
    )

    mass = np.asarray(
        S["mass"],
        dtype=np.float64,
    )

    ns = source.n_obs
    nt = target.n_obs

    if len(offsets) != ns + 1:
        raise ValueError(
            f"Ours sparse rows={len(offsets)-1}, "
            f"source n={ns}"
        )

    src_ann = (
        source.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    tgt_ann = (
        target.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    top1 = np.full(
        ns,
        -1,
        dtype=np.int32,
    )

    row_mass = np.zeros(
        ns,
        dtype=np.float64,
    )

    s_types = np.unique(
        src_ann
    )

    t_types = np.unique(
        tgt_ann
    )

    mass_table = pd.DataFrame(
        0.0,
        index=s_types,
        columns=t_types,
    )

    for i in range(ns):

        lo = offsets[i]
        hi = offsets[i + 1]

        if hi <= lo:
            continue

        vals = mass[
            lo:hi
        ]

        jj = target_indices[
            lo:hi
        ]

        rm = float(
            vals.sum()
        )

        row_mass[i] = rm

        if (
            np.isfinite(rm)
            and
            rm > 0
        ):
            top1[i] = jj[
                np.argmax(vals)
            ]

        # annotation mass
        labs = tgt_ann[
            jj
        ]

        for t in np.unique(
            labs
        ):
            mass_table.loc[
                src_ann[i],
                t
            ] += vals[
                labs == t
            ].sum()

    return {
        "top1": top1,
        "row_mass": row_mass,
        "mass_table": mass_table,
    }


def per_type_metrics(
    source,
    target,
    stats,
):

    src_ann = (
        source.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    tgt_ann = (
        target.obs["annotation"]
        .astype(str)
        .to_numpy()
    )

    top1 = stats[
        "top1"
    ]

    row_mass = stats[
        "row_mass"
    ]

    mass_table = stats[
        "mass_table"
    ]

    rows = []

    for s in sorted(
        np.unique(src_ann)
    ):

        ii = np.flatnonzero(
            src_ann == s
        )

        n_source = len(ii)

        # ----------------------------------------------------
        # active source
        # ----------------------------------------------------

        active_mask = (
            np.isfinite(
                row_mass[ii]
            )
            &
            (
                row_mass[ii]
                >
                0
            )
            &
            (
                top1[ii]
                >=
                0
            )
        )

        active_idx = ii[
            active_mask
        ]

        n_active = len(
            active_idx
        )

        active_fraction = (
            n_active
            /
            n_source
        )

        # ----------------------------------------------------
        # ALL-SOURCE curated-valid top1
        #
        # Start every source as False.
        # Inactive source therefore remains incorrect.
        # ----------------------------------------------------

        viable = ALLOWED.get(
            str(s),
            {str(s)},
        )

        correct_all = np.zeros(
            n_source,
            dtype=bool,
        )

        if n_active > 0:

            active_positions = np.flatnonzero(
                active_mask
            )

            endpoint_labels = tgt_ann[
                top1[
                    active_idx
                ]
            ]

            correct_all[
                active_positions
            ] = np.asarray(
                [
                    str(x) in viable
                    for x in endpoint_labels
                ],
                dtype=bool,
            )

        top1_allsource = float(
            correct_all.mean()
        )

        n_top1_correct = int(
            correct_all.sum()
        )

        # ----------------------------------------------------
        # optional: conditional accuracy among active only
        # useful for diagnosis, NOT primary accuracy
        # ----------------------------------------------------

        top1_active_only = (
            n_top1_correct
            /
            n_active
            if n_active > 0
            else np.nan
        )

        # ----------------------------------------------------
        # transported-mass accuracy
        # ----------------------------------------------------

        if s in mass_table.index:

            mass_row = mass_table.loc[
                s
            ]

            total_mass = float(
                mass_row.sum()
            )

            viable_cols = [
                t
                for t in mass_row.index
                if str(t) in viable
            ]

            correct_mass = float(
                mass_row[
                    viable_cols
                ].sum()
            )

            mass_accuracy = (
                correct_mass
                /
                total_mass
                if total_mass > 0
                else np.nan
            )

        else:

            total_mass = 0.0
            correct_mass = 0.0
            mass_accuracy = np.nan

        rows.append({
            "source_annotation":
                str(s),

            "n_source":
                n_source,

            "n_active":
                n_active,

            "active_fraction":
                active_fraction,

            "n_top1_correct":
                n_top1_correct,

            "top1_allsource":
                top1_allsource,

            "top1_active_only":
                top1_active_only,

            "mass_accuracy":
                mass_accuracy,

            "total_transport_mass":
                total_mass,
        })

    return pd.DataFrame(
        rows
    )


ALLOWED = load_allowed()
