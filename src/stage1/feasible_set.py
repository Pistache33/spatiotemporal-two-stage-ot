"""Stage-1 feasible-set support + post-hoc support evaluation.

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 0). The support is built
inside `pareto_routing.build_pareto_supports`; this module holds the curated
viable map and the post-hoc coverage/purity evaluation.
"""
import numpy as np
import pandas as pd
from ._constants import (CURATED_PATH, ANNOTATION_KEY, EVAL_DROP_LABELS)


def split_descendants(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if not s or s in {"/", "-", "None", "nan", "NA"}: return []
    return [z.strip() for z in s.replace(";", ",").replace("|", ",").split(",") if z.strip()]


def load_viable_map():
    df = pd.read_csv(CURATED_PATH, sep="\t")
    if not {"Cell_type", "Known_descendants"}.issubset(df.columns): raise KeyError("curated table columns missing")
    viable = {}
    for _, row in df.iterrows():
        src = str(row["Cell_type"]).strip()
        viable[src] = ({src} | set(split_descendants(row["Known_descendants"]))) - EVAL_DROP_LABELS
    return viable


def evaluate_support(pair_name, source, target, source_domain_code, support_by_domain):
    src_ann = source.obs[ANNOTATION_KEY].astype(str).to_numpy()
    tgt_ann = target.obs[ANNOTATION_KEY].astype(str).to_numpy()
    viable_map = load_viable_map()
    rows = []
    for anatomy in pd.unique(src_ann):
        if anatomy in EVAL_DROP_LABELS: continue
        src_mask = src_ann == anatomy
        domains = np.unique(source_domain_code[src_mask])
        blocks = [support_by_domain[int(d)] for d in domains if len(support_by_domain[int(d)])]
        support = np.unique(np.concatenate(blocks)).astype(np.int32) if blocks else np.empty(0, dtype=np.int32)
        viable_targets = viable_map.get(anatomy, {anatomy}) - EVAL_DROP_LABELS
        present = {x for x in viable_targets if np.any(tgt_ann == x)}
        viable_mask = np.isin(tgt_ann, list(present))
        hit = int(np.sum(np.isin(np.arange(target.n_obs), support) & viable_mask))
        viable_n = int(viable_mask.sum())
        support_n = int(len(support))
        coverage = hit / viable_n if viable_n else np.nan
        purity = hit / support_n if support_n else np.nan
        rows.append({"pair": pair_name, "anatomy": anatomy, "n_source": int(src_mask.sum()), "n_source_domains": int(len(domains)), "viable_targets": ",".join(sorted(present)), "viable_n": viable_n, "feasible_n": support_n, "fraction_target": support_n / target.n_obs, "viable_hit_n": hit, "coverage": coverage, "purity": purity})
    table = pd.DataFrame(rows)
    eval_mask = ~np.isin(src_ann, list(EVAL_DROP_LABELS))
    support_sizes = np.asarray([len(support_by_domain[int(d)]) for d in source_domain_code], dtype=np.int64)
    summary = {
        "pair": pair_name, "n_source_eval": int(eval_mask.sum()),
        "support_fraction": float(support_sizes[eval_mask].sum() / (eval_mask.sum() * target.n_obs)),
        "weighted_coverage": float(table["viable_hit_n"].sum() / table["viable_n"].sum()) if table["viable_n"].sum() else np.nan,
        "macro_coverage": float(table["coverage"].dropna().mean()),
        "weighted_purity": float(table["viable_hit_n"].sum() / table["feasible_n"].sum()) if table["feasible_n"].sum() else np.nan,
        "macro_purity": float(table["purity"].dropna().mean()),
    }
    return summary, table
