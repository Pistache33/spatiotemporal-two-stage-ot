from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

STAGE_RE = re.compile(r"E(?P<stage>\d+(?:\.\d+)?)")


def stage_from_section(section_id: str) -> float:
    match = STAGE_RE.search(str(section_id))
    if match is None:
        raise ValueError(f"Cannot parse embryonic stage from section ID: {section_id!r}")
    return float(match.group("stage"))


def parse_pair_name(name: str) -> Tuple[float, float]:
    parts = Path(name).stem.split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected pair name like E9.5__E10.5, got {name!r}")
    return stage_from_section(parts[0]), stage_from_section(parts[1])


def load_grid_row(grid_path: str | Path, task_id: int) -> Dict[str, str]:
    with open(grid_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"task_id={task_id} outside [0, {len(rows) - 1}]")
    return rows[task_id]


def json_dump(obj: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=True)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def read_curated_transitions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    table = pd.read_csv(path, sep=separator)
    table = table.rename(columns={c: c.strip().lower().replace(" ", "_") for c in table.columns})
    required = {"cell_type", "known_descendants"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Transition table is missing columns: {sorted(missing)}")
    return table


def allowed_edges_from_table(path: str | Path, include_self: bool = True) -> Set[Tuple[str, str]]:
    table = read_curated_transitions(path)
    edges: Set[Tuple[str, str]] = set()
    for row in table.itertuples(index=False):
        source = str(getattr(row, "cell_type")).strip()
        raw = str(getattr(row, "known_descendants"))
        if include_self:
            edges.add((source, source))
        if raw and raw.lower() != "nan" and raw.strip() != "/":
            for target in raw.split(","):
                target = target.strip()
                if target and target != "/":
                    edges.add((source, target))
    return edges


def row_normalize(table: pd.DataFrame) -> pd.DataFrame:
    denominator = table.sum(axis=1).replace(0.0, np.nan)
    return table.div(denominator, axis=0).fillna(0.0)


def evaluate_transition_mass(
    transition_mass: pd.DataFrame,
    allowed_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    values = transition_mass.to_numpy(dtype=np.float64, copy=False)
    total = float(np.nansum(values))
    correct = 0.0
    for i, source in enumerate(transition_mass.index.astype(str)):
        for j, target in enumerate(transition_mass.columns.astype(str)):
            if (source, target) in allowed_edges:
                correct += float(values[i, j])
    wrong = total - correct
    return {
        "accuracy": correct / total if total > 0 else math.nan,
        "correct_mass": correct,
        "wrong_mass": wrong,
        "total_mass": total,
    }


def per_source_accuracy(
    transition_mass: pd.DataFrame,
    allowed_edges: Set[Tuple[str, str]],
) -> pd.DataFrame:
    records = []
    for source, row in transition_mass.iterrows():
        source = str(source)
        allowed_targets = [
            str(target)
            for target in transition_mass.columns
            if (source, str(target)) in allowed_edges
        ]
        correct = float(row.reindex(allowed_targets, fill_value=0.0).sum())
        total = float(row.sum())
        records.append(
            {
                "source_label": source,
                "correct_probability": correct / total if total > 0 else math.nan,
                "correct_mass": correct,
                "wrong_mass": total - correct,
                "row_mass": total,
                "allowed_targets_present": ";".join(allowed_targets),
            }
        )
    return pd.DataFrame(records)


def macro_accuracy(per_source: pd.DataFrame) -> float:
    values = per_source["correct_probability"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else math.nan


def aggregate_dense_transport(
    transport: np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    *,
    row_block_size: int = 256,
) -> pd.DataFrame:
    """Aggregate a dense/memmapped coupling into a source-label × target-label table.

    This is blockwise in source rows, so it never creates a second cell-level matrix.
    """
    source_labels_arr = np.asarray(source_labels, dtype=str)
    target_labels_arr = np.asarray(target_labels, dtype=str)
    if transport.shape != (len(source_labels_arr), len(target_labels_arr)):
        raise ValueError(
            f"Transport shape {transport.shape} does not match labels "
            f"({len(source_labels_arr)}, {len(target_labels_arr)})"
        )

    source_levels = np.asarray(sorted(pd.unique(source_labels_arr)), dtype=str)
    target_levels = np.asarray(sorted(pd.unique(target_labels_arr)), dtype=str)
    source_code_map = {label: i for i, label in enumerate(source_levels)}
    target_code_map = {label: i for i, label in enumerate(target_levels)}
    source_codes = np.fromiter((source_code_map[x] for x in source_labels_arr), dtype=np.int32)
    target_codes = np.fromiter((target_code_map[x] for x in target_labels_arr), dtype=np.int32)

    # First aggregate source rows while retaining target-cell columns.
    by_source = np.zeros((len(source_levels), transport.shape[1]), dtype=np.float64)
    for start in range(0, transport.shape[0], row_block_size):
        stop = min(start + row_block_size, transport.shape[0])
        block = np.asarray(transport[start:stop], dtype=np.float64)
        np.add.at(by_source, source_codes[start:stop], block)

    result = np.zeros((len(source_levels), len(target_levels)), dtype=np.float64)
    for target_code in range(len(target_levels)):
        mask = target_codes == target_code
        result[:, target_code] = by_source[:, mask].sum(axis=1)

    return pd.DataFrame(result, index=source_levels, columns=target_levels)


def coupling_diagnostics(
    transport: np.ndarray,
    *,
    row_block_size: int = 256,
    col_block_size: int = 256,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute row/column entropy and concentration summaries blockwise."""
    n_rows, n_cols = transport.shape
    row_sums = np.zeros(n_rows, dtype=np.float64)
    row_entropy = np.zeros(n_rows, dtype=np.float64)
    row_top1 = np.zeros(n_rows, dtype=np.float64)
    row_topk = np.zeros(n_rows, dtype=np.float64)

    for start in range(0, n_rows, row_block_size):
        stop = min(start + row_block_size, n_rows)
        block = np.asarray(transport[start:stop], dtype=np.float64)
        sums = block.sum(axis=1)
        row_sums[start:stop] = sums
        safe = np.where(block > 0.0, block, 1.0)
        entropy_numerator = -(np.where(block > 0.0, block * np.log(safe), 0.0)).sum(axis=1)
        positive = sums > 0
        row_entropy[start:stop][positive] = (
            entropy_numerator[positive] / sums[positive] + np.log(sums[positive])
        ) / np.log(max(n_cols, 2))
        row_top1[start:stop] = block.max(axis=1)
        k = min(top_k, n_cols)
        row_topk[start:stop] = np.partition(block, n_cols - k, axis=1)[:, -k:].sum(axis=1)

    col_sums = np.zeros(n_cols, dtype=np.float64)
    col_entropy = np.zeros(n_cols, dtype=np.float64)
    col_top1 = np.zeros(n_cols, dtype=np.float64)
    col_topk = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n_cols, col_block_size):
        stop = min(start + col_block_size, n_cols)
        block = np.asarray(transport[:, start:stop], dtype=np.float64)
        sums = block.sum(axis=0)
        col_sums[start:stop] = sums
        safe = np.where(block > 0.0, block, 1.0)
        entropy_numerator = -(np.where(block > 0.0, block * np.log(safe), 0.0)).sum(axis=0)
        positive = sums > 0
        col_entropy[start:stop][positive] = (
            entropy_numerator[positive] / sums[positive] + np.log(sums[positive])
        ) / np.log(max(n_rows, 2))
        col_top1[start:stop] = block.max(axis=0)
        k = min(top_k, n_rows)
        col_topk[start:stop] = np.partition(block, n_rows - k, axis=0)[-k:, :].sum(axis=0)

    def frac(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        out = np.full_like(denominator, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out

    def finite_mean(x: np.ndarray) -> float:
        return float(np.nanmean(x)) if np.isfinite(x).any() else math.nan

    def finite_median(x: np.ndarray) -> float:
        return float(np.nanmedian(x)) if np.isfinite(x).any() else math.nan

    row_top1_frac = frac(row_top1, row_sums)
    row_topk_frac = frac(row_topk, row_sums)
    col_top1_frac = frac(col_top1, col_sums)
    col_topk_frac = frac(col_topk, col_sums)

    return {
        "transport_sum": float(row_sums.sum()),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "col_sum_mean": float(col_sums.mean()),
        "col_sum_min": float(col_sums.min()),
        "col_sum_max": float(col_sums.max()),
        "row_entropy_norm_mean": finite_mean(row_entropy),
        "row_entropy_norm_median": finite_median(row_entropy),
        "col_entropy_norm_mean": finite_mean(col_entropy),
        "col_entropy_norm_median": finite_median(col_entropy),
        "row_top1_mass_frac_mean": finite_mean(row_top1_frac),
        "row_top1_mass_frac_median": finite_median(row_top1_frac),
        f"row_top{top_k}_mass_frac_mean": finite_mean(row_topk_frac),
        f"row_top{top_k}_mass_frac_median": finite_median(row_topk_frac),
        "col_top1_mass_frac_mean": finite_mean(col_top1_frac),
        "col_top1_mass_frac_median": finite_median(col_top1_frac),
        f"col_top{top_k}_mass_frac_mean": finite_mean(col_topk_frac),
        f"col_top{top_k}_mass_frac_median": finite_median(col_topk_frac),
    }


def read_name_file(path: str | Path) -> np.ndarray:
    return np.asarray(
        [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()],
        dtype=str,
    )


def write_name_file(values: Iterable[str], path: str | Path) -> None:
    Path(path).write_text("".join(f"{x}\n" for x in values), encoding="utf-8")
