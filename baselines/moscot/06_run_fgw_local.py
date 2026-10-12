#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--transitions", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("/mnt/disk2/lirui/mosta_flatvi_fgw/results"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    rows = list(csv.DictReader(args.grid.open(encoding="utf-8"), delimiter="\t"))
    stop = len(rows) if args.stop is None else min(args.stop, len(rows))
    runner = Path(__file__).with_name("05_run_fgw_grid_point.py")
    for task in range(args.start, stop):
        cmd = [sys.executable, "-u", str(runner), "--grid", str(args.grid), "--task-id", str(task),
               "--transitions", str(args.transitions), "--output-root", str(args.output_root)]
        if args.overwrite: cmd.append("--overwrite")
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
