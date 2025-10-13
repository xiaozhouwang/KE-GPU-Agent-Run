#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cudaPCG colour sweep summary metrics.")
    parser.add_argument("--file", required=True, help="Path to summary.csv produced by gpu_colour_sweep.sh")
    parser.add_argument("--max-rel", type=float, default=1e-2, help="Maximum allowed relL2(Ux)")
    parser.add_argument("--max-iter", type=float, default=1000.0, help="Maximum allowed PCG iteration count")
    parser.add_argument("--max-time", type=float, default=300.0, help="Maximum allowed ExecutionTime (seconds)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.file)
    if not summary_path.is_file():
        print(f"[check_colour_summary] Missing summary file: {summary_path}", file=sys.stderr)
        return 2

    failures = []
    with summary_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case = row.get('case', '<unknown>')
            rel = row.get('relL2_Ux', '')
            pcg_iter = row.get('pcgIterations', '')
            exec_time = row.get('executionTime', '')

            try:
                rel_val = float(rel)
            except ValueError:
                rel_val = float('inf')
            if rel_val > args.max_rel:
                failures.append(f"{case}: relL2_Ux={rel_val:.6e} > {args.max_rel:.6e}")

            try:
                iter_val = float(pcg_iter)
            except ValueError:
                iter_val = float('inf')
            if iter_val > args.max_iter:
                failures.append(f"{case}: pcgIterations={iter_val} > {args.max_iter}")

            try:
                time_val = float(exec_time)
            except ValueError:
                time_val = float('inf')
            if time_val > args.max_time:
                failures.append(f"{case}: ExecutionTime={time_val}s > {args.max_time}s")

    if failures:
        print("[check_colour_summary] FAIL:", file=sys.stderr)
        for entry in failures:
            print(f"  - {entry}", file=sys.stderr)
        return 1

    print(f"[check_colour_summary] PASS: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
