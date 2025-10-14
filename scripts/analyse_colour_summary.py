#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistics from colour summary CSV.")
    parser.add_argument("summary", help="Path to summary.csv")
    parser.add_argument("--group-by-floor", action="store_true", help="Group statistics by diagonal floor as well")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.is_file():
        print(f"Missing summary file: {summary_path}")
        return 2

    with summary_path.open('r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    key_fields = ['omega', 'backwardOmega']
    if args.group_by_floor:
        key_fields.append('diagFloor')

    groups = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    print("key,avgIter,stdIter,minIter,maxIter,avgTime,avgRelL2")
    for key, items in sorted(groups.items()):
        iters = [float(item['pcgIterations']) for item in items if item['pcgIterations']]
        times = [float(item['executionTime']) for item in items if item['executionTime']]
        rels = [float(item['relL2_Ux']) for item in items if item['relL2_Ux']]
        avg_iter = mean(iters) if iters else 0.0
        std_iter = pstdev(iters) if len(iters) > 1 else 0.0
        min_iter = min(iters) if iters else 0.0
        max_iter = max(iters) if iters else 0.0
        avg_time = mean(times) if times else 0.0
        avg_rel = mean(rels) if rels else 0.0
        key_str = ";".join(key)
        print(f"{key_str},{avg_iter:.2f},{std_iter:.2f},{min_iter:.0f},{max_iter:.0f},{avg_time:.2f},{avg_rel:.3e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
