#!/usr/bin/env python3
import argparse
import subprocess
import sys
from itertools import product


def main():
    parser = argparse.ArgumentParser(description="Run colour sweeps over a parameter grid")
    parser.add_argument("--template", default="pitzDaily_clean")
    parser.add_argument("--cpu-case", default="pitzDaily_cpu")
    parser.add_argument("--omegas", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--backs", default="0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--floors", default="1e-12,5e-12")
    parser.add_argument("--run-root", default="/home/xiaozhou/OpenFOAM/xiaozhou-10/run")
    parser.add_argument("--prefix", default="pitzDaily_gpu_colour_grid")
    parser.add_argument("--summary-check", action="store_true")
    parser.add_argument("--max-rel", default="1e-3")
    parser.add_argument("--max-iter", default="900")
    parser.add_argument("--max-time", default="260")
    args = parser.parse_args()

    omegas = [float(x) for x in args.omegas.split(',')]
    backs = [float(x) for x in args.backs.split(',')]
    floors = args.floors.split(',')

    combos = [
        (o, b, f)
        for o in omegas
        for b in backs
        for f in floors
        if b >= o
    ]

    for omega, back, floor in combos:
        case_prefix = f"{args.prefix}_o{omega:.2f}_b{back:.2f}_d{floor.replace('-', 'm')}"
        cmd = [
            "bin/gpu_colour_sweep.sh",
            "--run-root", args.run_root,
            "--template", args.template,
            "--cpu-case", args.cpu_case,
            "--prefix", case_prefix,
            "--omegas", f"{omega}",
            "--backward", f"{back}",
            "--floors", floor,
            "--log-iter"
        ]
        if args.summary_check:
            cmd += [
                "--summary-check",
                "--max-rel", args.max_rel,
                "--max-iter", args.max_iter,
                "--max-time", args.max_time
            ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
