"""
Dump per-event kinematics from a LUND file to CSV.

Handy for (a) sanity checks and (b) producing a CSV-format 'real data'
file from a reconstructed ROOT ntuple that you have already converted.

Usage:
  python dump_kinematics.py --in events.lund --out mc_kin.csv
  python dump_kinematics.py --in events.lund --out mc_kin.csv --vars Q2 W xB
"""

import argparse
import csv
import numpy as np
from kinematics import compute_kinematics_batch

DEFAULT_VARS = ["Q2", "W", "xB", "y", "nu", "theta_e", "phi_e", "Ep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    args = ap.parse_args()

    k = compute_kinematics_batch(args.inp)
    cols = [np.asarray(k[v]) for v in args.vars]
    n = len(cols[0])
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(args.vars)
        for i in range(n):
            writer.writerow([f"{c[i]:.6f}" for c in cols])
    print(f"wrote {args.out} ({n} rows)")


if __name__ == "__main__":
    main()
