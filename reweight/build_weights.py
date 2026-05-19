"""
Build a weight table w(x) = data(x) / mc(x) on an N-D histogram.

Usage:
  python build_weights.py \
      --mc   events.lund \
      --data real_data.csv \
      --vars Q2 W \
      --bins "1,6,25" "1.6,3.2,20" \
      --out  weights.npz

- MC input: a LUND file produced by the generator.
- Data input: a CSV with named columns matching --vars (e.g. Q2,W,xB,...).
  The columns can be computed however you like from your real data
  (ROOT->CSV dump, pandas, etc).
- Output: an .npz file containing bin edges, the weight grid, and the
  variable names. Consumed by reweight_lund.py.

The weights are normalized so that <w>_MC = 1, meaning the total number
of weighted MC events equals the original MC yield.
"""

import argparse
import csv
import numpy as np

from kinematics import compute_kinematics_batch


def parse_bins(spec: str):
    lo, hi, n = spec.split(",")
    return np.linspace(float(lo), float(hi), int(n) + 1)


def load_mc(path, varnames):
    k = compute_kinematics_batch(path)
    missing = [v for v in varnames if v not in k]
    if missing:
        raise ValueError(f"Unknown kinematic vars: {missing}. "
                         f"Available: {list(k.keys())}")
    return {v: np.asarray(k[v]) for v in varnames}


def load_data_csv(path, varnames):
    cols = {v: [] for v in varnames}
    with open(path) as f:
        reader = csv.DictReader(f)
        missing = [v for v in varnames if v not in reader.fieldnames]
        if missing:
            raise ValueError(f"Data CSV missing columns: {missing}. "
                             f"Has: {reader.fieldnames}")
        for row in reader:
            for v in varnames:
                cols[v].append(float(row[v]))
    return {v: np.asarray(cols[v]) for v in varnames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc",   required=True, help="MC LUND file")
    ap.add_argument("--data", required=True, help="Real data CSV")
    ap.add_argument("--vars", required=True, nargs="+",
                    help="Variable names (from kinematics.py)")
    ap.add_argument("--bins", required=True, nargs="+",
                    help='One "lo,hi,nbins" per variable')
    ap.add_argument("--out",  required=True, help="Output .npz weight file")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Laplace smoothing count added to MC bins")
    ap.add_argument("--wmax", type=float, default=20.0,
                    help="Clip weights above this value")
    args = ap.parse_args()

    if len(args.bins) != len(args.vars):
        raise SystemExit("--bins must have same length as --vars")

    edges = [parse_bins(b) for b in args.bins]

    mc   = load_mc(args.mc, args.vars)
    data = load_data_csv(args.data, args.vars)

    mc_sample   = np.stack([mc[v]   for v in args.vars], axis=1)
    data_sample = np.stack([data[v] for v in args.vars], axis=1)

    H_mc,   _ = np.histogramdd(mc_sample,   bins=edges)
    H_data, _ = np.histogramdd(data_sample, bins=edges)

    # Normalize to densities
    H_mc_n   = H_mc   / max(H_mc.sum(),   1.0)
    H_data_n = H_data / max(H_data.sum(), 1.0)

    # Ratio with smoothing
    denom = H_mc_n + args.smooth / max(H_mc.sum(), 1.0)
    W = np.where(H_mc > 0, H_data_n / denom, 0.0)
    W = np.clip(W, 0.0, args.wmax)

    # Normalize so that mean weight over MC events == 1
    # (preserves total MC count after reweighting)
    idxs = [np.clip(np.digitize(mc[v], edges[i]) - 1, 0, len(edges[i]) - 2)
            for i, v in enumerate(args.vars)]
    w_per_event = W[tuple(idxs)]
    mean_w = w_per_event.mean() if w_per_event.size else 1.0
    if mean_w > 0:
        W /= mean_w

    np.savez(args.out, weights=W,
             edges=np.array(edges, dtype=object),
             varnames=np.array(args.vars))
    print(f"[build_weights] vars={args.vars}")
    print(f"[build_weights] MC events:   {len(mc_sample)}")
    print(f"[build_weights] Data events: {len(data_sample)}")
    print(f"[build_weights] weight range: [{W.min():.3f}, {W.max():.3f}]")
    print(f"[build_weights] wrote {args.out}")


if __name__ == "__main__":
    main()
