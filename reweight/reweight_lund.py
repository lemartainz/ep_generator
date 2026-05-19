"""
Apply a weight table to a LUND file.

Two modes:

  1) 'accept'  (default): accept-reject to produce a new unweighted LUND
     whose distribution matches the target. Each event is kept with
     probability w / wmax. Output is a valid LUND usable directly by GEMC.

  2) 'sidecar': write every event unchanged and also emit a parallel
     text file listing the per-event weight (one weight per line). Use
     this if your downstream analysis can handle weighted events.

Usage:
  python reweight_lund.py --in events.lund --weights weights.npz \
      --out events_reweighted.lund --mode accept --seed 42

  python reweight_lund.py --in events.lund --weights weights.npz \
      --out events.lund --weights-out event_weights.txt --mode sidecar
"""

import argparse
import numpy as np

from lund_io import read_lund, write_lund
from kinematics import compute_kinematics


def load_weight_table(path):
    z = np.load(path, allow_pickle=True)
    W = z["weights"]
    edges = list(z["edges"])
    varnames = list(z["varnames"])
    return W, edges, varnames


def weight_for_event(ev, W, edges, varnames):
    k = compute_kinematics(ev)
    idx = []
    for i, v in enumerate(varnames):
        val = k[v]
        if val < edges[i][0] or val > edges[i][-1]:
            return 0.0  # outside weight table
        j = np.digitize([val], edges[i])[0] - 1
        j = min(max(j, 0), len(edges[i]) - 2)
        idx.append(j)
    return float(W[tuple(idx)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",      dest="inp",     required=True)
    ap.add_argument("--weights", required=True, help=".npz from build_weights.py")
    ap.add_argument("--out",     required=True, help="Output LUND")
    ap.add_argument("--mode",    choices=["accept", "sidecar"], default="accept")
    ap.add_argument("--weights-out", default=None,
                    help="(sidecar mode) file to write per-event weights")
    ap.add_argument("--seed",    type=int, default=1)
    args = ap.parse_args()

    W, edges, varnames = load_weight_table(args.weights)
    wmax = float(W.max())
    if wmax <= 0:
        raise SystemExit("Weight table is empty / all zero.")
    print(f"[reweight] vars={varnames}  wmax={wmax:.3f}")

    rng = np.random.default_rng(args.seed)

    if args.mode == "accept":
        def gen():
            kept = tot = 0
            for ev in read_lund(args.inp):
                tot += 1
                w = weight_for_event(ev, W, edges, varnames)
                if w <= 0:
                    continue
                if rng.random() < (w / wmax):
                    kept += 1
                    yield ev
            print(f"[reweight] accepted {kept}/{tot} "
                  f"({100.0*kept/max(tot,1):.1f}%)")
        n = write_lund(args.out, gen())
        print(f"[reweight] wrote {n} events -> {args.out}")

    else:  # sidecar
        if args.weights_out is None:
            raise SystemExit("--weights-out required in sidecar mode")
        with open(args.weights_out, "w") as wf:
            def gen():
                for ev in read_lund(args.inp):
                    w = weight_for_event(ev, W, edges, varnames)
                    wf.write(f"{w:.6f}\n")
                    yield ev
            n = write_lund(args.out, gen())
        print(f"[reweight] wrote {n} events -> {args.out}")
        print(f"[reweight] wrote per-event weights -> {args.weights_out}")


if __name__ == "__main__":
    main()
