"""
Build a continuous 2-D weight function w(Q2, E') from real data, and
save it as a TH2D ROOT file for the generator to consume.

Method
------
- Fit a Gaussian KDE (scipy.stats.gaussian_kde) to (Q2, Ep) from the
  real-data CSV. This gives a smooth, continuous density estimate.
- Evaluate it on a fine regular grid over the generator's uniform
  sampling region (--q2-range, --e-range).
- Normalize so that the *maximum* of the surface equals 1.0, so the
  generator can use the value directly as an accept-reject probability.
- Store the grid as a TH2D (axes: Q2, E'). The generator calls
  TH2::Interpolate(Q2, Ep), which performs bilinear interpolation
  between bin centers — i.e. a continuous function, not a step function.

CSV must have columns: Q2, Ep

Usage
-----
    python build_weight_func.py \\
        --data real_data.csv \\
        --out  weight_func.root \\
        --name w_Q2_Ep \\
        --q2-range "0.5,9" \\
        --e-range  "1,9" \\
        --nx 200 --ny 200
"""

import argparse
import csv
import numpy as np
from scipy.stats import gaussian_kde
import ROOT


def parse_range(spec):
    lo, hi = spec.split(",")
    return float(lo), float(hi)


def load_csv(path, cols):
    out = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        for c in cols:
            if c not in r.fieldnames:
                raise SystemExit(
                    f"CSV missing '{c}'. Has: {r.fieldnames}")
        for row in r:
            for c in cols:
                out[c].append(float(row[c]))
    return {c: np.asarray(out[c]) for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with Q2,Ep columns")
    ap.add_argument("--out",  required=True, help="Output ROOT file")
    ap.add_argument("--name", default="w_Q2_Ep", help="TH2D name")
    ap.add_argument("--q2-range", required=True, help='"lo,hi"')
    ap.add_argument("--e-range",  required=True, help='"lo,hi"')
    ap.add_argument("--nx", type=int, default=200,
                    help="Grid points along Q2 (finer -> smoother interp)")
    ap.add_argument("--ny", type=int, default=200,
                    help="Grid points along E'")
    ap.add_argument("--bw", type=float, default=None,
                    help="KDE bandwidth factor (None = scipy default)")
    args = ap.parse_args()

    q2_lo, q2_hi = parse_range(args.q2_range)
    e_lo,  e_hi  = parse_range(args.e_range)

    d = load_csv(args.data, ["Q2", "Ep"])
    # keep only points inside range so the KDE isn't pulled by outliers
    mask = ((d["Q2"] >= q2_lo) & (d["Q2"] <= q2_hi) &
            (d["Ep"] >= e_lo)  & (d["Ep"] <= e_hi))
    pts = np.stack([d["Q2"][mask], d["Ep"][mask]], axis=0)
    print(f"[kde] {pts.shape[1]} data points inside range")
    if pts.shape[1] < 20:
        raise SystemExit("Too few data points for a KDE fit.")

    kde = gaussian_kde(pts, bw_method=args.bw)

    # Evaluate on grid. TH2D bin centers are at lo + (i+0.5)*dx.
    dx = (q2_hi - q2_lo) / args.nx
    dy = (e_hi  - e_lo)  / args.ny
    xs = q2_lo + (np.arange(args.nx) + 0.5) * dx
    ys = e_lo  + (np.arange(args.ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    zmax = Z.max()
    if zmax <= 0:
        raise SystemExit("KDE produced an all-zero surface.")
    Z /= zmax                       # now in [0, 1]
    print(f"[kde] surface normalized: max=1.0, mean={Z.mean():.3f}, "
          f"min={Z.min():.3e}")

    h = ROOT.TH2D(args.name,
                  "w(Q^{2},E');Q^{2} [GeV^{2}];E' [GeV]",
                  args.nx, q2_lo, q2_hi,
                  args.ny, e_lo,  e_hi)
    for ix in range(args.nx):
        for iy in range(args.ny):
            h.SetBinContent(ix + 1, iy + 1, float(Z[ix, iy]))

    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[kde] wrote {args.out}:{args.name}  "
          f"grid={args.nx}x{args.ny}  range "
          f"Q2=[{q2_lo},{q2_hi}]  E'=[{e_lo},{e_hi}]")


if __name__ == "__main__":
    main()
