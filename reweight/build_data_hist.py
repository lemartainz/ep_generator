"""
Build the 3D data histogram consumed by the generator's `data_hist:` key.

The generator samples (Q2, E', theta_e[deg]) from a TH3D named `h_data`
(or whatever you pass as the second argument to `data_hist:`) via
TH3::GetRandom3. This script fills that TH3D from a CSV of real data
and writes it to a ROOT file.

CSV must have columns: Q2, Ep, theta_e   (theta_e in degrees)
(These are the column names produced by dump_kinematics.py, so you can
 dump your reconstructed ntuple to CSV the same way.)

Usage
-----
    python build_data_hist.py \\
        --data real_data.csv \\
        --out  data_hist.root \\
        --name h_data \\
        --q2   "1,6,40" \\
        --e    "1,9,40" \\
        --theta "5,35,40"

Then add this line to your generator input.txt (alongside the existing
Q2_range/E_range/theta_range keys, which are ignored when data_hist is
set):

    data_hist: reweight/data_hist.root h_data
"""

import argparse
import csv
import ROOT


def parse_bins(spec):
    lo, hi, n = spec.split(",")
    return float(lo), float(hi), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with Q2,Ep,theta_e")
    ap.add_argument("--out",  required=True, help="Output ROOT file")
    ap.add_argument("--name", default="h_data", help="TH3D name inside file")
    ap.add_argument("--q2",    required=True, help='"lo,hi,nbins"')
    ap.add_argument("--e",     required=True, help='"lo,hi,nbins"')
    ap.add_argument("--theta", required=True, help='"lo,hi,nbins" (degrees)')
    args = ap.parse_args()

    q2_lo, q2_hi, q2_n = parse_bins(args.q2)
    e_lo,  e_hi,  e_n  = parse_bins(args.e)
    th_lo, th_hi, th_n = parse_bins(args.theta)

    h = ROOT.TH3D(args.name,
                  "real data;Q^{2} [GeV^{2}];E' [GeV];#theta_{e} [deg]",
                  q2_n, q2_lo, q2_hi,
                  e_n,  e_lo,  e_hi,
                  th_n, th_lo, th_hi)

    n_in = n_fill = 0
    with open(args.data) as f:
        r = csv.DictReader(f)
        for col in ("Q2", "Ep", "theta_e"):
            if col not in r.fieldnames:
                raise SystemExit(
                    f"CSV missing column '{col}'. Has: {r.fieldnames}")
        for row in r:
            n_in += 1
            q2 = float(row["Q2"])
            ep = float(row["Ep"])
            th = float(row["theta_e"])
            if (q2_lo <= q2 < q2_hi and
                e_lo  <= ep < e_hi  and
                th_lo <= th < th_hi):
                h.Fill(q2, ep, th)
                n_fill += 1

    print(f"[build_data_hist] read {n_in} events, filled {n_fill} "
          f"({100.0*n_fill/max(n_in,1):.1f}%)")
    if n_fill == 0:
        raise SystemExit("No events fell inside the histogram range. "
                         "Widen --q2 / --e / --theta.")

    # Smooth very sparse bins a bit (optional; comment out if unwanted)
    # h.Smooth(1)

    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[build_data_hist] wrote {args.out}:{args.name}")


if __name__ == "__main__":
    main()
