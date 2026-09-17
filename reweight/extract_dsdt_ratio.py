"""
The extraction: dsigma/dt | s_rpbar  over  dsigma/dt | s_rp, from a generated sample.

For e p -> e' p_recoil p pbar with t = (target - recoil)^2, define two
sub-energies of the SAME event,

    s_rpbar = (recoil + pbar)^2        the pbar-p rescattering system
    s_rp    = (recoil + produced p)^2  the p-p rescattering system

and, exactly as one would with data, histogram t for the events whose
s_rpbar falls in an s bin, histogram t for the events whose s_rp falls in
that same s bin, and take the ratio bin by bin in t:

    R_extracted(s, t) = dN/dt | s_rpbar in [s_lo, s_hi]
                        ---------------------------------
                        dN/dt | s_rp    in [s_lo, s_hi]

That ratio is what tells the two rescatterings apart (pbar-p: steeper
slope, no quark interchange; p-p: flatter, wins at large |t|). This
script does that extraction on a truth ntuple -- weighted by w_ratio when
the run carried one -- and overlays the INPUT ratio R(s, t) at the bin
centers, so a controlled test reads as "extracted on input, or not".

Unweighted, the generator is symmetric under p <-> pbar from X, so the
two histograms are the same distribution and R_extracted == 1 within
statistics. Whatever the weighting scheme imprints, this is what the
data-style extraction sees of it.

Usage
-----
    python extract_dsdt_ratio.py truth.root --out ../dsdt_ratio.pdf \\
        [--formula "<num>" [--formula-den "<den>"] | --table f.root[:h] [--table-den ...]] \\
        [--s-edges 3.5,4.5,5.5,7,9,12.4] [--t-range=-14,0] [--nbins 35] [--unweighted]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ratio_weight import eval_st, table_at                  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="dsigma/dt|s_rpbar over "
                                 "dsigma/dt|s_rp from a generated sample.")
    ap.add_argument("gen", help="truth ntuple ROOT file")
    ap.add_argument("--tree", default="truth")
    ap.add_argument("--out", default="../dsdt_ratio.pdf")
    ap.add_argument("--s-edges", default=None,
                    help="comma list of s bin edges [GeV^2]; default: 6 "
                         "quantile bins of s_rpbar")
    ap.add_argument("--t-range", default=None, help='"lo,hi" (write --t-range=-14,0)')
    ap.add_argument("--nbins", type=int, default=35, help="t bins (default 35)")
    ap.add_argument("--unweighted", action="store_true",
                    help="ignore w_ratio even if the run carried one")
    ap.add_argument("--formula", default=None, help="input numerator model in s, t")
    ap.add_argument("--formula-den", default=None, help="input denominator model")
    ap.add_argument("--table", default=None, help="input numerator table file.root[:hist]")
    ap.add_argument("--table-den", default=None, help="input denominator table")
    ap.add_argument("--ratio-range", default="0,4", help='"lo,hi" for the ratio axes')
    ap.add_argument("--ncol", type=int, default=3)
    args = ap.parse_args()

    import uproot
    tr = uproot.open(args.gen)[args.tree]
    need = ["t", "s_pbarp", "s_pp", "w_ratio"]
    missing = [b for b in need if b not in tr.keys()]
    if missing:
        raise SystemExit(f"{args.gen}:{args.tree} lacks {missing}")
    a = tr.arrays(need, library="np")
    ok = np.isfinite(a["t"]) & np.isfinite(a["s_pbarp"]) & np.isfinite(a["s_pp"])
    t, s1, s2 = a["t"][ok], a["s_pbarp"][ok], a["s_pp"][ok]
    w = np.ones_like(t) if args.unweighted else a["w_ratio"][ok]
    carried = not np.all(w == 1.0)
    print(f"[gen] {args.gen}: {len(t)} events, "
          f"{'weighted by w_ratio (mean %.4f)' % w.mean() if carried else 'unweighted'}")

    if args.s_edges:
        s_edges = np.array([float(v) for v in args.s_edges.split(",")])
    else:
        s_edges = np.percentile(s1, np.linspace(0, 100, 7))
        s_edges[0], s_edges[-1] = min(s1.min(), s2.min()), max(s1.max(), s2.max())
    if args.t_range:
        lo, hi = (float(v) for v in args.t_range.split(","))
    else:
        lo, hi = float(t.min()), float(t.max())
    t_edges = np.linspace(lo, hi, args.nbins + 1)
    tc = 0.5 * (t_edges[:-1] + t_edges[1:])
    r_lo, r_hi = (float(v) for v in args.ratio_range.split(","))

    # input R(s, t) = num(s, t) / den(s, t) at the SAME s: what the
    # extraction is trying to measure
    def input_ratio(S, T):
        if args.formula:
            num = eval_st(args.formula, S, T)
            den = eval_st(args.formula_den or args.formula, S, T)
        elif args.table:
            num = table_at(args.table, S, T)
            den = table_at(args.table_den or args.table, S, T)
        else:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    ns = len(s_edges) - 1
    ncol = max(1, min(args.ncol, ns))
    nrow = int(np.ceil(ns / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 4.2 * nrow + 0.6))
    outer = GridSpec(nrow, ncol, figure=fig, hspace=0.45, wspace=0.3)
    summary = []

    for k in range(ns):
        s_lo, s_hi = s_edges[k], s_edges[k + 1]
        sel1 = (s1 >= s_lo) & (s1 < s_hi)      # events by their s_rpbar
        sel2 = (s2 >= s_lo) & (s2 < s_hi)      # events by their s_rp
        n1, _ = np.histogram(t[sel1], bins=t_edges, weights=w[sel1])
        n2, _ = np.histogram(t[sel2], bins=t_edges, weights=w[sel2])
        v1, _ = np.histogram(t[sel1], bins=t_edges, weights=w[sel1] ** 2)
        v2, _ = np.histogram(t[sel2], bins=t_edges, weights=w[sel2] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(n2 > 0, n1 / n2, np.nan)
            dR = R * np.sqrt(np.where(n1 > 0, v1 / n1**2, 0) + np.where(n2 > 0, v2 / n2**2, 0))
        s_c = 0.5 * (s_lo + s_hi)
        R_in = input_ratio(np.full_like(tc, s_c), tc)

        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // ncol, k % ncol],
                                        height_ratios=[2, 1.3], hspace=0.06)
        ax = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=ax)
        ax.step(t_edges, np.r_[n1, n1[-1]], where="post", color="C3", lw=1.5,
                label=r"$d N/dt\,|\,s_{r\bar p}$ in bin")
        ax.step(t_edges, np.r_[n2, n2[-1]], where="post", color="C0", lw=1.5,
                label=r"$d N/dt\,|\,s_{rp}$ in bin")
        ax.set(yscale="log", title=f"s in [{s_lo:.2f}, {s_hi:.2f}] GeV$^2$")
        ax.set_ylabel("events / bin", fontsize=9)
        if k == 0:
            ax.legend(frameon=False, fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)

        good = np.isfinite(R)
        axr.errorbar(tc[good], np.clip(R[good], r_lo, r_hi), yerr=dR[good], fmt="o",
                     ms=3, color="k", label="extracted")
        if R_in is not None:
            axr.plot(tc, R_in, "-", color="C2", lw=2, label="input R(s, t)")
        axr.axhline(1.0, color="0.6", lw=0.8)
        axr.set(ylim=(r_lo, r_hi), ylabel="ratio")
        axr.set_xlabel(r"$t$ [GeV$^2$]", fontsize=9)
        if k == 0:
            axr.legend(frameon=False, fontsize=8, loc="upper left")

        if R_in is not None and good.any():
            g = good & np.isfinite(R_in) & (dR > 0)
            pull = (R[g] - R_in[g]) / dR[g]
            summary.append((s_lo, s_hi, np.nanmean(R[good]), np.sqrt(np.mean(pull**2))))
        else:
            summary.append((s_lo, s_hi, np.nanmean(R[good]), np.nan))

    fig.suptitle(r"extracted  $\frac{d\sigma/dt\,|\,s_{r\bar p}}{d\sigma/dt\,|\,s_{rp}}$"
                 + ("  (sample weighted by $w_{ratio}$)" if carried else "  (unweighted sample)"),
                 fontsize=12, y=1.0)
    fig.savefig(args.out, bbox_inches="tight")
    print("      s bin              <R_extracted>   rms pull vs input")
    for s_lo, s_hi, m, p in summary:
        print(f"      [{s_lo:6.2f}, {s_hi:6.2f}]   {m:12.4f}   "
              + (f"{p:8.2f}" if np.isfinite(p) else "     n/a"))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
