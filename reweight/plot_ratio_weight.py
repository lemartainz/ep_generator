"""
Show what the carried ratio weight does to the t distribution.

Reads the truth ntuple of a run with `ratio_weight:` / `ratio_weight_formula:`
(branches w_ratio, t, s_pbarp, s_pp) and draws, in t:

    top     dN/dt unweighted and weighted by w_ratio (both normalized to
            unit area, so the panel compares SHAPES)
    middle  their ratio -- the mean w_ratio per t bin, i.e. the reweighting
            factor the pbar-p / p-p hypothesis applies at each t. The raw
            (not shape-normalized) mean is drawn too. With --formula (and
            optionally --formula-den) the INPUT ratio is overlaid: the
            same numpy expression(s) in s, t the generator was given,
            evaluated per event from the s_pbarp, s_pp, t branches and
            averaged in the same t bins -- so "input" and "extracted"
            are directly comparable.
    bottom  the spread of w_ratio at each t, as a 2-D histogram in log10 w


Usage
-----
    python plot_ratio_weight.py gen_truth.root [--out ../ratio_weight.pdf]
        [--tree truth] [--nbins 60] [--t-range=-14,0]
        [--formula "exp((4.0 + 0.5*log(s))*t) * pow(s,-2)" [--formula-den "..."]]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xsec                                                      # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.gridspec import GridSpec                         # noqa: E402


def eval_st(expr, S, T):
    """numpy expression in s, t (same whitelist as build_dsdt_table.py)."""
    ns = dict(xsec._SAFE_FUNCS)
    ns.update({"s": S, "t": T, "x": S, "y": T, "M_P": xsec.M_P})
    try:
        val = eval(expr, {"__builtins__": {}}, ns)                # noqa: S307
    except Exception as err:
        raise SystemExit(f"--formula failed: {err}")
    return np.broadcast_to(np.asarray(val, float), S.shape).astype(float)


def table_at(spec, S, T):
    """Evaluate a dsigma/dt TH2D the way the generator does (clamped
    bilinear between bin centers; exp() for a log table)."""
    import uproot
    from build_dsdt_table import table_lookup
    path, _, hist = spec.partition(":")
    f = uproot.open(path)
    if not hist:
        names = [k.split(";")[0] for k in f.keys()]
        if len(names) != 1:
            raise SystemExit(f"--table {path}: give the hist name, has {names}")
        hist = names[0]
    D, s_edges, t_edges = f[hist].to_numpy()
    val = table_lookup(D, s_edges, t_edges, S, T)
    if hist.startswith("log_"):
        val = np.exp(val)
    # outside the axis range the generator gives 0 (Surface2D::covers)
    inside = ((S > s_edges[0]) & (S < s_edges[-1]) &
              (T > t_edges[0]) & (T < t_edges[-1]))
    return np.where(inside, val, 0.0)


def main():
    ap = argparse.ArgumentParser(description="t distribution with and "
                                 "without the carried ratio weight.")
    ap.add_argument("gen", help="truth ntuple ROOT file")
    ap.add_argument("--tree", default="truth")
    ap.add_argument("--out", default="../ratio_weight.pdf")
    ap.add_argument("--nbins", type=int, default=60)
    ap.add_argument("--t-range", default=None,
                    help='"lo,hi" (write --t-range=-14,0); default: data range')
    ap.add_argument("--formula", default=None,
                    help="the input dsigma/dt(s, t) (numerator), to overlay "
                         "the input ratio")
    ap.add_argument("--formula-den", default=None,
                    help="separate denominator model (default: --formula at s_pp)")
    ap.add_argument("--table", default=None,
                    help="input dsigma/dt table file.root[:hist] (the one the "
                         "run used), to overlay the input ratio; log tables "
                         "are recognised by a hist name starting with log_")
    ap.add_argument("--table-den", default=None,
                    help="separate denominator table file.root[:hist]")
    args = ap.parse_args()
    if args.formula and args.table:
        raise SystemExit("give --formula or --table, not both")

    import uproot
    if not os.path.exists(args.gen):
        raise SystemExit(f"file not found: {args.gen}")
    tr = uproot.open(args.gen)[args.tree]
    need = ["w_ratio", "t", "s_pbarp", "s_pp"]
    missing = [b for b in need if b not in tr.keys()]
    if missing:
        raise SystemExit(f"{args.gen}:{args.tree} lacks {missing}")
    a = tr.arrays(need, library="np")
    ok = np.isfinite(a["t"]) & np.isfinite(a["w_ratio"])
    t, w = a["t"][ok], a["w_ratio"][ok]

    w_in = None
    if args.formula:
        num = eval_st(args.formula, a["s_pbarp"][ok], t)
        den = eval_st(args.formula_den or args.formula, a["s_pp"][ok], t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_in = np.where(den > 0, num / den, 0.0)
    elif args.table:
        num = table_at(args.table, a["s_pbarp"][ok], t)
        den = table_at(args.table_den or args.table, a["s_pp"][ok], t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_in = np.where(den > 0, num / den, 0.0)
    if w_in is not None:
        good = (w > 0) & (w_in > 0)
        rel = np.abs(w[good] / w_in[good] - 1.0)
        print(f"[input] per-event |w_gen/w_input - 1|: median {np.median(rel):.2e} "
              f"max {rel.max():.2e}")
    if np.all(w == 1.0):
        print("[warn] w_ratio == 1 for every event: this run carried no ratio "
              "weight. The plot will show a flat ratio.")

    if args.t_range:
        lo, hi = (float(v) for v in args.t_range.split(","))
    else:
        lo, hi = float(t.min()), float(t.max())
    edges = np.linspace(lo, hi, args.nbins + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    wd = np.diff(edges)

    n_u, _ = np.histogram(t, bins=edges)
    n_w, _ = np.histogram(t, bins=edges, weights=w)
    n_w2, _ = np.histogram(t, bins=edges, weights=w * w)
    n_u = n_u.astype(float)
    if w_in is not None:
        n_in, _ = np.histogram(t, bins=edges, weights=w_in)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_in = np.where(n_u > 0, n_in / n_u, np.nan)

    # mean weight per bin = weighted / unweighted, with its statistical error
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = np.where(n_u > 0, n_w / n_u, np.nan)
        var_w = np.where(n_u > 1, n_w2 / n_u - mean_w ** 2, np.nan)
        err_w = np.sqrt(np.clip(var_w, 0, None) / np.clip(n_u, 1, None))
        # shape-normalized ratio: how the t SHAPE changes, integral fixed
        shape_ratio = mean_w / (n_w.sum() / n_u.sum())

    n_zero = int(np.sum(w == 0))
    print(f"[gen] {args.gen}:{args.tree}: {len(t)} events, mean w_ratio = "
          f"{w.mean():.4f}, {n_zero} with w = 0; t in [{t.min():.3f}, {t.max():.3f}]")

    fig = plt.figure(figsize=(8.5, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 2, 2.4], hspace=0.08, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax0.step(edges, np.r_[n_u, n_u[-1]] / (n_u.sum() * wd[0]), where="post",
             color="0.35", lw=1.4, label="unweighted (generated t)")
    ax0.step(edges, np.r_[n_w, n_w[-1]] / (n_w.sum() * wd[0]), where="post",
             color="C3", lw=1.6, label=r"weighted by $w_{ratio}$")
    ax0.set(ylabel="dN/dt  (unit area)", yscale="log")
    ax0.legend(frameon=False)
    ax0.set_title(r"$w_{ratio} = \frac{d\sigma/dt(s_{\bar p p},\,t)}"
                  r"{d\sigma/dt(s_{pp},\,t)}$   effect on the t distribution",
                  fontsize=11)
    plt.setp(ax0.get_xticklabels(), visible=False)

    if w_in is not None:
        ax1.step(edges, np.r_[mean_in, mean_in[-1]], where="post", color="k",
                 lw=1.8, label="input ratio (from --formula / --table, same t bins)")
    ax1.errorbar(c, mean_w, yerr=err_w, fmt="o", ms=3, color="C3",
                 label=r"extracted: weighted / unweighted  $= \langle w_{ratio}\rangle(t)$")
    ax1.plot(c, shape_ratio, "-", color="C0", lw=1.2,
             label="same, shape-normalized (integral fixed)")
    ax1.axhline(1.0, color="0.5", lw=0.8)
    ax1.set(ylabel="ratio")
    ax1.legend(frameon=False, fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False)

    pos = w > 0
    lw = np.log10(w[pos])
    ax2.hist2d(t[pos], lw, bins=[edges, 80], cmap="viridis", cmin=1)
    ax2.axhline(0.0, color="w", lw=0.6, alpha=0.6)
    ax2.set(xlabel=r"$t = (p_{target} - p_{recoil})^2$  [GeV$^2$]",
            ylabel=r"$\log_{10} w_{ratio}$  per event")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
