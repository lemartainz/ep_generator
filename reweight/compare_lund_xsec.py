"""
Compare a generator LUND file with the measured cross sections it was
weighted to -- the 3-D (Q2, W, M) comparison cell by cell, and the 1-D
projections onto Q2, W and each pair mass.

    python compare_lund_xsec.py \\
        --lund ../events_xsec_weighted.lund \\
        --target 2212,-2212 ../xsec3d_Mppbar_pooled.npz \\
        --target 2212,2212  ../xsec3d_Mpp.npz \\
        --weight ../pair_weight.root \\
        --unweighted ../events_unweighted.lund \\
        --out ../compare

writes

    <out>_3d_<pair>.pdf     page per Q2 bin, panel per W bin, M on x, ratio
                            underneath (plot_xsec_closure.draw_closure), in
                            the same normalized counts as the 1-D page
    <out>_1d.pdf            one page: Q2, W, and M for every target, each
                            with the measured cross section (with its
                            errors), the weighted LUND, the unweighted LUND
                            if given, and a ratio panel. NORMALIZED COUNTS:
                            every curve is divided by its own integral over
                            the live cells, so the y axis is the fraction of
                            entries per bin and only shapes are compared
                            (--y counts shows the LUND's raw counts instead,
                            with the cross section scaled to them).

Every histogram is filled the way the measurement was: M of EVERY (A, B)
pair in the event -- two entries per event for p pbar -- on the target's
own grid, and only entries inside the grid count. The LUND is scaled to
the target once, over the cells the weight allows (the <name>_fitted mask
in --weight, else every cell with generated events), so relative
normalization between bins is part of what is compared. Errors on the
ratio combine the generator's statistics with the target's.

Q2 and W are projections of the FIRST target (both targets come from the
same events, so their Q2 and W marginals agree up to the pooling factor);
--unweighted is drawn scaled to the same integral, to show what the
weighting did.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402

from truth_ntuple import load_truth, pair_masses, parse_pids, pair_label  # noqa: E402
from plot_xsec_closure import load_xsec, load_weight_mask, draw_closure   # noqa: E402

RED, BLUE, GREY = "#c0392b", "#1f3f8b", "0.55"


def pair_hist(a, pids, edges):
    """(Q2, W, M_AB) histogram of every (A, B) pair, on `edges`."""
    ev, m, _ = pair_masses(a, *pids)
    pts = np.column_stack([np.asarray(a["Q2"], float)[ev],
                           np.asarray(a["W"], float)[ev], m])
    pts = pts[np.isfinite(pts).all(axis=1)]
    H, _ = np.histogramdd(pts, bins=edges)
    return H


def tex_label(pids):
    return "$M(" + pair_label(*pids).replace("pbar", r"\bar{p}") + ")$"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lund", required=True, help="the weighted generator run")
    ap.add_argument("--target", action="append", nargs=2, required=True,
                    metavar=("PIDS", "NPZ"),
                    help="'<pidA,pidB> <xsec3d npz>', repeatable")
    ap.add_argument("--weight", default=None,
                    help="pair_weight.root the run used: its <name>_fitted masks "
                         "decide which cells enter the normalization and are hatched")
    ap.add_argument("--unweighted", default=None,
                    help="an unweighted LUND to overlay on the 1-D projections")
    ap.add_argument("--out", default="../compare", help="output prefix")
    ap.add_argument("--norm", choices=("global", "panel"), default="global")
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--panel-size", default="5.2,4.6",
                    help='"w,h" in inches of one W panel of the 3-D pages')
    ap.add_argument("--ratio-gap", type=float, default=0.22,
                    help="gap between each spectrum and its ratio panel "
                         "(matplotlib hspace, default 0.22)")
    ap.add_argument("--ratio-range", default="0.8,1.2")
    ap.add_argument("--units", choices=("integral", "density"), default=None,
                    help="how to read the npz values; default what the file records")
    ap.add_argument("--y", choices=("norm", "counts"), default="norm",
                    help="1-D y axis: norm (default) -- each curve divided by its "
                         "integral over the live cells; counts -- LUND entries per "
                         "bin, cross section scaled to the same total")
    args = ap.parse_args()
    r_lo, r_hi = (float(v) for v in args.ratio_range.split(","))

    gen = load_truth(args.lund, final_state=True)
    unw = load_truth(args.unweighted, final_state=True) if args.unweighted else None

    targets = []
    for spec, path in args.target:
        pids = parse_pids(spec)
        D, E, edges = load_xsec(path, args.units, measured=True)
        G = pair_hist(gen, pids, edges)
        U = pair_hist(unw, pids, edges) if unw is not None else None
        name = f"w_pair_{pids[0]}_{pids[1]}"
        if args.weight:
            live = load_weight_mask(args.weight, name + "_fitted", D.shape)
        else:
            live = G > 0
        scale = G[live].sum() / D[live].sum()
        tag = pair_label(*pids).replace(" ", "")
        print(f"[{tag}] {path}: {int(G.sum())} LUND entries on the grid, "
              f"{int(live.sum())}/{D.size} live cells, scale LUND/xsec = {scale:.4g}")
        targets.append(dict(pids=pids, D=D, E=E, edges=edges, G=G, U=U, live=live,
                            scale=scale, tag=tag, path=path))

    # ---- one common scale: LUND entries per target unit, from the single-
    # ---- entry target (or the first, if none is single-entry) ----
    ref = next((t for t in targets if t["pids"][0] == t["pids"][1]), targets[0])
    scale = ref["G"][ref["live"]].sum() / ref["D"][ref["live"]].sum()
    ydiv = ref["G"][ref["live"]].sum() if args.y == "norm" else None
    print(f"[scale] {scale:.4g} LUND entries per {os.path.basename(ref['path'])} unit, "
          f"from {ref['tag']} over its live cells")
    for t in targets:
        d = t["D"][t["live"]].sum(); g = t["G"][t["live"]].sum() / scale
        print(f"[norm] {t['tag']}: xsec integral over live cells {d:.4g}, LUND {g:.4g} "
              f"-> LUND/xsec = {g/d:.3f}")
    if len(targets) >= 2:
        a, b = targets[0], targets[1]
        rx_ = a["D"][a["live"]].sum() / b["D"][b["live"]].sum()
        rg_ = a["G"][a["live"]].sum() / b["G"][b["live"]].sum()
        print(f"[norm] {a['tag']} / {b['tag']} integral ratio: cross section {rx_:.3f}, "
              f"LUND {rg_:.3f}  (2 by construction, minus each map's dead cells)")

    same_grid = (len(targets) >= 2 and all(
        np.allclose(e1, e2) for e1, e2 in zip(targets[0]["edges"], targets[1]["edges"])))
    ynorm = args.y == "norm"

    # ---- 3-D: one PDF per target; the second target overlaid on the first ----
    for i, t in enumerate(targets):
        out = f"{args.out}_3d_{t['tag']}.pdf"
        ov = None
        if i == 0 and same_grid:
            o = targets[1]
            ov = dict(D=o["D"], E=o["E"], G=o["G"], live=o["live"],
                      label=tex_label(o["pids"]) + " cross section")
        rms, med = draw_closure(t["D"], t["E"], t["G"], t["edges"], t["live"], out,
                                norm=args.norm, ncol=args.ncol, ratio_range=(r_lo, r_hi),
                                tgt="cross section", m_label=tex_label(t["pids"]),
                                pair=t["pids"], ynorm=ynorm, scale=scale, ydiv=ydiv,
                                overlay=ov,
                                panel_size=tuple(float(v) for v in args.panel_size.split(",")),
                                ratio_gap=args.ratio_gap,
                                title=f"Weighted LUND, {tex_label(t['pids'])}"
                                      + (f"  (+ {tex_label(o['pids'])} overlaid)" if ov else ""))
        print(f"[3d] {out}: rms pull {rms:.2f}, median |LUND/xsec - 1| = {med:.3f}")

    # ---- 1-D projections, every target in every panel, common scale ----
    def proj(t, axis, arr):
        keep = np.where(t["live"], arr, 0.0)
        return keep.sum(axis=tuple(a for a in range(3) if a != axis))

    yf = (1.0 / ydiv) if ynorm else 1.0          # LUND counts -> y units

    def series(t, axis):
        T = proj(t, axis, t["D"]) * scale * yf
        TE = np.sqrt(proj(t, axis, t["E"] ** 2)) * scale * yf
        g = proj(t, axis, t["G"])
        G = (g * yf, g * yf ** 2)
        U = None
        if t["U"] is not None:
            u = proj(t, axis, t["U"]); U = u * (G[0].sum() / max(u.sum(), 1e-300))
        return T, TE, G, U

    ylabel = ("entries / bin  (norm. to " + tex_label(ref["pids"]) + ")"
              if ynorm else "LUND entries / bin")
    styles = [dict(tc=RED, gc=BLUE, fmt="o", mfc=BLUE), dict(tc="#d35400", gc="#1e8449", fmt="s", mfc="none")]
    axes_spec = [(0, r"$Q^2$ [GeV$^2$]", r"$Q^2$"), (1, "$W$ [GeV]", "$W$"), (2, "$M$ [GeV]", "pair mass")]
    ncol = 2
    nrow = int(np.ceil((len(axes_spec) + 1) / ncol))
    fig = plt.figure(figsize=(6.4 * ncol, 4.3 * nrow))
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    outer = GridSpec(nrow, ncol, figure=fig, hspace=0.38, wspace=0.28)
    for ip, (axis, xl, title) in enumerate(axes_spec):
        r, c = divmod(ip, ncol)
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c],
                                        height_ratios=[3, 1.15], hspace=args.ratio_gap)
        ax = fig.add_subplot(inner[0]); rx = fig.add_subplot(inner[1], sharex=ax)
        rx.axhline(1.0, color="0.3", lw=1.0)
        pulls = []
        for t, st in zip(targets, styles):
            edges = t["edges"][axis]; cc = 0.5 * (edges[:-1] + edges[1:])
            T, TE, G, U = series(t, axis)
            lab = tex_label(t["pids"]) + (" pooled" if t["pids"][0] != t["pids"][1] else "")
            if U is not None:
                ax.step(edges, np.r_[U, U[-1]], where="post", color=GREY, lw=1.0, ls=":",
                        label=f"unweighted LUND, {lab} (scaled)")
            ax.fill_between(cc, T, step="mid", color=st["tc"], alpha=0.10)
            ax.errorbar(cc, T, yerr=TE, fmt="none", ecolor=st["tc"], elinewidth=1, capsize=2)
            ax.step(cc, T, where="mid", color=st["tc"], lw=1.6, label=f"cross section, {lab}")
            ax.errorbar(cc, G[0], yerr=np.sqrt(G[1]), fmt=st["fmt"], ms=3.4, color=st["gc"],
                        mfc=st["mfc"], label=f"weighted LUND, {lab}")
            ok = T > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.where(ok, G[0] / T, np.nan)
                re = np.where(ok, np.sqrt(G[1] + TE ** 2) / T, np.nan)
            rx.errorbar(cc, rr, yerr=re, fmt=st["fmt"], ms=3, color=st["gc"], mfc=st["mfc"])
            pull = ((G[0] - T) / np.sqrt(G[1] + TE ** 2))[ok]
            pulls.append(f"{lab}: rms pull {np.sqrt((pull ** 2).mean()):.2f}")
            ax.set_xlim(edges[0], edges[-1])
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_ylim(0, None); ax.tick_params(labelsize=8, labelbottom=False)
        if ip == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        rx.set_ylim(r_lo, r_hi); rx.set_ylabel("LUND / xsec", fontsize=8)
        rx.set_xlabel(xl, fontsize=9); rx.tick_params(labelsize=8)
        rx.text(0.02, 0.78, "   ".join(pulls), transform=rx.transAxes, fontsize=7)
    if len(targets) >= 2:
        r, c = divmod(len(axes_spec), ncol)
        axt = fig.add_subplot(outer[r, c]); axt.axis("off")
        # axt.text(0.0, 0.95, "Relative normalization (one LUND scale for every target)\n\n"
        #          f"{tex_label(a['pids'])} pooled / {tex_label(b['pids'])}, integrals over live cells:\n\n"
        #          f"    cross section   {rx_:.3f}\n    weighted LUND   {rg_:.3f}\n\n"
        #          "2 by construction (two pairings per event vs one),\n"
        #          "minus what each map's dead / pass-through cells hold.",
        #          transform=axt.transAxes, fontsize=9.5, va="top", family="monospace")
    fig.suptitle(f"Generated vs Measured cross sections -- 1-D projections", fontsize=11, y=0.995)
    out1 = f"{args.out}_1d.pdf"
    with PdfPages(out1) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
