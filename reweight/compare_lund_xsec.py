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

4-D targets, (Q2, W, M, -t') with -t' that of the particle the pair leaves
out (build_pair_weight.py), add

    <out>_tp_<pair>.pdf     the same pages with -t' on x, M summed out

while the _3d pages show M with -t' summed out, and the 1-D page gets a
-t' panel. Both sums run over EVERY cell: a bin is hatched when less than
half of its cross section sits in cells the builder fitted (the rest got the
pass-through factor and follow the proposal, not the data). The
normalization still uses the fitted cells only, and the 1-D panels draw the
fitted-only cross section dashed.
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

from truth_ntuple import (load_truth, pair_masses, pair_entries, parse_pids,  # noqa: E402
                          pair_label)
from plot_xsec_closure import load_xsec, load_weight_mask, draw_closure   # noqa: E402

RED, BLUE, GREY = "#c0392b", "#1f3f8b", "0.55"
M_PROTON = 0.93827208816


def pair_hist(a, pids, edges):
    """(Q2, W, M_AB) histogram of every (A, B) pair, on `edges` -- or
    (Q2, W, M_AB, -t') for a 4-D grid, -t' that of the rest of the event."""
    cols = [np.asarray(a["Q2"], float), np.asarray(a["W"], float)]
    if len(edges) == 4:
        ev, m, tp, _ = pair_entries(a, *pids, a.get("target_mass", M_PROTON))
        pts = np.column_stack([c[ev] for c in cols] + [m, tp])
    else:
        ev, m, _ = pair_masses(a, *pids)
        pts = np.column_stack([c[ev] for c in cols] + [m])
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
                    help="'<pidA,pidB> <xsec npz>', repeatable; 3-D (Q2, W, M) "
                         "or 4-D (Q2, W, M, -t')")
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
        D, E, edges = load_xsec(path, args.units, measured=True, dims=(3, 4))
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

    def view(t, drop):
        """(D, E, G, live, edges) on (Q2, W, x): as is for a 3-D target; for a
        4-D one with axis `drop` summed out over EVERY cell, and a bin counted
        as fitted (not hatched) when at least half of its cross section sits
        in cells the builder fitted."""
        if drop is None:
            return t["D"], t["E"], t["G"], t["live"], t["edges"]
        D = t["D"].sum(axis=drop)
        fitted = np.where(t["live"], t["D"], 0.0).sum(axis=drop)
        with np.errstate(divide="ignore", invalid="ignore"):
            live = np.where(D > 0, fitted / D >= 0.5, t["live"].any(axis=drop))
        return (D, np.sqrt((t["E"] ** 2).sum(axis=drop)), t["G"].sum(axis=drop), live,
                [e for a, e in enumerate(t["edges"]) if a != drop])

    # ---- cell by cell: one PDF per target and x variable (M; -t' for a 4-D
    # ---- target); the second target overlaid on the first ----
    for i, t in enumerate(targets):
        o = targets[1] if (i == 0 and same_grid) else None
        four = t["D"].ndim == 4
        pages = [("3d", 3 if four else None, tex_label(t["pids"]), "GeV", True)]
        if four:
            pages.append(("tp", 2, r"$-t'$ of the rest", "GeV$^2$", False))
        for kind, drop, xlab, unit, edge in pages:
            D, E, G, live, edges = view(t, drop)
            ov = None
            if o is not None:
                oD, oE, oG, olive, _ = view(o, drop)
                ov = dict(D=oD, E=oE, G=oG, live=olive,
                          label=tex_label(o["pids"]) + " cross section")
            out = f"{args.out}_{kind}_{t['tag']}.pdf"
            rms, med = draw_closure(D, E, G, edges, live, out,
                                    norm=args.norm, ncol=args.ncol, ratio_range=(r_lo, r_hi),
                                    tgt="cross section", m_label=xlab,
                                    pair=t["pids"], ynorm=ynorm, scale=scale, ydiv=ydiv,
                                    overlay=ov, x_unit=unit, kin_edge=edge,
                                    panel_size=tuple(float(v) for v in args.panel_size.split(",")),
                                    ratio_gap=args.ratio_gap,
                                    title=f"Weighted LUND, {tex_label(t['pids'])}"
                                          + (f"  (+ {tex_label(o['pids'])} overlaid)" if ov else ""))
            print(f"[{kind}] {out}: rms pull {rms:.2f}, median |LUND/xsec - 1| = {med:.3f}")

    # ---- 1-D projections, every target in every panel, common scale ----
    def proj(t, axis, arr, fitted_only=False):
        keep = np.where(t["live"], arr, 0.0) if fitted_only else arr
        return keep.sum(axis=tuple(a for a in range(keep.ndim) if a != axis))

    yf = (1.0 / ydiv) if ynorm else 1.0          # LUND counts -> y units

    def series(t, axis):
        T = proj(t, axis, t["D"]) * scale * yf
        TE = np.sqrt(proj(t, axis, t["E"] ** 2)) * scale * yf
        TF = proj(t, axis, t["D"], fitted_only=True) * scale * yf
        g = proj(t, axis, t["G"])
        G = (g * yf, g * yf ** 2)
        U = None
        if t["U"] is not None:
            u = proj(t, axis, t["U"]); U = u * (G[0].sum() / max(u.sum(), 1e-300))
        return T, TE, TF, G, U

    ylabel = ("entries / bin  (norm. to " + tex_label(ref["pids"]) + ")"
              if ynorm else "LUND entries / bin")
    styles = [dict(tc=RED, gc=BLUE, fmt="o", mfc=BLUE), dict(tc="#d35400", gc="#1e8449", fmt="s", mfc="none")]
    axes_spec = [(0, r"$Q^2$ [GeV$^2$]", r"$Q^2$"), (1, "$W$ [GeV]", "$W$"), (2, "$M$ [GeV]", "pair mass")]
    if all(t["D"].ndim == 4 for t in targets):
        axes_spec.append((3, r"$-t'$ [GeV$^2$]", r"$-t'$ of the rest of the event"))
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
            T, TE, TF, G, U = series(t, axis)
            unfit = 1.0 - TF.sum() / max(T.sum(), 1e-300)
            lab = tex_label(t["pids"]) + (" pooled" if t["pids"][0] != t["pids"][1] else "")
            if U is not None:
                ax.step(edges, np.r_[U, U[-1]], where="post", color=GREY, lw=1.0, ls=":",
                        label=f"unweighted LUND, {lab} (scaled)")
            ax.fill_between(cc, T, step="mid", color=st["tc"], alpha=0.10)
            ax.errorbar(cc, T, yerr=TE, fmt="none", ecolor=st["tc"], elinewidth=1, capsize=2)
            ax.step(cc, T, where="mid", color=st["tc"], lw=1.6, label=f"cross section, {lab}")
            if unfit > 1e-3:
                ax.step(cc, TF, where="mid", color=st["tc"], lw=1.0, ls="--",
                        label=f"fitted cells only, {lab}")
            ax.errorbar(cc, G[0], yerr=np.sqrt(G[1]), fmt=st["fmt"], ms=3.4, color=st["gc"],
                        mfc=st["mfc"], label=f"weighted LUND, {lab}")
            ok = T > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.where(ok, G[0] / T, np.nan)
                re = np.where(ok, np.sqrt(G[1] + TE ** 2) / T, np.nan)
            rx.errorbar(cc, rr, yerr=re, fmt=st["fmt"], ms=3, color=st["gc"], mfc=st["mfc"])
            pull = ((G[0] - T) / np.sqrt(G[1] + TE ** 2))[ok]
            pulls.append(f"{lab}: rms pull {np.sqrt((pull ** 2).mean()):.2f}"
                         + (f", {unfit:.0%} unfitted" if unfit > 1e-3 else ""))
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
