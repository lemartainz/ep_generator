"""
Overlay the GENERATED distribution on the CROSS SECTION, with a ratio panel.

This is the closure test for the 3-D weighting: after a generator run that
used `xsec_weight:`, the accepted events should follow the cross section
that built the weight. The layout mirrors make_pseudo_xsec.py so the two
PDFs can be flipped through side by side:

    page i  ->  Q2 bin i
      each cell on the page  ->  one W bin
      upper axes             ->  generated points over the cross-section curve
      lower axes             ->  their ratio, generated / cross section
      x axis                 ->  M

Normalization
-------------
--norm global (default): the cross section is scaled by ONE factor for the
whole file, chosen so its total over the grid equals the total number of
generated events. Every panel then tests the shape in M *and* the relative
normalization between Q2 and W bins -- which is the whole point of a 3-D
weight. A panel sitting systematically above or below 1 means that
(Q2, W) cell got the wrong share of events, even if its M shape is perfect.

--norm panel: each panel is normalized to its own generated count, so only
the M shape is tested. Use it to separate a shape problem from a
normalization problem.

Pass --weight (the TH3D the run actually used) so the scale is computed
over the cells the weight ALLOWS. Cells where w = 0 -- zeroed by the
low-statistics guards, or unreachable by the generator -- can never be
populated, and folding their cross section into the normalization biases
every other cell upward by exactly the fraction they hold. With the
default guards that is ~15% here, which would otherwise read as a 15%
closure failure everywhere. Those cells are drawn hatched: the cross
section is there, the generator cannot deliver it.

Usage
-----
    python plot_xsec_closure.py \\
        --gen  gen_truth_weighted.root \\
        --xsec ../pseudo_xsec.npz \\
        --out  ../xsec_closure.pdf
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402

M_P = 0.9382720813


def load_xsec(path, units_arg):
    """Load the cross section and convert it to a per-bin yield.

    Must match what build_xsec_weight3d.py did, or the closure compares
    against a differently-normalized target and reads as a failure on
    every non-uniform bin.
    """
    z = np.load(path, allow_pickle=True)
    edges = [np.asarray(e, dtype=float) for e in z["edges"]]
    if len(edges) != 3:
        raise SystemExit(f"--xsec {path}: expected 3-D, got {len(edges)}-D.")
    D = np.clip(np.asarray(z["counts"], float), 0.0, None)
    units = units_arg or (str(z["units"]) if "units" in z else None)
    xsec.warn_if_ambiguous(edges, units is not None, "xsec")
    return xsec.to_bin_yield(D, edges, units or "integral", "xsec"), edges


def load_gen(path, tree, edges):
    """Bin the truth ntuple on the cross section's own grid."""
    import uproot
    if not os.path.exists(path):
        raise SystemExit(f"--gen: file not found: {path}")
    t = uproot.open(path)[tree]
    a = t.arrays(["Q2", "W", "M"], library="np")
    pts = np.column_stack([a["Q2"], a["W"], a["M"]])
    pts = pts[np.isfinite(pts).all(axis=1)]
    H, _ = np.histogramdd(pts, bins=edges)
    print(f"[gen] {path}:{tree}: {len(pts)} rows, {int(H.sum())} inside the "
          f"grid ({100.0*H.sum()/max(len(pts),1):.1f}%)")
    return H


def load_weight_mask(path, name, shape):
    """Boolean array of cells the weight allows (w > 0)."""
    import ROOT
    if not os.path.exists(path):
        raise SystemExit(f"--weight: file not found: {path}")
    tf = ROOT.TFile(path)
    h = tf.Get(name)
    if not h:
        tf.Close()
        raise SystemExit(f"--weight: no TH3 named '{name}' in {path}")
    got = (h.GetNbinsX(), h.GetNbinsY(), h.GetNbinsZ())
    if got != shape:
        tf.Close()
        raise SystemExit(f"--weight grid {got} != cross-section grid {shape}.")
    w = np.array([[[h.GetBinContent(i + 1, j + 1, k + 1)
                    for k in range(shape[2])]
                   for j in range(shape[1])]
                  for i in range(shape[0])], dtype=float)
    tf.Close()
    return w > 0


def main():
    ap = argparse.ArgumentParser(
        description="Generated vs cross section, with ratio panels.")
    ap.add_argument("--gen", required=True,
                    help="truth ntuple from the WEIGHTED generator run")
    ap.add_argument("--tree", default="truth")
    ap.add_argument("--xsec", required=True,
                    help="the 3-D cross-section npz the weight was built from")
    ap.add_argument("--out", default="../xsec_closure.pdf")
    ap.add_argument("--norm", choices=("global", "panel"), default="global",
                    help="global (default): one scale factor for the whole "
                         "file, so relative normalization between bins is "
                         "tested too. panel: per-panel, testing M shape only.")
    ap.add_argument("--weight", default=None,
                    help="the weight TH3D the run used. Cells with w = 0 are "
                         "excluded from the normalization and drawn hatched: "
                         "they cannot be populated, so counting their cross "
                         "section would bias every other cell upward.")
    ap.add_argument("--weight-name", default="w_Q2_W_M")
    ap.add_argument("--xsec-units", choices=("integral", "density"),
                    default=None,
                    help="must match what build_xsec_weight3d.py used; "
                         "default is whatever the npz records")
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--ratio-range", default="0.8,1.2",
                    help='"lo,hi" for the ratio axes (default 0.8,1.2). '
                         "Points outside are drawn as triangles on the "
                         "boundary rather than silently dropped.")
    args = ap.parse_args()

    D, edges = load_xsec(args.xsec, args.xsec_units)
    G = load_gen(args.gen, args.tree, edges)
    q2e, we, me = edges
    nq, nw, nm = D.shape
    mc = 0.5 * (me[:-1] + me[1:])
    r_lo, r_hi = (float(v) for v in args.ratio_range.split(","))

    if D.sum() <= 0:
        raise SystemExit("cross section is empty")

    # Which cells the weight allows. A w = 0 cell is not a closure failure,
    # it is a cell the generator was told never to fill -- so it must be
    # left out of the scale, or its cross section inflates every other cell.
    if args.weight:
        live = load_weight_mask(args.weight, args.weight_name, D.shape)
        src = f"w > 0 in {os.path.basename(args.weight)}"
    else:
        live = G > 0
        src = "cells with generated events (pass --weight to be exact)"
    lost = float(D[~live].sum() / D.sum())
    print(f"[norm] live cells: {int(live.sum())}/{D.size} ({src}); "
          f"they hold {100*(1-lost):.1f}% of the cross section")
    if lost > 0.01:
        print(f"[warn] {100*lost:.1f}% of the cross section sits in cells the "
              "weight zeroes (guards, or unreachable by the generator). Those "
              "are excluded from the scale and hatched in the plots -- the "
              "shape is reproduced, that fraction of the rate is not.")

    scale = G[live].sum() / D[live].sum()
    print(f"[norm] {args.norm}; global scale = {scale:.4g} "
          f"({int(G[live].sum())} generated / {D[live].sum():.4g} cross "
          "section, over live cells)")

    dead = (G <= 0) & (D > 0) & live
    if dead.any():
        print(f"[warn] {int(dead.sum())} cells are allowed by the weight but "
              "got no events -- undersampled, run more.")

    pdf = PdfPages(args.out)
    nrow = int(np.ceil(nw / args.ncol))
    for iq in range(nq):
        fig = plt.figure(figsize=(8.5, 11.0))
        # Nested grid: the OUTER cells are generously spaced so a ratio
        # panel's tick labels clear the title of the row beneath it, while
        # each inner pair (spectrum + its ratio) stays visually joined.
        outer = GridSpec(nrow, args.ncol, figure=fig,
                         hspace=0.46, wspace=0.34)

        for iw in range(nw):
            r, c = divmod(iw, args.ncol)
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c],
                height_ratios=[3, 1.15], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            rx = fig.add_subplot(inner[1], sharex=ax)

            gen = G[iq, iw]
            lv = live[iq, iw]
            pred = D[iq, iw] * scale
            if args.norm == "panel" and D[iq, iw][lv].sum() > 0 and gen.sum() > 0:
                pred = D[iq, iw] * (gen.sum() / D[iq, iw][lv].sum())

            err = np.sqrt(np.maximum(gen, 0.0))
            # rms pull for this panel -- computed here so it can go in the
            # title, where it cannot sit on top of the ratio points.
            okp = (pred > 0) & (gen > 0) & lv
            rms = (np.sqrt((((gen[okp] - pred[okp]) / err[okp]) ** 2).mean())
                   if okp.sum() else np.nan)

            # cross section: the curve being matched. Where the weight is
            # zero it is drawn hatched -- present in the model, impossible
            # for the generator to produce.
            ax.step(mc, pred, where="mid", color="#c0392b", lw=1.4,
                    zorder=3, label="cross section")
            ax.fill_between(mc, pred, step="mid", color="#c0392b",
                            alpha=0.12, zorder=1, where=lv)
            if (~lv).any():
                ax.fill_between(mc, pred, step="mid", where=~lv,
                                facecolor="none", edgecolor="#c0392b",
                                hatch="////", lw=0.0, alpha=0.55, zorder=2)
            # generated: points with statistical errors
            ax.errorbar(mc, gen, yerr=err, fmt="o", ms=2.4, lw=0.9,
                        color="#1f3f8b", zorder=4, label="generated")

            ttl = f"W = {we[iw]:.2f} - {we[iw+1]:.2f} GeV"
            if np.isfinite(rms):
                ttl += f"    pull {rms:.1f}"
            ax.set_title(ttl, fontsize=8.5, pad=3,
                         color="#b03030" if np.isfinite(rms) and rms >= 2
                         else "0.1")
            ax.tick_params(labelsize=7, labelbottom=False)
            ax.set_xlim(me[0], me[-1])
            top = max(pred.max(), (gen + err).max()) if gen.size else 1.0
            ax.set_ylim(0, top * 1.25 if top > 0 else 1.0)

            # the kinematic edge, same marking as the cross-section PDF
            m_edge_hi = we[iw + 1] - M_P
            for a in (ax, rx):
                if m_edge_hi < me[-1]:
                    a.axvspan(max(m_edge_hi, me[0]), me[-1], color="0.85",
                              alpha=0.7, lw=0, zorder=0)

            # ---- ratio ----
            with np.errstate(divide="ignore", invalid="ignore"):
                good = (pred > 0) & lv
                ratio = np.where(good, gen / pred, np.nan)
                rerr = np.where(good, err / pred, np.nan)
            rx.axhline(1.0, color="#c0392b", lw=1.1, zorder=2)
            rx.errorbar(mc, ratio, yerr=rerr, fmt="o", ms=2.2, lw=0.8,
                        color="#1f3f8b", zorder=3)
            # Anything outside the ratio window is marked on the boundary
            # so a badly-off cell cannot vanish off the top of the panel.
            hi_out = np.asarray(ratio > r_hi)
            lo_out = np.asarray((ratio < r_lo) & np.isfinite(ratio))
            if hi_out.any():
                rx.plot(mc[hi_out], np.full(hi_out.sum(), r_hi), "^",
                        ms=3.0, color="#b03030", clip_on=False, zorder=5)
            if lo_out.any():
                rx.plot(mc[lo_out], np.full(lo_out.sum(), r_lo), "v",
                        ms=3.0, color="#b03030", clip_on=False, zorder=5)

            rx.set_ylim(r_lo, r_hi)
            rx.set_xlim(me[0], me[-1])
            rx.tick_params(labelsize=6.5)
            # Ticks that suit whatever window was asked for: the midpoint
            # plus one step either side, rounded to something readable.
            step = (r_hi - r_lo) / 4.0
            rx.set_yticks([round(1.0 - step, 4), 1.0, round(1.0 + step, 4)])
            rx.set_ylabel("gen/xs", fontsize=6.5, labelpad=1)
            rx.set_xlabel(r"$M_{p\bar{p}}$ [GeV]", fontsize=7.5, labelpad=1)
            if c == 0:
                ax.set_ylabel("events / bin", fontsize=8)
            if iw == 0:
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

        fig.suptitle(
            "Generated vs cross section"
            "\n"
            rf"$Q^2$ = {q2e[iq]:.2f} $-$ {q2e[iq+1]:.2f} GeV$^2$"
            f"   (page {iq+1} of {nq})",
            fontsize=11.5, y=0.978)
        fig.text(0.5, 0.016,
                 f"normalization: {args.norm}"
                 + ("  (one scale for the whole file -- relative "
                    "normalization between bins is tested)"
                    if args.norm == "global"
                    else "  (per panel -- M shape only)")
                 + "\ngrey = closed by $M \\leq W - m_p$  |  "
                   "hatched = weight is zero, cross section undeliverable  |  "
                   "error bars are generator statistics only",
                 ha="center", fontsize=7.5, color="0.35")
        fig.subplots_adjust(left=0.085, right=0.975, top=0.918, bottom=0.075)
        pdf.savefig(fig)
        plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "Generated vs cross section, with ratios"
    pdf.close()
    print(f"[write] {args.out}  ({nq} pages, {nw} panels each)")

    # overall closure number
    ok = (D * scale > 0) & (G > 0) & live
    if ok.sum():
        pull = (G[ok] - D[ok] * scale) / np.sqrt(G[ok])
        print(f"[closure] over {int(ok.sum())} populated cells: "
              f"rms pull = {np.sqrt((pull**2).mean()):.2f}, "
              f"median |gen/xs - 1| = "
              f"{np.median(np.abs(G[ok]/(D[ok]*scale) - 1)):.3f}")


if __name__ == "__main__":
    main()
