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

Pair-mass weights
-----------------
The same plot checks a `pair_weight:` run against its measured target:
`--pair 2212,-2212` histograms M of EVERY (p, pbar) pair in the ntuple's
final-state block -- two entries per event, exactly as the pooled data
histogram was filled -- instead of the truth M_X branch. Point --weight at
the surface's `<name>_fitted` mask so cells the builder passed through
unfitted are hatched, and the target's own errors (the npz `errors` key)
are folded into the pulls.

    python plot_xsec_closure.py --gen ../events_pair.lund \\
        --xsec ../subtracted_Mppbar_pooled.npz --pair 2212,-2212 \\
        --weight ../pair_weight.root --weight-name w_pair_2212_-2212_fitted \\
        --out ../pair_closure_ppbar.pdf

With --pair, --gen may be the LUND file itself (no truth ntuple needed).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xsec                                                      # noqa: E402
from truth_ntuple import load_truth, pair_masses, parse_pids, pair_label  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402

M_P = 0.9382720813


def load_xsec(path, units_arg, measured=False):
    """Load the cross section and convert it to a per-bin yield.

    Must match what build_xsec_weight3d.py did, or the closure compares
    against a differently-normalized target and reads as a failure on
    every non-uniform bin.

    `measured`: the target is a histogram of data (the --pair case), so it
    has statistical errors -- the npz's own if present, else Poisson. A
    cross-section model has none.
    """
    z = np.load(path, allow_pickle=True)
    edges = [np.asarray(e, dtype=float) for e in z["edges"]]
    if len(edges) != 3:
        raise SystemExit(f"--xsec {path}: expected 3-D, got {len(edges)}-D.")
    D = np.clip(np.asarray(z["counts"], float), 0.0, None)
    if "errors" in z and np.shape(z["errors"]) == D.shape:
        E = np.abs(np.asarray(z["errors"], float))
    elif measured:
        E = np.sqrt(D)
        print(f"[xsec] {path} has no 'errors' key; using sqrt(N) for the target")
    else:
        E = np.zeros_like(D)
    units = units_arg or (str(z["units"]) if "units" in z else None)
    xsec.warn_if_ambiguous(edges, units is not None, "xsec")
    return (xsec.to_bin_yield(D, edges, units or "integral", "xsec"),
            xsec.to_bin_yield(E, edges, units or "integral", "xsec"), edges)


def load_gen(path, tree, edges, pair=None):
    """Bin the truth ntuple on the target's own grid.

    Without `pair`, the M axis is the truth M_X branch (one entry per
    event). With `pair` = (pidA, pidB), it is the invariant mass of every
    such pair in the final state, several entries per event -- the pooled
    histogram a pair_weight surface was fitted to.
    """
    a = load_truth(path, tree, final_state=pair is not None)
    if pair is None:
        pts = np.column_stack([a["Q2"], a["W"], a["M"]])
        n_rows = len(pts)
    else:
        ev, m, _ = pair_masses(a, *pair)
        pts = np.column_stack([np.asarray(a["Q2"])[ev], np.asarray(a["W"])[ev], m])
        n_rows = len(a["Q2"])
    pts = pts[np.isfinite(pts).all(axis=1)]
    H, _ = np.histogramdd(pts, bins=edges)
    what = "rows" if pair is None else f"events, {len(pts)} {pair_label(*pair)} entries"
    print(f"[gen] {path}:{tree}: {n_rows} {what}, {int(H.sum())} inside the "
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


def draw_closure(D, E, G, edges, live, out, *, norm="global", ncol=3,
                 ratio_range=(0.8, 1.2), tgt="cross section",
                 m_label=r"$M_{p\bar{p}}$", pair=None, title=None,
                 ynorm=False, scale=None, ydiv=None, overlay=None,
                 panel_size=(5.2, 4.6), ratio_gap=0.22):
    """Write the page-per-Q2-bin closure PDF and return the overall numbers.

    D, E : target per-bin yield and its error on `edges`; G the generated
    histogram on the same grid; `live` the cells that count (the weight
    allows them). Returns (rms_pull, median_abs_dev) over populated cells.

    ynorm: draw normalized counts -- both curves divided by `ydiv` (default:
    the generated total over the live cells), so the y axis is the fraction
    of entries per bin. `scale` (generated entries per target unit) can be
    given instead of being fitted here, so several targets can share one.

    overlay: a second target drawn in the SAME panels, dict(D, E, G, live,
    label) -- same grid, same scale and ydiv -- with its own ratio points;
    for the pooled p pbar vs p p comparison, where the relative
    normalization is part of the closure.

    panel_size: (width, height) in inches of ONE W panel (spectrum + ratio);
    the page grows with the grid instead of squeezing a fixed page.
    ratio_gap: vertical gap between a spectrum and its ratio panel, as a
    fraction of the mean axes height (matplotlib's hspace).
    """
    q2e, we, me = edges
    nq, nw, nm = D.shape
    mc = 0.5 * (me[:-1] + me[1:])
    r_lo, r_hi = ratio_range

    if D.sum() <= 0:
        raise SystemExit("cross section is empty")

    if scale is None:
        scale = G[live].sum() / D[live].sum()
    # y-axis factor: 1 for generated counts, 1/N for normalized counts
    yf = (1.0 / (ydiv if ydiv is not None else G[live].sum())) if ynorm else 1.0
    ov = overlay
    if ov is not None and tuple(ov["D"].shape) != tuple(D.shape):
        raise SystemExit("overlay target is not on the same grid")
    print(f"[norm] {norm}; global scale = {scale:.4g} "
          f"({int(G[live].sum())} generated / {D[live].sum():.4g} cross "
          "section, over live cells)")

    dead = (G <= 0) & (D > 0) & live
    if dead.any():
        print(f"[warn] {int(dead.sum())} cells are allowed by the weight but "
              "got no events -- undersampled, run more.")

    pdf = PdfPages(out)
    nrow = int(np.ceil(nw / ncol))
    for iq in range(nq):
        pw, ph = panel_size
        fig = plt.figure(figsize=(pw * ncol, ph * nrow + 1.0))
        # Nested grid: the OUTER cells are generously spaced so a ratio
        # panel's tick labels clear the title of the row beneath it, while
        # each inner pair (spectrum + its ratio) stays visually joined.
        # Margins in inches (tight_layout cannot handle the nested grids):
        # room for the y label on the left, the suptitle on top, the x
        # label at the bottom.
        fw, fh = fig.get_figwidth(), fig.get_figheight()
        outer = GridSpec(nrow, ncol, figure=fig,
                         left=0.75 / fw, right=1 - 0.2 / fw,
                         top=1 - 1.05 / fh, bottom=0.55 / fh,
                         hspace=0.42, wspace=0.28)

        for iw in range(nw):
            r, c = divmod(iw, ncol)
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c],
                height_ratios=[3, 1.15], hspace=ratio_gap)
            ax = fig.add_subplot(inner[0])
            rx = fig.add_subplot(inner[1], sharex=ax)

            gen = G[iq, iw] * yf
            lv = live[iq, iw]
            pred = D[iq, iw] * scale * yf
            if norm == "panel" and D[iq, iw][lv].sum() > 0 and gen.sum() > 0:
                pred = D[iq, iw] * (gen.sum() / D[iq, iw][lv].sum())

            # generator statistics, plus the target's own error where it
            # has one (scaled like the target)
            err = np.sqrt(np.maximum(G[iq, iw], 0.0) + (E[iq, iw] * scale) ** 2) * yf
            # rms pull for this panel -- computed here so it can go in the
            # title, where it cannot sit on top of the ratio points.
            okp = (pred > 0) & (gen > 0) & lv
            rms = (np.sqrt((((gen[okp] - pred[okp]) / err[okp]) ** 2).mean())
                   if okp.sum() else np.nan)

            # cross section: the curve being matched. Where the weight is
            # zero it is drawn hatched -- present in the model, impossible
            # for the generator to produce.
            ax.errorbar(mc, pred, fmt=".", ms=5, lw=1.0,
                                    color="#c0392b", zorder=4, label=tgt)
            # ax.errorbar(mc, pred, yerr=0, where="mid", color="#c0392b", lw=1.4,
            #         zorder=3, label=tgt)
            # ax.fill_between(mc, pred, step="mid", color="#c0392b",
            #                 alpha=0.12, zorder=1, where=lv)
            if (~lv).any():
                ax.fill_between(mc, pred, step="mid", where=~lv,
                                facecolor="none", edgecolor="#c0392b",
                                hatch="////", lw=0.0, alpha=0.55, zorder=2)
            # generated: points with statistical errors
            ax.errorbar(mc, gen, yerr=err, fmt="o", ms=3.2, lw=1.0,
                        color="#1f3f8b", zorder=4, label="Generated", alpha=0.7)
            if ov is not None:
                lv2 = ov["live"][iq, iw]
                pred2 = ov["D"][iq, iw] * scale * yf
                gen2 = ov["G"][iq, iw] * yf
                err2 = np.sqrt(np.maximum(ov["G"][iq, iw], 0.0)
                               + (ov["E"][iq, iw] * scale) ** 2) * yf
                ax.errorbar(mc, np.where(lv2, pred2, np.nan), fmt="x", ms=5, lw=1.0,
                            color="#d35400", zorder=4, label=ov["label"])
                ax.errorbar(mc, gen2, yerr=err2, fmt="s", ms=3.4, lw=0.9, mfc="none",
                            color="#1e8449", zorder=4, alpha=0.8,
                            label="Generated, " + ov["label"])

            ttl = rf"$W\in[{we[iw]:.2f} - {we[iw+1]:.2f}]$ GeV"
            # if np.isfinite(rms):
            #     ttl += f"    pull {rms:.1f}"
            ax.set_title(ttl, fontsize=11, pad=4,
                         color="#b03030" if np.isfinite(rms) and rms >= 2
                         else "0.1")
            ax.tick_params(labelsize=9, labelbottom=False)
            ax.set_xlim(me[0], me[-1])
            top = max(pred.max(), (gen + err).max()) if gen.size else 1.0
            if ov is not None and np.isfinite(pred2).any():
                top = max(top, np.nanmax(pred2), np.nanmax(gen2 + err2))
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
            rx.errorbar(mc, ratio, yerr=rerr, fmt="o", ms=3.0, lw=0.9,
                        color="#1f3f8b", zorder=3)
            if ov is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    good2 = (pred2 > 0) & lv2
                    ratio2 = np.where(good2, gen2 / pred2, np.nan)
                    rerr2 = np.where(good2, err2 / pred2, np.nan)
                rx.errorbar(mc, ratio2, yerr=rerr2, fmt="s", ms=2.8, lw=0.8, mfc="none",
                            color="#1e8449", zorder=3)
            rx.tick_params(labelsize=9, labeltop=False, labelbottom=True)
            rx.minorticks_on()
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
            rx.tick_params(labelsize=9)
            # Ticks that suit whatever window was asked for: the midpoint
            # plus one step either side, rounded to something readable.
            # step = (r_hi - r_lo) / 4.0
            # rx.set_yticks([round(1.0 - step, 4), 1.0, round(1.0 + step, 4)])
            rx.set_ylabel("Ratio", fontsize=9.5, labelpad=2)
            rx.set_xlabel(m_label + " [GeV]", fontsize=10.5, labelpad=2)
            if c == 0:
                ax.set_ylabel("Normalized counts" if ynorm else "LUND entries / bin",
                              fontsize=10.5)
            if iw == 0:
                ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

        fig.suptitle(
            (title or f"Generated vs {tgt}") + "\n"
            + rf"$Q^2 \in [{q2e[iq]:.2f} - {q2e[iq+1]:.2f}]$ GeV$^2$",
            fontsize=15, y=1 - 0.15 / fh, va="top")
        # fig.text(0.5, 0.016,
        #          f"normalization: {norm}"
        #          + ("  (one scale for the whole file -- relative "
        #             "normalization between bins is tested)"
        #             if norm == "global"
        #             else "  (per panel -- M shape only)")
        #          + "\ngrey = closed by $M \\leq W - m_p$  |  "
        #            + ("hatched = cell not fitted, passed through uncorrected  |  "
        #               "error bars: generator + target statistics"
        #               if pair else
        #               "hatched = weight is zero, cross section undeliverable  |  "
        #               "error bars are generator statistics only"),
        #          ha="center", fontsize=7.5, color="0.35")
        # fig.subplots_adjust(left=0.085, right=0.975, top=0.918, bottom=0.075)
        pdf.savefig(fig)
        plt.close(fig)

    d = pdf.infodict()
    d["Title"] = f"Generated vs {tgt}, with ratios"
    pdf.close()
    print(f"[write] {out}  ({nq} pages, {nw} panels each)")

    # overall closure number
    ok = (D * scale > 0) & (G > 0) & live
    if ok.sum():
        pull = (G[ok] - D[ok] * scale) / np.sqrt(G[ok] + (E[ok] * scale) ** 2)
        rms = float(np.sqrt((pull**2).mean()))
        med = float(np.median(np.abs(G[ok]/(D[ok]*scale) - 1)))
        print(f"[closure] over {int(ok.sum())} populated cells: "
              f"rms pull = {rms:.2f}, median |gen/xs - 1| = {med:.3f}")
        return rms, med
    return float("nan"), float("nan")



def main():
    ap = argparse.ArgumentParser(
        description="Generated vs cross section, with ratio panels.")
    ap.add_argument("--gen", required=True,
                    help="the WEIGHTED generator run: its truth ntuple (.root), "
                         "or -- with --pair -- its LUND file")
    ap.add_argument("--tree", default="truth")
    ap.add_argument("--xsec", "--target", required=True, dest="xsec",
                    help="the 3-D npz the weight was built from: a cross "
                         "section, or (with --pair) a measured pair-mass target")
    ap.add_argument("--pair", default=None,
                    help="'pidA,pidB': compare the pooled M of every such pair "
                         "in the final state (a pair_weight closure) instead of "
                         "the truth M_X branch")
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
    ap.add_argument("--panel-size", default="5.2,4.6",
                    help='"w,h" in inches of one W panel; the page is sized '
                         "from it and the grid (default 5.2,4.6)")
    ap.add_argument("--ratio-gap", type=float, default=0.22,
                    help="gap between each spectrum and its ratio panel "
                         "(matplotlib hspace, default 0.22)")
    ap.add_argument("--y", choices=("counts", "norm"), default="counts",
                    help="y axis: generated events per bin with the target "
                         "scaled to them (default), or normalized counts -- "
                         "both divided by their integral over the live cells")
    ap.add_argument("--ratio-range", default="0.8,1.2",
                    help='"lo,hi" for the ratio axes (default 0.8,1.2). '
                         "Points outside are drawn as triangles on the "
                         "boundary rather than silently dropped.")
    args = ap.parse_args()

    pair = parse_pids(args.pair) if args.pair else None
    if pair is None and args.gen.endswith(".lund"):
        raise SystemExit("--gen as a LUND file only works with --pair: the "
                         "truth M_X is not in the LUND (the two protons are "
                         "interchangeable). Use the truth ntuple here.")
    D, E, edges = load_xsec(args.xsec, args.xsec_units, measured=pair is not None)
    G = load_gen(args.gen, args.tree, edges, pair)
    tgt = "target" if pair else "cross section"
    m_label = (r"$M_{p\bar{p}}$" if pair is None
               else "$M(" + pair_label(*pair).replace("pbar", r"\bar{p}") + ")$")
    q2e, we, me = edges
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

    r_lo, r_hi = (float(v) for v in args.ratio_range.split(","))
    draw_closure(D, E, G, edges, live, args.out, norm=args.norm, ncol=args.ncol,
                 ratio_range=(r_lo, r_hi), tgt=tgt, m_label=m_label, pair=pair,
                 ynorm=(args.y == "norm"),
                 panel_size=tuple(float(v) for v in args.panel_size.split(",")),
                 ratio_gap=args.ratio_gap)


if __name__ == "__main__":
    main()
