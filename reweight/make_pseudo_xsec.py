"""
Generate a PSEUDO 3-D cross section sigma(Q2, W, M) and draw it as a PDF.

The values are ARBITRARY -- this is a placeholder with the right shape and
the right binning, meant for laying out the 3-D weighting before the real
cross section exists. Nothing here is a physics prediction.

Layout of the PDF, one figure per Q2 bin:

    page i  ->  Q2 bin i
      each panel on the page  ->  one W bin
      x axis of every panel   ->  M (the ppbar invariant mass)

so flipping through the pages walks you along Q2, reading down a page
walks you along W, and every panel is a spectrum in M.

The pseudo cross section
------------------------
A product of three smooth factors times the two phase-space factors that
actually constrain the reaction in input.txt
(`reaction: 2212, 9999: 9999, 2212, -2212`, i.e. W -> p X, X -> p pbar):

    sigma  =  f(Q2) * f(W) * [ (1-r) * continuum(M) + r * BW(M) ]
                     * beta(M) * q(W, M)

    f(Q2)      1/Q2 times a dipole -- a plausible electroproduction falloff
    f(W)       exponential falloff above W_min
    BW(M)      Breit-Wigner at --m0 / --gamma, defaulting to the 2.0 GeV,
               0.4 GeV wide state the input card already declares for
               placeholder PDG 9999 (`mass_9999: BW 2. 0.4`)
    r          resonance fraction, falling with Q2, so the peak weakens
               across pages instead of every page being a rescale of the
               first
    beta(M)    ppbar breakup factor sqrt(1 - (2 m_p / M)^2), which kills
               the region below the 2 m_p threshold
    q(W, M)    breakup momentum of the p + X system in the W rest frame,
               which vanishes at M = W - m_p

The last one is the important one to look at: it is a HARD kinematic edge
that moves with W, so the accessible M range grows panel by panel down a
page. Any real 3-D cross section has to respect it too, and the bins beyond
it can never be populated no matter what weight you assign them.

Usage
-----
    cd reweight
    python make_pseudo_xsec.py                       # PDF + npz, defaults
    python make_pseudo_xsec.py --q2-edges "1,2,3,4.5,7" \\
        --w-range "2.85,4.65" --nw 9 \\
        --m-range "1.85,3.05" --nm 24 \\
        --out ../pseudo_xsec.pdf

The .npz it writes alongside the PDF holds `counts` (the [nq, nw, nm]
values) and `edges`, the same layout the weight builders read, so the same
grid can be handed straight to the weighting step.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402

M_P = 0.9382720813
M_PPBAR_THR = 2.0 * M_P          # 1.8765 GeV, the ppbar threshold


def parse_range(spec):
    lo, hi = spec.split(",")
    return float(lo), float(hi)


def breakup_momentum(W, m1, m2):
    """Momentum of either body in the two-body rest frame; 0 below threshold.

    q = sqrt( (W^2-(m1+m2)^2) * (W^2-(m1-m2)^2) ) / 2W

    This is what closes off M > W - m_p: as M approaches that edge the
    first bracket goes to zero, so the cross section does too. It is a
    kinematic boundary, not a modelling choice.
    """
    with np.errstate(invalid="ignore"):
        a = W ** 2 - (m1 + m2) ** 2
        b = W ** 2 - (m1 - m2) ** 2
        val = np.sqrt(np.clip(a, 0.0, None) * np.clip(b, 0.0, None)) / (2.0 * W)
    return np.where(np.isfinite(val), val, 0.0)


def pseudo_xsec(Q2, W, M, args):
    """The arbitrary sigma(Q2, W, M). Shapes broadcast together."""
    # --- Q2: 1/Q2 times a dipole form factor -------------------------
    f_q2 = 1.0 / np.clip(Q2, 1e-6, None) / (1.0 + Q2 / args.q2_dipole) ** 2

    # --- W: falling above the generator's W_min ----------------------
    f_w = np.exp(-(W - args.w_ref) / args.w_slope)

    # --- M: a Breit-Wigner sitting on a smooth continuum -------------
    bw = (args.gamma ** 2 / 4.0) / ((M - args.m0) ** 2 + args.gamma ** 2 / 4.0)
    cont = np.exp(-(M - M_PPBAR_THR) / args.m_slope)
    # Resonance fraction falls with Q2: the peak washes out page by page.
    r = args.res_frac / (1.0 + Q2 / args.res_q2)
    f_m = (1.0 - r) * cont + r * bw

    # --- phase space: the two hard edges -----------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.sqrt(np.clip(1.0 - (M_PPBAR_THR / M) ** 2, 0.0, None))
    q = breakup_momentum(W, M_P, M)

    sigma = f_q2 * f_w * f_m * beta * q
    return np.where(np.isfinite(sigma), np.clip(sigma, 0.0, None), 0.0)


def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def build_grid(q2_edges, w_edges, m_edges, args):
    """Average the pseudo cross section over each 3-D bin.

    Sub-sampling inside the bin matters here: the M = W - m_p edge cuts
    through bins diagonally, and a single center evaluation would turn a
    partially open bin fully on or fully off.
    """
    n = max(1, args.supersample)
    frac = (np.arange(n) + 0.5) / n

    def sub(edges):
        lo, hi = edges[:-1, None], edges[1:, None]
        return (lo + (hi - lo) * frac[None, :]).ravel()          # [nbins*n]

    qs, ws, ms = sub(q2_edges), sub(w_edges), sub(m_edges)
    Q, Wg, Mg = np.meshgrid(qs, ws, ms, indexing="ij")
    fine = pseudo_xsec(Q, Wg, Mg, args)

    nq, nw, nm = len(q2_edges) - 1, len(w_edges) - 1, len(m_edges) - 1
    return fine.reshape(nq, n, nw, n, nm, n).mean(axis=(1, 3, 5))


def draw(sigma, q2_edges, w_edges, m_edges, args):
    """One page per Q2 bin; one panel per W bin; M along x."""
    nq, nw, nm = sigma.shape
    mc = bin_centers(m_edges)
    ncol = args.ncol
    nrow = int(np.ceil(nw / ncol))

    pdf = PdfPages(args.out)
    for iq in range(nq):
        fig, axes = plt.subplots(nrow, ncol, figsize=(8.5, 11.0),
                                 sharex=True, sharey=args.share_y)
        axes = np.atleast_1d(axes).ravel()

        page = sigma[iq]
        # A shared y-scale per page keeps the W bins comparable to each
        # other. Across pages the normalization changes by orders of
        # magnitude, so a single global scale would flatten the last pages
        # into the axis.
        ymax = page.max()
        if ymax <= 0:
            ymax = 1.0

        for iw in range(nw):
            ax = axes[iw]
            vals = page[iw]
            ax.step(mc, vals, where="mid", color="#1f5f8b", lw=1.3)
            ax.fill_between(mc, vals, step="mid",
                            color="#1f5f8b", alpha=0.18)

            # the moving kinematic edge M = W - m_p
            m_edge_lo = w_edges[iw] - M_P
            m_edge_hi = w_edges[iw + 1] - M_P
            if m_edge_hi < m_edges[-1]:
                ax.axvspan(max(m_edge_hi, m_edges[0]), m_edges[-1],
                           color="0.85", alpha=0.7, lw=0, zorder=0)
            if m_edges[0] < m_edge_lo < m_edges[-1]:
                ax.axvline(m_edge_lo, color="0.45", ls="--", lw=0.9)

            ax.axvline(M_PPBAR_THR, color="#b03030", ls=":", lw=0.9)

            ax.set_title(f"W = {w_edges[iw]:.2f} - {w_edges[iw+1]:.2f} GeV",
                         fontsize=8.5, pad=3)
            ax.tick_params(labelsize=7)
            if args.log:
                ax.set_yscale("log")
                pos = page[page > 0]
                ax.set_ylim(pos.min() * 0.5 if pos.size else 1e-6, ymax * 2)
            elif args.share_y:
                ax.set_ylim(0, ymax * 1.12)
            ax.set_xlim(m_edges[0], m_edges[-1])
            if iw // ncol == nrow - 1:
                ax.set_xlabel(r"$M_{p\bar{p}}$ [GeV]", fontsize=8)
            if iw % ncol == 0:
                ax.set_ylabel(r"$\sigma$ [arb.]", fontsize=8)

        for ax in axes[nw:]:
            ax.axis("off")

        # The normalization drops by more than an order of magnitude from
        # the first Q2 page to the last, so state each page's peak: the
        # panels are drawn on a per-page scale and would otherwise look
        # identical from page to page.
        rel = ymax / sigma.max() if sigma.max() > 0 else 1.0
        fig.suptitle(
            r"Pseudo cross section  $\sigma(Q^2,\,W,\,M_{p\bar{p}})$"
            "\n"
            rf"$Q^2$ = {q2_edges[iq]:.2f} $-$ {q2_edges[iq+1]:.2f} GeV$^2$"
            f"   (page {iq+1} of {nq})"
            f"   |   page peak = {ymax:.4g} arb."
            f" ({rel:.3f} of the overall peak)",
            fontsize=11.5, y=0.975)
        fig.text(0.5, 0.022,
                 "arbitrary values  |  grey = closed by $M \\leq W - m_p$  |  "
                 "dashed = edge at the bin's lower $W$  |  "
                 "dotted red = $2m_p$ threshold\n"
                 "y-scale shared within a page, not across pages",
                 ha="center", fontsize=7.5, color="0.35")
        fig.tight_layout(rect=(0, 0.045, 1, 0.945))
        pdf.savefig(fig)
        plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "Pseudo cross section sigma(Q2, W, M_ppbar)"
    d["Subject"] = "Arbitrary placeholder values on the 3-D weighting grid"
    pdf.close()


def main():
    ap = argparse.ArgumentParser(
        description="Pseudo 3-D cross section sigma(Q2, W, M) as a PDF.")
    ap.add_argument("--q2-edges", default="1,2,3,4.5,7",
                    help="comma-separated Q2 bin edges [GeV^2]; one PDF page "
                         "per bin (default 1,2,3,4.5,7)")
    ap.add_argument("--w-range", default="2.85,4.65",
                    help='"lo,hi" W range [GeV] (default 2.85,4.65 -- the '
                         "input card's W_min upward)")
    ap.add_argument("--nw", type=int, default=9,
                    help="W bins, one panel each (default 9)")
    ap.add_argument("--m-range", default="1.85,3.05",
                    help='"lo,hi" M_ppbar range [GeV] (default 1.85,3.05, '
                         "just below the 2m_p threshold upward)")
    ap.add_argument("--nm", type=int, default=24,
                    help="M bins along the x axis (default 24)")

    sh = ap.add_argument_group("pseudo cross-section shape (all arbitrary)")
    sh.add_argument("--m0", type=float, default=2.0,
                    help="Breit-Wigner mass [GeV] (default 2.0, matching "
                         "`mass_9999: BW 2. 0.4` in input.txt)")
    sh.add_argument("--gamma", type=float, default=0.4,
                    help="Breit-Wigner width [GeV] (default 0.4)")
    sh.add_argument("--res-frac", type=float, default=0.75,
                    help="resonance fraction at Q2 -> 0 (default 0.75)")
    sh.add_argument("--res-q2", type=float, default=3.0,
                    help="Q2 [GeV^2] at which the resonance fraction halves "
                         "(default 3.0)")
    sh.add_argument("--q2-dipole", type=float, default=2.0,
                    help="dipole scale in 1/(1+Q2/L)^2 (default 2.0)")
    sh.add_argument("--w-ref", type=float, default=2.85,
                    help="W [GeV] where the W falloff starts (default 2.85)")
    sh.add_argument("--w-slope", type=float, default=0.9,
                    help="W falloff constant [GeV] (default 0.9)")
    sh.add_argument("--m-slope", type=float, default=0.8,
                    help="continuum falloff constant in M [GeV] (default 0.8)")

    io = ap.add_argument_group("output")
    io.add_argument("--out", default="../pseudo_xsec.pdf",
                    help="output PDF (default ../pseudo_xsec.pdf)")
    io.add_argument("--save-table", default=None,
                    help="npz for the binned values (default: the PDF path "
                         "with a .npz suffix; pass 'none' to skip)")
    io.add_argument("--ncol", type=int, default=3,
                    help="panels per row on a page (default 3)")
    io.add_argument("--log", action="store_true",
                    help="log y axis")
    io.add_argument("--no-share-y", dest="share_y", action="store_false",
                    help="give every panel its own y scale instead of one "
                         "shared across the page")
    io.add_argument("--supersample", type=int, default=4,
                    help="sub-samples per bin per axis when averaging the "
                         "cross section into bins (default 4)")
    args = ap.parse_args()

    q2_edges = np.array([float(v) for v in args.q2_edges.split(",")])
    if len(q2_edges) < 2 or np.any(np.diff(q2_edges) <= 0):
        raise SystemExit("--q2-edges needs >= 2 increasing values.")
    w_lo, w_hi = parse_range(args.w_range)
    m_lo, m_hi = parse_range(args.m_range)
    w_edges = np.linspace(w_lo, w_hi, args.nw + 1)
    m_edges = np.linspace(m_lo, m_hi, args.nm + 1)

    sigma = build_grid(q2_edges, w_edges, m_edges, args)
    nq, nw, nm = sigma.shape
    print("[grid] Q2 " + f"{nq} bins: "
          + ", ".join(f"{a:g}-{b:g}" for a, b in zip(q2_edges[:-1], q2_edges[1:])))
    print(f"[grid] W  {nw} bins [{w_lo:g},{w_hi:g}]  "
          f"M {nm} bins [{m_lo:g},{m_hi:g}]  -> {sigma.size} cells")

    open_cells = int(np.sum(sigma > 0))
    print(f"[grid] {open_cells}/{sigma.size} cells are kinematically open "
          f"({100.0*open_cells/sigma.size:.1f}%); the rest sit beyond "
          "M = W - m_p or below the 2m_p threshold.")

    draw(sigma, q2_edges, w_edges, m_edges, args)
    print(f"[write] {args.out}  ({nq} pages, {nw} panels each)")

    table = args.save_table
    if table is None:
        table = os.path.splitext(args.out)[0] + ".npz"
    if table.lower() != "none":
        # 'density': build_grid averages sigma over each bin, so these are
        # dsigma/dQ2 dW dM values, not per-bin integrals. The weight builder
        # multiplies by the bin volume -- which matters, since the default
        # Q2 edges are not uniform.
        np.savez(table, counts=sigma,
                 edges=np.array([q2_edges, w_edges, m_edges], dtype=object),
                 varnames=np.array(["Q2", "W", "M"]),
                 units="density")
        print(f"[write] {table}  (counts[{nq},{nw},{nm}] + edges + varnames)")


if __name__ == "__main__":
    main()
