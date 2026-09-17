"""
Build the generator's 3-D weight surface w(Q2, W, M) from a cross section.

Companion to build_xsec_weight.py, for a cross section that is 3-D in
(Q2, W, M) instead of 2-D in (Q2, E'). It writes a TH3D that the generator
reads through one line in input.txt:

    xsec_weight: xsec_weight.root w_Q2_W_M

Where the weight is applied, and why it has to be there
-------------------------------------------------------
The 2-D w(Q2, E') surface acts at electron-sampling time, inside
generateOneElectron. A 3-D weight cannot: M is the invariant mass of the
intermediate X from the first vertex (M_ppbar for
`reaction: 2212, 9999: 9999, 2212, -2212`), and it does not exist until
the intermediate mass has been sampled and the chain decayed. So this
weight is a THIRD accept-reject stage in the main loop, next to the
existing mom_weight one, using the truth 4-vector of X.

The proposal density is NOT flat
--------------------------------
build_xsec_weight.py can skip the denominator because the generator draws
Q2 and E' uniformly. Nothing like that holds here:

    W  is a derived quantity -- fixed by Q2 and E' -- so its density is
       whatever the uniform (Q2, E') square induces through
       W^2 = M_p^2 + 2 M_p (E - E') - Q2, further shaped by the W_min and
       theta cuts.
    M  is drawn from the Breit-Wigner declared by `mass_9999: BW 2. 0.4`,
       resampled until above the decay threshold and then filtered by
       whether the two-body decay actually closes.

So g(Q2, W, M) has to be MEASURED, and --gen is required. It comes from
the generator's truth ntuple (`truth_ntuple:` in the input card), not from
the LUND file: the LUND holds two interchangeable protons and picking the
one that came from X is ambiguous, whereas the ntuple records M from the
truth 4-vector.

    w(Q2, W, M) = (1/C) * d(Q2, W, M) / g(Q2, W, M),   C = max(d/g)

Workflow
--------
    1. run the generator with NO xsec_weight, with
           truth_ntuple: gen_truth_unweighted.root
    2. python build_xsec_weight3d.py \\
           --xsec ../pseudo_xsec.npz \\
           --gen  gen_truth_unweighted.root \\
           --out  ../xsec_weight.root
    3. add `xsec_weight: xsec_weight.root w_Q2_W_M` to input.txt, rerun
       (keep truth_ntuple: on, pointing somewhere new)
    4. python plot_xsec_closure.py --gen gen_truth_weighted.root \\
           --xsec ../pseudo_xsec.npz --out ../xsec_closure.pdf

Step 4 is the check that matters: it overlays the generated distribution
on the cross section with a ratio panel, per (Q2, W) bin.

Exactness
---------
With per-bin lookup (`xsec_weight_mode: bin`, the generator default), the
accepted density is proportional to d bin by bin BY CONSTRUCTION -- the
weighted events match the cross section exactly, up to Poisson noise, once
you have enough of them. Measured here on the 4x9x24 grid: rms pull 1.02
and median |gen/xs - 1| = 2.2% over all 354 delivered cells, at 600k
weighted events (2.4% expected from statistics alone).

Only two things break that, and both are under your control:

  * A CLIP on the ratio (--wclip-pct / --wmax) caps cells you still
    deliver, so they come out low -- the five cells clipped at the 99th
    percentile in an earlier run were 15-28% under-produced, at 3-5 sigma.
    Clipping is therefore OFF by default. It buys speed by lying; prefer
    --min-gen, which drops untrustworthy cells honestly (they show up
    hatched in the closure plot) and is usually FASTER anyway, because the
    cells it removes are the ones setting the max=1 scale.

  * DENOMINATOR NOISE. w = d/g, so a cell whose g was measured from N
    events carries a 1/sqrt(N) error into the weight, and the closure
    inherits it. --min-gen is what bounds this: it is an accuracy floor on
    the cells you deliver, not a cosmetic cut. Set it from the accuracy
    you want -- --target-accuracy does that arithmetic for you -- and give
    --gen plenty of events.

Run --scan to see the whole trade-off (delivered cross section vs
efficiency vs worst-cell weight error) on your own grid before choosing.
"""

import argparse
import os
import sys

import numpy as np
import ROOT
from array import array as _darr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xsec                                                     # noqa: E402
from build_weight_func import apply_ratio_guards, _archive_copy  # noqa: E402


def load_xsec_npz(path):
    """Read a 3-D cross section npz (counts + edges + varnames).

    This is what make_pseudo_xsec.py writes. Its edges ARE the weight grid:
    the proposal is binned on the same edges so the ratio is bin-by-bin.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--xsec: file not found: {path}")
    z = np.load(path, allow_pickle=True)
    for k in ("counts", "edges"):
        if k not in z:
            raise SystemExit(f"--xsec {path}: missing '{k}' (has {list(z.keys())})")
    edges = [np.asarray(e, dtype=float) for e in z["edges"]]
    if len(edges) != 3:
        raise SystemExit(f"--xsec {path}: expected a 3-D histogram, got "
                         f"{len(edges)}-D.")
    D = np.asarray(z["counts"], dtype=float)
    if D.shape != tuple(len(e) - 1 for e in edges):
        raise SystemExit(f"--xsec {path}: counts shape {D.shape} does not "
                         f"match edges {[len(e)-1 for e in edges]}.")
    names = [str(v) for v in z["varnames"]] if "varnames" in z else ["?"] * 3
    units = str(z["units"]) if "units" in z else None
    print(f"[xsec] {path}: vars={names} grid="
          f"{'x'.join(str(len(e)-1) for e in edges)}"
          + (f" units={units}" if units else ""))
    return np.clip(D, 0.0, None), edges, names, units


def eval_formula_3d(expr, edges, supersample):
    """Evaluate a numpy expression in Q2, W, M, averaged over each bin."""
    n = max(1, supersample)
    frac = (np.arange(n) + 0.5) / n

    def sub(e):
        lo, hi = e[:-1, None], e[1:, None]
        return (lo + (hi - lo) * frac[None, :]).ravel()

    Q2, W, M = np.meshgrid(*[sub(e) for e in edges], indexing="ij")
    ns = dict(xsec._SAFE_FUNCS)
    ns.update({"Q2": Q2, "W": W, "M": M, "M_P": xsec.M_P})
    try:
        val = eval(expr, {"__builtins__": {}}, ns)              # noqa: S307
    except Exception as err:
        raise SystemExit(f"--formula failed: {err}\n"
                         "Available: Q2, W, M, M_P and the numpy functions.")
    fine = np.broadcast_to(np.asarray(val, float), Q2.shape).astype(float)
    fine = np.where(np.isfinite(fine), np.clip(fine, 0.0, None), 0.0)
    shape = []
    for e in edges:
        shape += [len(e) - 1, n]
    return fine.reshape(shape).mean(axis=(1, 3, 5))


def load_gen_density(path, tree, edges):
    """Bin the generator's truth ntuple into g(Q2, W, M) on the weight grid.

    Returns (probability-normalized density, raw counts). The counts feed
    the low-statistics guards: a bin the generator visited twice gives a
    ratio with 70% error, and after the max=1 rescale that noise bin can
    set the scale for the entire surface.
    """
    import uproot
    if not os.path.exists(path):
        raise SystemExit(f"--gen: file not found: {path}")
    t = uproot.open(path)[tree]
    a = t.arrays(["Q2", "W", "M"], library="np")
    pts = np.column_stack([a["Q2"], a["W"], a["M"]])
    finite = np.isfinite(pts).all(axis=1)
    G_counts, _ = np.histogramdd(pts[finite], bins=edges)
    n_in = int(G_counts.sum())
    if n_in <= 0:
        raise SystemExit(f"--gen {path}: no events inside the weight grid. "
                         "Check that the grid covers the generated range.")
    print(f"[gen] {path}:{tree}: {len(pts)} rows, {n_in} inside the grid "
          f"({100.0*n_in/len(pts):.1f}%)")
    return G_counts / n_in, G_counts


def write_th3(W, name, title, edges, out):
    nx, ny, nz = W.shape
    xe, ye, ze = (_darr('d', [float(v) for v in e]) for e in edges)
    h = ROOT.TH3D(name, title, nx, xe, ny, ye, nz, ze)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                h.SetBinContent(i + 1, j + 1, k + 1, float(W[i, j, k]))
    tf = ROOT.TFile(out, "RECREATE")
    h.Write()
    tf.Close()


def read_prev_th3(path, name, edges):
    """Load a previous cumulative TH3D, checking the grid matches exactly."""
    if not os.path.exists(path):
        raise SystemExit(f"--prev: file not found: {path}")
    tf = ROOT.TFile(path)
    if tf.IsZombie():
        tf.Close()
        raise SystemExit(f"--prev: cannot open {path}")
    h = tf.Get(name)
    if not h:
        tf.Close()
        raise SystemExit(f"--prev: no TH3 named '{name}' in {path}")
    shape = (h.GetNbinsX(), h.GetNbinsY(), h.GetNbinsZ())
    want = tuple(len(e) - 1 for e in edges)
    if shape != want:
        tf.Close()
        raise SystemExit(f"--prev grid {shape} != current {want}; reweight "
                         "iterations must share binning.")
    axes = (h.GetXaxis(), h.GetYaxis(), h.GetZaxis())
    for ax, e, lbl in zip(axes, edges, "xyz"):
        got = np.array([ax.GetBinLowEdge(i + 1) for i in range(ax.GetNbins())]
                       + [ax.GetBinUpEdge(ax.GetNbins())])
        if not np.allclose(got, e):
            tf.Close()
            raise SystemExit(f"--prev {lbl} bin edges differ from the current "
                             "grid; iterations must share identical edges.")
    prev = np.array([[[h.GetBinContent(i + 1, j + 1, k + 1)
                       for k in range(shape[2])]
                      for j in range(shape[1])]
                     for i in range(shape[0])], dtype=float)
    tf.Close()
    return prev


def scan_min_gen(D, G, G_counts,
                 floors=(0, 25, 100, 200, 500, 1000, 2000, 5000, 10000)):
    """Print how --min-gen trades delivered cross section for speed.

    Three numbers move together as the floor rises: you deliver LESS of the
    cross section, the run gets FASTER (the dropped cells are the ones with
    a huge d/g ratio, which is what sets the max=1 scale), and the weight in
    the cells you keep gets MORE accurate. The efficiency column often has a
    cliff -- one badly-sampled cell holding the whole surface hostage.
    """
    raw = np.divide(D, G, out=np.zeros_like(D), where=G > 0.0)
    tot = D.sum()
    print(f"\n  {'--min-gen':>10} {'cells':>7} {'% of sigma':>11} "
          f"{'efficiency':>11} {'worst cell wt err':>18}")
    prev_eff = None
    for mg in floors:
        w = np.where(G_counts < mg, 0.0, raw)
        if w.max() <= 0:
            continue
        w = w / w.max()
        live = w > 0
        eff = float((G * w).sum())
        worst = 1.0 / np.sqrt(G_counts[live].min()) if live.any() else np.nan
        flag = ""
        if prev_eff is not None and prev_eff > 0 and eff / prev_eff > 3:
            flag = "  <- cliff: one cell was holding the scale"
        prev_eff = eff
        print(f"  {mg:>10} {int(live.sum()):>7} "
              f"{100*D[live].sum()/tot:>10.1f}% {eff:>11.4f} "
              f"{100*worst:>17.1f}%{flag}")
    print("\n  Pick the floor from the accuracy you need (--target-accuracy), "
          "then\n  check what fraction of the cross section you are giving up.")


def main():
    ap = argparse.ArgumentParser(
        description="Build a 3-D generator weight w(Q2, W, M) from a cross "
                    "section.")
    ap.add_argument("--xsec", default=None,
                    help="3-D cross-section npz (counts + edges + varnames, "
                         "e.g. from make_pseudo_xsec.py). Its edges define "
                         "the weight grid.")
    ap.add_argument("--formula", default=None,
                    help="numpy expression in Q2, W, M instead of --xsec; "
                         "needs --q2-edges/--w-range/--m-range.")
    ap.add_argument("--q2-edges", default=None,
                    help="comma-separated Q2 edges, with --formula")
    ap.add_argument("--w-range", default=None, help='"lo,hi", with --formula')
    ap.add_argument("--nw", type=int, default=9)
    ap.add_argument("--m-range", default=None, help='"lo,hi", with --formula')
    ap.add_argument("--nm", type=int, default=24)
    ap.add_argument("--supersample", type=int, default=4,
                    help="sub-samples per bin per axis for --formula "
                         "(default 4)")
    ap.add_argument("--xsec-units", choices=("integral", "density"),
                    default=None,
                    help="are the --xsec values a per-bin INTEGRAL (the "
                         "number of events that bin should get) or a DENSITY "
                         "(dsigma/dQ2 dW dM)? A density is multiplied by the "
                         "bin volume first. Only matters for non-uniform "
                         "binning -- and then by up to the volume ratio, "
                         "2.5x for Q2 edges 1,2,3,4.5,7. Default: whatever "
                         "the npz records, else integral (density for "
                         "--formula, which is sampled inside the bin).")

    ap.add_argument("--gen", required=True,
                    help="truth ntuple ROOT file from an UNWEIGHTED generator "
                         "run (`truth_ntuple:` in the input card). Required: "
                         "the proposal density in (Q2, W, M) is not flat and "
                         "has to be measured.")
    ap.add_argument("--tree", default="truth", help="tree name (default truth)")

    ap.add_argument("--out", required=True, help="output ROOT file")
    ap.add_argument("--name", default="w_Q2_W_M",
                    help="TH3D name; must match the second field of the "
                         "`xsec_weight:` line (default w_Q2_W_M)")
    ap.add_argument("--min-gen", type=float, default=None, dest="min_rec",
                    help="zero bins whose generated denominator holds fewer "
                         "than this many events. This is an ACCURACY FLOOR, "
                         "not a cosmetic cut: a delivered cell with N events "
                         "carries a 1/sqrt(N) weight error into the closure. "
                         "Default: derived from --target-accuracy. It also "
                         "usually speeds the run up, since the cells it "
                         "removes are the ones setting the max=1 scale.")
    ap.add_argument("--target-accuracy", type=float, default=0.05,
                    help="how well the weight must be known in every cell "
                         "you deliver (default 0.05 = 5%%). Sets --min-gen to "
                         "1/accuracy^2 when --min-gen is not given.")
    ap.add_argument("--wmax", type=float, default=None,
                    help="hard cap on the raw d/g ratio. BREAKS EXACTNESS: "
                         "capped cells are still delivered, and come out "
                         "under-produced by however much they were capped. "
                         "Prefer --min-gen. Takes precedence over --wclip-pct")
    ap.add_argument("--wclip-pct", type=float, default=None,
                    help="cap the raw d/g ratio at this percentile of the "
                         "nonzero bins. OFF by default, and breaks exactness "
                         "the same way --wmax does.")
    ap.add_argument("--scan", action="store_true",
                    help="print the --min-gen trade-off (delivered cross "
                         "section vs efficiency vs worst-cell weight error) "
                         "and exit without writing anything.")
    ap.add_argument("--prev", default=None,
                    help="previous cumulative TH3D to multiply into")
    ap.add_argument("--archive", default=None,
                    help="directory for a versioned copy w_xsec3d_iter<N>.root")
    args = ap.parse_args()
    args.mode = "xsec3d"

    if bool(args.xsec) == bool(args.formula):
        raise SystemExit("give exactly one of --xsec / --formula.")

    if args.xsec:
        D, edges, names, file_units = load_xsec_npz(args.xsec)
        units = args.xsec_units or file_units
        xsec.warn_if_ambiguous(edges, units is not None, "xsec")
        D = xsec.to_bin_yield(D, edges, units or "integral", "xsec")
    else:
        if not (args.q2_edges and args.w_range and args.m_range):
            raise SystemExit("--formula needs --q2-edges, --w-range and "
                             "--m-range.")
        q2e = np.array([float(v) for v in args.q2_edges.split(",")])
        wlo, whi = (float(v) for v in args.w_range.split(","))
        mlo, mhi = (float(v) for v in args.m_range.split(","))
        edges = [q2e, np.linspace(wlo, whi, args.nw + 1),
                 np.linspace(mlo, mhi, args.nm + 1)]
        D = eval_formula_3d(args.formula, edges, args.supersample)
        names = ["Q2", "W", "M"]
        print(f"[xsec] formula: {args.formula}")
        # A formula is sampled inside the bin, so it is a density by
        # construction: it has to be integrated over the bin to become a
        # yield. --xsec-units integral overrides if the expression already
        # returns a per-bin number.
        D = xsec.to_bin_yield(D, edges, args.xsec_units or "density", "xsec")

    if D.sum() <= 0:
        raise SystemExit("Cross section is zero everywhere on the grid.")
    D = D / D.sum()

    G, G_counts = load_gen_density(args.gen, args.tree, edges)

    # w = d / g. An empty proposal bin is unreachable: no accept-reject can
    # populate a region the generator never proposes, so w = 0 there.
    Wt = np.divide(D, G, out=np.zeros_like(D), where=G > 0.0)
    if Wt.max() <= 0:
        raise SystemExit("Weight surface is all zero -- the cross section and "
                         "the generated events have no bins in common. Check "
                         "the grid against the generated ranges.")

    holes = int(np.sum((G_counts <= 0) & (D > 0)))
    frac_d = float(D[G_counts <= 0].sum())
    if holes:
        print(f"[gen] {holes} bins carry cross section but no generated "
              f"events ({100*frac_d:.2f}% of the total cross section) -> w=0. "
              "These are unreachable, not merely undersampled, if the "
              "generator's own kinematics forbid them.")

    if args.wclip_pct is not None and args.wclip_pct >= 100:
        args.wclip_pct = None

    if args.scan:
        scan_min_gen(D, G, G_counts)
        return

    if args.min_rec is None:
        acc = max(args.target_accuracy, 1e-6)
        args.min_rec = float(np.ceil(1.0 / (acc * acc)))
        print(f"[guard] --target-accuracy {acc:g} -> --min-gen "
              f"{args.min_rec:g} (a delivered cell then has its weight known "
              f"to {100*acc:.1f}% or better)")

    if args.wmax is not None or args.wclip_pct is not None:
        print("[warn] a ratio cap is active. Capped cells are still "
              "delivered and will come out UNDER-PRODUCED by however much "
              "they were capped -- the weighted events will no longer match "
              "the cross section there, at any statistics. Use --min-gen "
              "instead unless you know you want this.")

    Wt = apply_ratio_guards(Wt, G_counts, args)

    if args.prev:
        prev = read_prev_th3(args.prev, args.name, edges)
        Wt = Wt * prev
        if Wt.max() <= 0:
            raise SystemExit("Cumulative weight is all zero after --prev.")
        print(f"[prev] multiplied in {args.prev} -> cumulative surface")

    Wt = Wt / Wt.max()

    # Expected keep fraction: the generator proposes with density g, so the
    # mean accept probability is sum(g * w), not the plain mean over bins.
    eff = float((G * Wt).sum())
    print(f"[eff] mean accept probability = {eff:.4f} "
          f"(~1 kept event per {1.0/eff:.1f} fully built events); "
          f"zero-fraction={float(np.mean(Wt <= 0)):.3f}")
    if eff < 0.02:
        print("[eff] that is expensive -- every rejected event here has "
              "already been through the full decay chain. Run --scan: this "
              "is usually one badly-sampled cell setting the max=1 scale, "
              "and raising --min-gen past it costs a little cross section "
              "and buys back an order of magnitude. Do NOT reach for a clip: "
              "it keeps that cell and under-produces it instead.")

    title = (f"w({names[0]},{names[1]},{names[2]});"
             "Q^{2} [GeV^{2}];W [GeV];M [GeV]")
    write_th3(Wt, args.name, title, edges, args.out)
    print(f"[write] {args.out}:{args.name}  grid="
          f"{'x'.join(str(n) for n in Wt.shape)}")

    if args.archive:
        print(f"[archive] stored -> "
              f"{_archive_copy(args.out, args.archive, args.mode)}")

    print(f"\nAdd this line to the generator's input card:\n"
          f"    xsec_weight: {args.out} {args.name}")


if __name__ == "__main__":
    main()
