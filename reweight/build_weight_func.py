"""
Build a 2-D rejection-sampling weight  w(x, y) = data / gen  and save it as
a TH2D ROOT file for the generator to consume.

Two variable pairs are supported via --mode:
  * q2ep (default): w(Q2, E'), applied at electron-sampling time.
  * pmom          : w(p_lead, p_sub) over the leading/sub-leading
                    proton (2212 only) momentum magnitudes, applied as a
                    SECOND accept-reject after the full event is built.
                    Data columns: p_p1 (leading), p_p2 (sub-leading).

Method (pure rejection sampling, NO kernel smoothing)
-----------------------------------------------------
The generator is a proposal with density g(x); we want its accepted output
to follow the data density d(x).  Rejection sampling keeps each event with
probability

        w(x) = (1/C) * d(x) / g(x),     C = max_x d(x)/g(x)

and the kept events then follow  g * w  ~  d.  The only thing the generator
needs is the number w(x); this script produces it.

Both densities are estimated by plain, identically-binned 2-D histograms
(no Gaussian KDE):

    - Bin the data       -> counts  Hd[i,j]
    - Bin the MC/gen LUND -> counts  Hg[i,j]   (same edges)
    - Per-bin ratio:
          w[i,j] = (Hd[i,j] / sum Hd) / (Hg[i,j] / sum Hg)
      i.e. probability-normalized data over probability-normalized MC.
    - Empty MC bins (Hg[i,j] == 0) are set to w = 0 (guard the 0/0; the
      generator cannot populate a region the proposal never reaches).
    - Divide by max(w) so max(w) = 1.0  -> a valid accept probability in
      [0, 1]  (this is the constant C above).

The grid is stored as a TH2D whose bins ARE the histogram bins, so bin
(i, j) holds w[i,j] at its center.  The generator calls
TH2::Interpolate(x, y) -> bilinear interpolation between bin centers, a
continuous w(x, y) built directly from the raw binned counts.

Why d/g and not just d?
-----------------------
Accept-reject sculpts the proposal g into the target d by keeping each
event with probability proportional to d/g.  If the generator samples
uniformly then g is flat and d/g is proportional to d.  Writing g out
explicitly keeps the weight correct even when the proposal is not uniform
(data-hist sampling, or an already-applied upstream weight).  Pass --mc to
supply g; omit it to fall back to an explicit uniform proposal (flat g).

CSV must have columns: Q2, Ep (q2ep mode) or p_p1, p_p2 (pmom mode).

Sequential reweighting (pmom mode)
----------------------------------
The proton momenta depend on the Q2/E' kinematics, so the pmom denominator
g(p_lead, p_sub) must be measured from a generator run that ALREADY has the
Q2/E' weight applied.  The one-pass workflow is therefore:

    1. run generator unweighted            -> events_unweighted.lund
    2. build q2ep weight from (1)          -> weight_func.root
    3. run generator with weight_func only -> events_q2ep.lund
    4. build pmom weight from (3)          -> mom_weight.root
    5. run generator with BOTH weights

Usage
-----
    # q2ep (default)
    python build_weight_func.py \\
        --data real_data.csv \\
        --mc   events_unweighted.lund \\
        --out  weight_func.root \\
        --name w_Q2_Ep \\
        --x-range "0.5,9" \\
        --y-range "1,9" \\
        --nx 60 --ny 60

    # pmom (proton momenta); --mc is the q2ep-weighted run from step 3
    python build_weight_func.py \\
        --mode pmom \\
        --data real_data.csv \\
        --mc   events_q2ep.lund \\
        --out  mom_weight.root \\
        --name w_pp \\
        --x-range "0,10" \\
        --y-range "0,6" \\
        --nx 60 --ny 60
"""

import argparse
import csv
import numpy as np
import ROOT

from kinematics import compute_kinematics_batch, proton_momenta_batch


# Per-mode configuration: data CSV columns, the generator-quantity keys,
# the TH2D name/axis titles.  Keeps main() free of mode-specific branching.
MODES = {
    "q2ep": {
        "data_cols": ("Q2", "Ep"),
        "default_name": "w_Q2_Ep",
        "title": "w(Q^{2},E');Q^{2} [GeV^{2}];E' [GeV]",
        "xlabel": "Q2", "ylabel": "E'",
    },
    "pmom": {
        "data_cols": ("p_p1", "p_p2"),
        "default_name": "w_pp",
        "title": "w(p_{lead},p_{sub});p_{lead} [GeV];p_{sub} [GeV]",
        "xlabel": "p_lead", "ylabel": "p_sub",
    },
}


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


_OPS = {
    ">=": np.greater_equal, "<=": np.less_equal,
    ">":  np.greater,       "<":  np.less,
    "==": np.equal,         "!=": np.not_equal,
}


def parse_cut(spec):
    """Parse 'COL OP VAL' (e.g. 'W>=2.85') into (col, op_fn, value)."""
    s = spec.replace(" ", "")
    for op in (">=", "<=", "==", "!=", ">", "<"):  # longest first
        if op in s:
            col, val = s.split(op, 1)
            return col, _OPS[op], float(val), op
    raise SystemExit(f"Bad --cut '{spec}'. Use e.g. \"W>=2.85\".")


M_E = 0.000511  # GeV, for E' = sqrt(|p_e|^2 + m_e^2)


def gen_xy(mode, mc_path):
    """Return (x, y) generator arrays for the chosen mode from a LUND file."""
    if mode == "q2ep":
        k = compute_kinematics_batch(mc_path)
        return np.asarray(k["Q2"]), np.asarray(k["Ep"])
    elif mode == "pmom":
        return proton_momenta_batch(mc_path, pid=2212)
    raise SystemExit(f"unknown mode {mode!r}")


def root_xy(mode, path, tree, cut_cols):
    """Return (x, y, cutvals) for the mode from a reconstructed ROOT TTree.

    Reads the analysis branches (Q2, P_mag_e, P_mag_p1, P_mag_p2, ...) and
    builds the same quantities that real_data.csv was made from:
        q2ep : x = Q2,                       y = E' = sqrt(P_mag_e^2 + m_e^2)
        pmom : x = p_lead = max(|p1|,|p2|),  y = p_sub = min(|p1|,|p2|)
    `cut_cols` are extra raw branch names (e.g. "W") needed for selections.
    Both the reco-sim and the real-data trees share this branch layout, so
    numerator and denominator are computed identically (reco level).
    """
    import uproot
    if mode == "q2ep":
        need = ["Q2", "P_mag_e"]
    elif mode == "pmom":
        need = ["P_mag_p1", "P_mag_p2"]
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    branches = sorted(set(need) | set(cut_cols))
    t = uproot.open(path)[tree]
    a = t.arrays(branches, library="np")

    if mode == "q2ep":
        x = a["Q2"]
        y = np.sqrt(a["P_mag_e"] ** 2 + M_E ** 2)
    else:  # pmom
        x = np.maximum(a["P_mag_p1"], a["P_mag_p2"])
        y = np.minimum(a["P_mag_p1"], a["P_mag_p2"])
    cutvals = {c: a[c] for c in cut_cols}
    return x, y, cutvals


def load_source(spec, tree, mode, cuts, tag):
    """Load (x, y) for `mode` from a CSV, LUND, or ROOT-tree source and apply
    `cuts`.  Source type is auto-detected from the file extension:
        *.csv  -> real-data style CSV (columns per MODES[mode]['data_cols'])
        *.lund -> generator truth LUND (no cuts supported)
        *.root -> reconstructed TTree `tree` (branch-based, cuts supported)
    """
    cut_cols = sorted({c[0] for c in cuts})

    if spec.endswith(".csv"):
        xcol, ycol = MODES[mode]["data_cols"]
        d = load_csv(spec, sorted({xcol, ycol, *cut_cols}))
        x, y = d[xcol], d[ycol]
        cutvals = {c: d[c] for c in cut_cols}
    elif spec.endswith(".lund"):
        if cuts:
            raise SystemExit("cuts are not supported on a LUND source "
                             f"({spec}); use a reconstructed .root tree.")
        x, y = gen_xy(mode, spec)
        cutvals = {}
    elif ".root" in spec:
        x, y, cutvals = root_xy(mode, spec, tree, cut_cols)
    else:
        raise SystemExit(f"Unknown source type for {spec!r} "
                         "(expected .csv, .lund, or .root).")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if cuts:
        mask = np.ones(len(x), dtype=bool)
        for col, op_fn, val, _op in cuts:
            mask &= op_fn(np.asarray(cutvals[col], dtype=float), val)
        desc = " AND ".join(f"{c[0]}{c[3]}{c[2]:g}" for c in cuts)
        print(f"[cut] {tag} selection [{desc}]: kept {int(mask.sum())}/{len(x)} "
              f"({100*mask.mean():.1f}%)")
        x, y = x[mask], y[mask]
    return x, y


def hist_density(x, y, x_lo, x_hi, y_lo, y_hi, nx, ny, tag):
    """Probability-normalized 2-D histogram of (x, y) on the given grid.

    Points outside [lo, hi] are dropped by np.histogram2d's range clip.
    Returns (prob[nx, ny], n_inside).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    H, _, _ = np.histogram2d(x, y, bins=[nx, ny],
                             range=[[x_lo, x_hi], [y_lo, y_hi]])
    n_inside = int(H.sum())
    print(f"[hist] {tag}: {n_inside} points inside range "
          f"(of {finite.sum()} finite)")
    if n_inside == 0:
        raise SystemExit(f"No {tag} points inside range.")
    return H / n_inside, n_inside


def build_from_data_hist(args, cfg, name):
    """
    Weight surface from a pre-built, sideband-subtracted DATA histogram.

    The .npz (written by sideband_subtract_nd in the analysis notebook) holds
    the subtracted bin `counts` and the `edges` for each axis. Those edges ARE
    the weight grid: the MC proposal g is binned on the SAME edges, and
    w = d / g is written as a variable-bin TH2D. Negative subtracted bins
    (statistical fluctuations) are clipped to zero, since neither a density nor
    an accept probability can be negative.
    """
    import array as _arr

    if not args.mc:
        raise SystemExit("--data-hist needs --mc (the g denominator).")

    npz    = np.load(args.data_hist, allow_pickle=True)
    vnames = [str(v) for v in npz["varnames"]]
    edges  = [np.asarray(e, dtype=float) for e in npz["edges"]]
    if len(edges) != 2:
        raise SystemExit(f"--data-hist expects a 2-D histogram, got {len(edges)}-D "
                         f"(vars={vnames}).")
    x_edges, y_edges = edges
    nx, ny = len(x_edges) - 1, len(y_edges) - 1

    counts = np.asarray(npz["counts"], dtype=float)
    D = np.clip(counts, 0.0, None)              # no negative density
    net = counts.sum()
    if D.sum() <= 0:
        raise SystemExit("Subtracted data histogram is <= 0 after clipping.")
    D = D / D.sum()
    print(f"[data-hist] {args.data_hist}: vars={vnames} grid={nx}x{ny} "
          f"net signal={net:.0f} (clipped {int((counts < 0).sum())} negative bins)")

    gx, gy = gen_xy(args.mode, args.mc)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    finite = np.isfinite(gx) & np.isfinite(gy)
    G, _, _ = np.histogram2d(gx[finite], gy[finite], bins=[x_edges, y_edges])
    if G.sum() <= 0:
        raise SystemExit("No MC points fell inside the data-hist edges.")
    G = G / G.sum()

    # rejection weight w = d / g; empty MC bins are unreachable -> w = 0
    W = np.divide(D, G, out=np.zeros_like(D), where=G > 0.0)
    wmax = W.max()
    if wmax <= 0:
        raise SystemExit("Weight surface is all zero (check --mode / edges / --mc).")
    W /= wmax
    print(f"[data-hist] w = d/g normalized: max=1.0, mean={W.mean():.3f}, "
          f"zero-fraction={float(np.mean(W <= 0.0)):.3f}")

    h = ROOT.TH2D(name, cfg["title"],
                  nx, _arr.array('d', x_edges),
                  ny, _arr.array('d', y_edges))
    for ix in range(nx):
        for iy in range(ny):
            h.SetBinContent(ix + 1, iy + 1, float(W[ix, iy]))

    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[data-hist] wrote {args.out}:{name}  mode={args.mode}  "
          f"grid={nx}x{ny}  (sideband-subtracted target)")


def _load_hist_npz(path, tag):
    """Load a 2-D histogram npz (keys 'varnames','edges','counts').

    Returns (varnames, x_edges, y_edges, counts). Same layout that
    sideband_subtract_nd writes and that build_from_data_hist consumes.
    """
    npz    = np.load(path, allow_pickle=True)
    vnames = [str(v) for v in npz["varnames"]]
    edges  = [np.asarray(e, dtype=float) for e in npz["edges"]]
    if len(edges) != 2:
        raise SystemExit(f"{tag} {path}: expected a 2-D histogram, got "
                         f"{len(edges)}-D (vars={vnames}).")
    counts = np.asarray(npz["counts"], dtype=float)
    return vnames, edges[0], edges[1], counts


def build_from_rec_hist(args, cfg, name):
    """
    Weight surface  w = N_sub / N_rec  for RECO-level matching.

    Numerator   : sideband-subtracted DATA histogram (--data-hist .npz, keys
                  'counts'+'edges' from sideband_subtract_nd).
    Denominator : RECONSTRUCTED simulation on the SAME edges (--rec), i.e. the
                  reco of the unweighted baseline generator that will be
                  reweighted. Sourced from a pre-binned .npz (same keys;
                  recommended, so the reco selection matches the notebook
                  exactly) or from a reco .root/.csv histogrammed on the
                  numerator edges.

    Unlike build_from_data_hist (w = d/g against the THROWN generator, which
    makes the *thrown* spectrum match data), this divides by the RECONSTRUCTED
    sim so that AFTER GEMC + reconstruction the reweighted sim matches the
    subtracted data. It folds the inverse acceptance (1/eps) into the weight:
    reco_weighted = w * g * eps = (N_sub / N_rec) * N_rec = N_sub, under the
    assumption of ~diagonal bin migration (thrown bin ~= reco bin). If
    migration is non-negligible, iterate: generate -> reconstruct -> recompute
    N_sub/N_rec (-> 1 at closure) -> multiply into the weight -> repeat.
    """
    import array as _arr

    # --- numerator: sideband-subtracted data ---
    _, x_edges, y_edges, counts = _load_hist_npz(args.data_hist, "[data-hist]")
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    net = counts.sum()
    D = np.clip(counts, 0.0, None)                 # no negative density
    if D.sum() <= 0:
        raise SystemExit("Subtracted data histogram is <= 0 after clipping.")

    # --- denominator: reconstructed sim on the SAME edges ---
    if args.rec.endswith(".npz"):
        _, rxe, rye, R = _load_hist_npz(args.rec, "[rec-hist]")
        if not (np.allclose(rxe, x_edges) and np.allclose(rye, y_edges)):
            raise SystemExit("--rec edges do not match --data-hist edges; "
                             "rebuild both on the same binning.")
        R = np.clip(R, 0.0, None)                  # counts, but guard anyway
    else:
        cuts = [parse_cut(c) for c in args.cut]
        rx, ry = load_source(args.rec, args.tree, args.mode, cuts, "rec")
        rx = np.asarray(rx, dtype=float)
        ry = np.asarray(ry, dtype=float)
        fin = np.isfinite(rx) & np.isfinite(ry)
        R, _, _ = np.histogram2d(rx[fin], ry[fin], bins=[x_edges, y_edges])
    if R.sum() <= 0:
        raise SystemExit("Reconstructed-sim histogram is empty on these edges.")

    # probability-normalize both, then w = N_sub / N_rec (overall constant is
    # absorbed by the max=1 rescale below).
    Dn = D / D.sum()
    Rn = R / R.sum()
    W  = np.divide(Dn, Rn, out=np.zeros_like(Dn), where=Rn > 0.0)

    # A low-statistics reco bin (tiny N_rec) produces a runaway ratio that,
    # after the max=1 rescale, sets wmax and crushes every other weight -- this
    # tanks the accept-reject efficiency AND spikes the thrown output into that
    # one bin. Guard with a minimum-N_rec floor and/or a cap on the raw ratio.
    nz = W > 0
    if args.min_rec > 0:
        killed = int(np.sum(nz & (R < args.min_rec)))
        W[R < args.min_rec] = 0.0                  # ratio not trustworthy -> drop
        nz = W > 0
        print(f"[rec-hist] min-rec={args.min_rec:g}: zeroed {killed} bins with "
              f"N_rec below threshold.")
    cap = None
    if args.wmax is not None:
        cap = float(args.wmax)
    elif args.wclip_pct is not None and nz.any():
        cap = float(np.percentile(W[nz], args.wclip_pct))
    if cap is not None:
        n_clip = int(np.sum(W > cap))
        W = np.minimum(W, cap)
        print(f"[rec-hist] clipped {n_clip} bins at raw-ratio cap={cap:.4g}.")

    wpk = W.max()
    if wpk <= 0:
        raise SystemExit("Weight surface is all zero (check --rec / edges).")
    W /= wpk                                        # accept-reject prob in [0,1]

    n_holes = int(np.sum((R <= 0) & (D > 0)))
    print(f"[rec-hist] w = N_sub/N_rec normalized: max=1.0, mean={W.mean():.3f}, "
          f"zero-fraction={float(np.mean(W <= 0.0)):.3f}")
    print(f"[rec-hist] net signal={net:.0f}, clipped {int((counts < 0).sum())} "
          f"negative data bins; {n_holes} bins have data but no reco-sim "
          f"(acceptance holes -> w=0, cannot be populated).")

    h = ROOT.TH2D(name, cfg["title"],
                  nx, _arr.array('d', x_edges),
                  ny, _arr.array('d', y_edges))
    for ix in range(nx):
        for iy in range(ny):
            h.SetBinContent(ix + 1, iy + 1, float(W[ix, iy]))

    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[rec-hist] wrote {args.out}:{name}  mode={args.mode}  "
          f"grid={nx}x{ny}  (reco-level N_sub/N_rec target)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(MODES), default="q2ep",
                    help="q2ep: w(Q2,E'); pmom: w(p_lead,p_sub) over the "
                         "two 2212 protons. Default q2ep.")
    ap.add_argument("--data", default=None,
                    help="CSV with the mode's columns (q2ep: Q2,Ep; "
                         "pmom: p_p1,p_p2). Required unless --data-hist is set.")
    ap.add_argument("--mc",   default=None,
                    help="MC LUND file for the g denominator. For q2ep this "
                         "is an UNWEIGHTED run; for pmom it is a run with the "
                         "q2ep weight already applied (sequential one-pass). "
                         "If omitted, a flat (uniform) proposal is assumed.")
    ap.add_argument("--out",  required=True, help="Output ROOT file")
    ap.add_argument("--name", default=None,
                    help="TH2D name (default depends on --mode)")
    # Generic axis ranges; --q2-range/--e-range kept as back-compat aliases.
    ap.add_argument("--x-range", default=None, help='"lo,hi" for the x axis')
    ap.add_argument("--y-range", default=None, help='"lo,hi" for the y axis')
    ap.add_argument("--q2-range", default=None, help="alias for --x-range")
    ap.add_argument("--e-range",  default=None, help="alias for --y-range")
    ap.add_argument("--nx", type=int, default=60,
                    help="Number of bins along x (also the TH2D x bins)")
    ap.add_argument("--ny", type=int, default=60,
                    help="Number of bins along y (also the TH2D y bins)")
    ap.add_argument("--cut", action="append", default=[],
                    help="Signal-region selection applied to the DATA only "
                         "(the MC already lives in the generator's phase "
                         "space), e.g. --cut \"W>=2.85\". Repeatable; cuts are "
                         "AND-ed. Needs the column present in the CSV.")
    ap.add_argument("--data-hist", default=None,
                    help="Pre-built sideband-subtracted DATA histogram (.npz "
                         "from sideband_subtract_nd: keys 'counts' + 'edges'). "
                         "When set, the numerator density comes from this file "
                         "instead of re-binning --data, and its bin edges "
                         "define the weight grid. Requires --mc (or --rec).")
    ap.add_argument("--rec", default=None,
                    help="RECONSTRUCTED-sim denominator for reco-level matching: "
                         "build w = N_sub/N_rec instead of d/g. Either a "
                         "pre-binned .npz on the SAME edges as --data-hist "
                         "(keys 'varnames','edges','counts'; recommended), or a "
                         "reco .root/.csv histogrammed onto those edges. Use "
                         "this (not --mc) when the RECONSTRUCTED sim must match "
                         "the data after GEMC. Requires --data-hist.")
    ap.add_argument("--tree", default="Individual",
                    help="TTree name for a reconstructed .root passed to --rec "
                         "(default 'Individual').")
    ap.add_argument("--min-rec", dest="min_rec", type=float, default=0,
                    help="For --rec: zero the weight in bins with N_rec below "
                         "this count (untrustworthy ratio). Default 0 (off).")
    ap.add_argument("--wmax", type=float, default=None,
                    help="For --rec: absolute cap on the raw N_sub/N_rec ratio "
                         "before the max=1 rescale (tames low-stat spikes).")
    ap.add_argument("--wclip-pct", dest="wclip_pct", type=float, default=None,
                    help="For --rec: cap the raw ratio at this percentile of the "
                         "nonzero bins (e.g. 95). Ignored if --wmax is set.")
    args = ap.parse_args()

    cfg  = MODES[args.mode]
    name = args.name or cfg["default_name"]
    xcol, ycol = cfg["data_cols"]

    # --- reco-level matching: w = N_sub / N_rec (reconstructed denominator) ---
    if args.rec:
        if not args.data_hist:
            raise SystemExit("--rec (reconstructed denominator) requires "
                             "--data-hist (the sideband-subtracted numerator).")
        build_from_rec_hist(args, cfg, name)
        return

    # --- pre-built subtracted-histogram path: d(x,y) is read, not re-binned ---
    if args.data_hist:
        build_from_data_hist(args, cfg, name)
        return

    if not args.data:
        raise SystemExit("Need --data (a CSV) unless --data-hist is given.")

    x_spec = args.x_range or args.q2_range
    y_spec = args.y_range or args.e_range
    if x_spec is None or y_spec is None:
        raise SystemExit("Need --x-range and --y-range (or the q2/e aliases).")
    x_lo, x_hi = parse_range(x_spec)
    y_lo, y_hi = parse_range(y_spec)

    # --- numerator: data density d(x, y) as a probability histogram ---
    cuts = [parse_cut(c) for c in args.cut]
    cut_cols = sorted({c[0] for c in cuts})
    d = load_csv(args.data, sorted({xcol, ycol, *cut_cols}))
    dx_, dy_ = d[xcol], d[ycol]
    if cuts:
        mask = np.ones(len(dx_), dtype=bool)
        for col, op_fn, val, op in cuts:
            mask &= op_fn(d[col], val)
        n0 = len(dx_)
        dx_, dy_ = dx_[mask], dy_[mask]
        desc = " AND ".join(f"{c[0]}{c[3]}{c[2]:g}" for c in cuts)
        print(f"[cut] data selection [{desc}]: kept {mask.sum()}/{n0} "
              f"({100*mask.mean():.1f}%)")
    D, _ = hist_density(dx_, dy_,
                        x_lo, x_hi, y_lo, y_hi, args.nx, args.ny, "data")

    # --- denominator: proposal/gen density g(x, y) ---
    if args.mc:
        gx, gy = gen_xy(args.mode, args.mc)
        G, _ = hist_density(gx, gy,
                            x_lo, x_hi, y_lo, y_hi, args.nx, args.ny, "mc")
    else:
        # Explicit uniform proposal: every bin equally probable.
        print("[hist] --mc not given: assuming a FLAT (uniform) proposal g. "
              "This is only valid if the generator samples (x,y) uniformly.")
        G = np.full((args.nx, args.ny), 1.0 / (args.nx * args.ny))

    # --- rejection weight w = d / g, guarding the 0/0 in empty MC bins ---
    # An empty MC bin means the proposal never reaches that region, so no
    # amount of rejection can populate it -> w = 0 there.
    W = np.where(G > 0.0, D / G, 0.0)

    wmax = W.max()
    if wmax <= 0:
        raise SystemExit("Weight surface is all zero (check ranges / inputs).")
    W /= wmax                        # accept-reject probability in [0, 1]
    frac_zero = float(np.mean(W <= 0.0))
    print(f"[hist] w = d/g normalized: max=1.0, mean={W.mean():.3f}, "
          f"zero-fraction={frac_zero:.3f}")

    h = ROOT.TH2D(name, cfg["title"],
                  args.nx, x_lo, x_hi,
                  args.ny, y_lo, y_hi)
    for ix in range(args.nx):
        for iy in range(args.ny):
            h.SetBinContent(ix + 1, iy + 1, float(W[ix, iy]))

    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[hist] wrote {args.out}:{name}  mode={args.mode}  "
          f"bins={args.nx}x{args.ny}  range "
          f"{cfg['xlabel']}=[{x_lo},{x_hi}]  {cfg['ylabel']}=[{y_lo},{y_hi}]")


if __name__ == "__main__":
    main()
