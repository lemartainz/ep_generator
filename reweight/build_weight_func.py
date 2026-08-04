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

Reco-level denominator (--rec)
------------------------------
For the iterative data-driven scheme the denominator must be the
RECONSTRUCTED simulation (post-GEMC/GEANT), not the generator truth. The
ppbar_weights.ipynb notebook writes both histograms on identical edges:

    save_subtracted_nd -> subtracted_q2ep.npz  (numerator D: subtracted DATA)
    save_rec_hist_nd   -> rec_q2ep.npz         (denominator R: reconstructed SIM)

Pass D via --data-hist and R via --rec; the weight is then the reco-level
ratio w = D_subtracted / R_reconstructed. (--mc, which bins generator TRUTH
kinematics from a LUND, remains available as the truth-level fallback.)

Low-statistics guards (--min-rec / --wmax / --wclip-pct)
--------------------------------------------------------
The weight is renormalized so max(w) = 1, which means the single largest bin
of the raw d/g ratio sets the scale for every other bin. A denominator bin
holding one or two entries produces a ratio with 100%-level error, and that
noise bin is usually the maximum -- so every trustworthy bin gets divided by
it. The accepted SHAPE is unaffected (rejection sampling is invariant to the
constant C), but the keep fraction collapses and the generator burns most of
its samples.

    --min-rec N     drop bins whose denominator has fewer than N raw counts
    --wclip-pct P   cap the raw ratio at its P-th percentile over nonzero bins
    --wmax V        cap the raw ratio at V (overrides --wclip-pct)

All three act on the per-pass d/g correction, before --prev is multiplied in.
Start with --min-rec 5: it removes the untrustworthy bins without touching
where the signal actually lives, and typically buys a several-fold speedup.

Iterative reweighting (--prev / --archive)
------------------------------------------
The D/R ratio is measured in RECONSTRUCTED kinematics but applied at
GENERATOR (truth) level, so one pass does not land exactly on the data -- you
iterate until D/R -> 1. Each pass measures the residual correction from a run
that ALREADY has the previous weight applied (its reco -> a fresh rec_*.npz),
so the generator must see the running PRODUCT:

        w_total_{n+1} = w_total_n * (d / g_n),   renormalized to max = 1.

--prev feeds the previous cumulative surface in; the new correction is
multiplied into it and renormalized before writing --out (which stays what
the generator reads via `weight_func:` in input.txt). --archive keeps a
versioned copy of each pass so you can plot d/g -> 1 and roll back.

    # iteration 0: no --prev, seed the archive. --rec is the reco of the
    # UNWEIGHTED baseline run.
    python build_weight_func.py --data-hist subtracted_q2ep.npz \\
        --rec rec_q2ep.npz --out weight_func.root \\
        --name w_Q2_Ep --archive reweight/iters
    # -> weight_func.root  and  reweight/iters/w_q2ep_iter0.root

    # iteration n>=1: --rec is the reco of the run that used the PREVIOUS
    # surface, --prev is the last archived cumulative weight
    python build_weight_func.py --data-hist subtracted_q2ep.npz \\
        --rec rec_q2ep_iter1.npz --out weight_func.root \\
        --name w_Q2_Ep --prev reweight/iters/w_q2ep_iter0.root \\
        --archive reweight/iters
    # -> weight_func.root (cumulative)  and  reweight/iters/w_q2ep_iter1.root

All iterations MUST share identical bin edges (keep the same --data-hist /
--rec grid), or the bin-by-bin product misaligns. --prev is optional: omit it
for iteration 0, add it (pointing at the last archived iter) for every pass
after.

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
import glob
import os
import re
import shutil
from array import array as _darr

import numpy as np
import ROOT

from kinematics import compute_kinematics_batch, proton_momenta_batch


def _read_prev_surface(path, name, x_edges, y_edges):
    """Load a previous cumulative-weight TH2D as an [nx, ny] numpy array.

    The stored surface must share the current grid exactly (same bin edges),
    otherwise a bin-by-bin product would silently misalign kinematics across
    reweight iterations.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    if not os.path.exists(path):
        raise SystemExit(f"--prev: file not found: {path}")
    tf = ROOT.TFile(path)                 # constructor, not TFile.Open (segfaults)
    if tf.IsZombie():
        tf.Close()
        raise SystemExit(f"--prev: cannot open {path}")
    h = tf.Get(name)
    if not h:
        tf.Close()
        raise SystemExit(f"--prev: no TH2 named '{name}' in {path}")
    if h.GetNbinsX() != nx or h.GetNbinsY() != ny:
        tf.Close()
        raise SystemExit(
            f"--prev grid {h.GetNbinsX()}x{h.GetNbinsY()} != current {nx}x{ny}; "
            "reweight iterations must share binning.")
    xe = np.array([h.GetXaxis().GetBinLowEdge(i + 1) for i in range(nx)]
                  + [h.GetXaxis().GetBinUpEdge(nx)])
    ye = np.array([h.GetYaxis().GetBinLowEdge(i + 1) for i in range(ny)]
                  + [h.GetYaxis().GetBinUpEdge(ny)])
    if not (np.allclose(xe, x_edges) and np.allclose(ye, y_edges)):
        tf.Close()
        raise SystemExit(
            "--prev bin edges differ from the current grid; reweight "
            "iterations must share identical edges.")
    prev = np.array([[h.GetBinContent(ix + 1, iy + 1) for iy in range(ny)]
                     for ix in range(nx)], dtype=float)
    tf.Close()
    return prev


def _load_rec_hist(path, x_edges, y_edges, data_vnames):
    """Load the reconstructed-sim denominator R from an npz on identical edges.

    The npz (written by save_rec_hist_nd in ppbar_weights.ipynb) has keys
    'counts' + 'edges', binned on the SAME grid as the sideband-subtracted
    target. This is the RECONSTRUCTED simulation (post-GEMC/GEANT), so
    w = D_subtracted / R_reconstructed is a reco-level weight. Returns
    (probability-normalized [nx, ny] array, raw counts), the latter so the
    low-statistics guards can be applied on actual bin counts.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--rec: file not found: {path}")
    rz = np.load(path, allow_pickle=True)
    r_edges = [np.asarray(e, float) for e in rz["edges"]]
    if len(r_edges) != 2:
        raise SystemExit(f"--rec {path}: expected a 2-D histogram, got "
                         f"{len(r_edges)}-D.")
    if not (np.allclose(r_edges[0], x_edges) and np.allclose(r_edges[1], y_edges)):
        raise SystemExit(f"--rec {path}: bin edges differ from the --data-hist "
                         "grid; numerator D and denominator R must share edges.")
    if "varnames" in rz:
        r_vnames = [str(v) for v in rz["varnames"]]
        if r_vnames != list(data_vnames):
            print(f"[warn] --rec var order {r_vnames} != data {list(data_vnames)}; "
                  "make sure the axes line up.")
    R = np.clip(np.asarray(rz["counts"], dtype=float), 0.0, None)
    if R.sum() <= 0:
        raise SystemExit(f"--rec {path}: reconstructed histogram is empty.")
    print(f"[rec] {path}: N_rec={R.sum():.0f} (reconstructed-sim denominator)")
    return R / R.sum(), R


def _archive_copy(out_path, archive_dir, mode):
    """Copy the just-written surface to archive_dir/w_<mode>_iter<N>.root.

    N is one past the highest existing index in the directory, so successive
    calls leave w_<mode>_iter0.root, w_<mode>_iter1.root, ... as a rollback and
    convergence trail. Returns the destination path.
    """
    os.makedirs(archive_dir, exist_ok=True)
    idxs = []
    for p in glob.glob(os.path.join(archive_dir, f"w_{mode}_iter*.root")):
        m = re.search(rf"w_{re.escape(mode)}_iter(\d+)\.root$",
                      os.path.basename(p))
        if m:
            idxs.append(int(m.group(1)))
    n = max(idxs) + 1 if idxs else 0
    dst = os.path.join(archive_dir, f"w_{mode}_iter{n}.root")
    shutil.copy(out_path, dst)
    return dst


def apply_ratio_guards(W, denom_counts, args):
    """Tame runaway d/g ratios driven by low-statistics denominator bins.

    A bin whose denominator holds a couple of entries yields a ratio with
    100%-level statistical error, and that noise bin is usually the MAXIMUM of
    the surface. The max=1 rescale in finalize_and_write then divides every
    trustworthy bin by it: the shape survives (accept-reject is invariant to an
    overall constant) but the keep fraction collapses, so the generator throws
    away most of its samples for nothing.

    --min-rec zeroes bins whose denominator is below a raw-count floor;
    --wmax / --wclip-pct cap the raw ratio. Both act on the per-pass d/g
    correction, before --prev is multiplied in, since it is this pass's
    statistics that decide which bins are trustworthy.

    `denom_counts` must be RAW counts on the weight grid, not a normalized
    density, or the --min-rec floor is meaningless.
    """
    nz = W > 0
    if args.min_rec > 0:
        killed = int(np.sum(nz & (denom_counts < args.min_rec)))
        W = np.where(denom_counts < args.min_rec, 0.0, W)
        nz = W > 0
        print(f"[guard] min-rec={args.min_rec:g}: zeroed {killed} bins whose "
              f"denominator is below the count floor.")

    cap = None
    if args.wmax is not None:
        cap = float(args.wmax)
    elif args.wclip_pct is not None and nz.any():
        cap = float(np.percentile(W[nz], args.wclip_pct))
    if cap is not None:
        n_clip = int(np.sum(W > cap))
        W = np.minimum(W, cap)
        print(f"[guard] clipped {n_clip} bins at raw-ratio cap={cap:.4g}.")

    if W.max() <= 0:
        raise SystemExit(
            "Weight surface is all zero after the low-statistics guards; "
            "lower --min-rec or raise --wclip-pct.")
    return W


def finalize_and_write(W, name, cfg, args, x_edges, y_edges, label=""):
    """Multiply in --prev (if any), renormalize max->1, write --out, archive.

    W is the raw d/g correction for this pass (0 in unreachable bins). When
    --prev is given, W is multiplied into the previous cumulative surface so
    that --out holds the running product

        w_total_{n+1} = w_total_n * (d / g)

    across reweight iterations; the result is renormalized to max=1 so it stays
    a valid accept-reject probability. With no --prev this is the iteration-0
    surface (current behavior). --archive additionally drops a versioned copy.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1

    if args.prev:
        prev = _read_prev_surface(args.prev, name, x_edges, y_edges)
        W = W * prev
        if W.max() <= 0:
            raise SystemExit(
                "Cumulative weight is all zero after multiplying --prev "
                "(new correction and previous surface have no common support).")
        print(f"[prev] multiplied in {args.prev} -> cumulative surface")

    W = W / W.max()                       # accept-reject probability in [0, 1]
    kind = "cumulative" if args.prev else "iteration-0"
    print(f"[{kind}] w normalized: max=1.0, mean={W.mean():.3f}, "
          f"zero-fraction={float(np.mean(W <= 0.0)):.3f}")

    xe = _darr('d', [float(e) for e in x_edges])
    ye = _darr('d', [float(e) for e in y_edges])
    h = ROOT.TH2D(name, cfg["title"], nx, xe, ny, ye)
    for ix in range(nx):
        for iy in range(ny):
            h.SetBinContent(ix + 1, iy + 1, float(W[ix, iy]))
    tf = ROOT.TFile(args.out, "RECREATE")
    h.Write()
    tf.Close()
    print(f"[write] {args.out}:{name}  mode={args.mode}  grid={nx}x{ny}"
          + (f"  {label}" if label else ""))

    if args.archive:
        dst = _archive_copy(args.out, args.archive, args.mode)
        print(f"[archive] stored -> {dst}")


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
    if not (args.rec or args.mc):
        raise SystemExit("--data-hist needs a denominator: --rec <reco npz> "
                         "(reconstructed sim, for reco-level weights) or "
                         "--mc <lund> (generator truth).")

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

    # --- denominator g: reconstructed sim (--rec, reco-level) or LUND truth ---
    # G is probability-normalized for the ratio; G_counts keeps the raw bin
    # counts so apply_ratio_guards can judge which bins have enough statistics.
    if args.rec:
        G, G_counts = _load_rec_hist(args.rec, x_edges, y_edges, vnames)
    else:
        gx, gy = gen_xy(args.mode, args.mc)
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)
        finite = np.isfinite(gx) & np.isfinite(gy)
        G_counts, _, _ = np.histogram2d(gx[finite], gy[finite],
                                        bins=[x_edges, y_edges])
        if G_counts.sum() <= 0:
            raise SystemExit("No MC points fell inside the data-hist edges.")
        G = G_counts / G_counts.sum()

    # rejection weight w = d / g; empty MC bins are unreachable -> w = 0
    W = np.divide(D, G, out=np.zeros_like(D), where=G > 0.0)
    if W.max() <= 0:
        raise SystemExit("Weight surface is all zero (check --mode / edges / --mc).")

    W = apply_ratio_guards(W, G_counts, args)

    n_holes = int(np.sum((G_counts <= 0) & (D > 0)))
    print(f"[data-hist] {n_holes} bins have data but no denominator "
          "(acceptance holes -> w=0, cannot be populated).")

    finalize_and_write(W, name, cfg, args, x_edges, y_edges,
                       label="(sideband-subtracted target)")


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
                    help="Reconstructed-sim histogram npz (save_rec_hist_nd in "
                         "ppbar_weights.ipynb: keys 'counts'+'edges' on the SAME "
                         "grid as --data-hist). Used as the denominator R so "
                         "w = D_subtracted / R_reconstructed is a RECO-level "
                         "weight. Preferred over --mc for --data-hist; requires "
                         "--data-hist.")
    ap.add_argument("--min-rec", type=float, default=25.0,
                    help="Zero any bin whose DENOMINATOR holds fewer than this "
                         "many raw counts. Such a bin gives a d/g ratio with "
                         "100%%-level error that typically sets the surface "
                         "maximum and craters the accept-reject keep fraction. "
                         "5-10 is a reasonable floor; 0 (default) disables.")
    ap.add_argument("--wmax", type=float, default=None,
                    help="Hard cap on the raw d/g ratio before the max=1 "
                         "rescale. Takes precedence over --wclip-pct.")
    ap.add_argument("--wclip-pct", type=float, default=None,
                    help="Cap the raw d/g ratio at this percentile of the "
                         "nonzero bins (e.g. 99). Ignored when --wmax is set.")
    ap.add_argument("--prev", default=None,
                    help="Previous CUMULATIVE weight ROOT file (same TH2D name "
                         "and identical binning). The new d/g correction is "
                         "multiplied into it and renormalized, so --out holds "
                         "the running product across reweight iterations. Omit "
                         "for the first (iteration-0) pass.")
    ap.add_argument("--archive", default=None,
                    help="Directory to also store a versioned copy of the "
                         "written surface as w_<mode>_iter<N>.root (N "
                         "auto-incremented). Keeps every iteration for "
                         "convergence checks and rollback.")
    args = ap.parse_args()

    cfg  = MODES[args.mode]
    name = args.name or cfg["default_name"]
    xcol, ycol = cfg["data_cols"]

    if args.rec and not args.data_hist:
        raise SystemExit("--rec is the reconstructed-sim denominator for the "
                         "--data-hist path; pass --data-hist too (or drop --rec).")

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
        G, n_mc = hist_density(gx, gy,
                               x_lo, x_hi, y_lo, y_hi, args.nx, args.ny, "mc")
        G_counts = G * n_mc            # raw counts back out, for --min-rec
    else:
        # Explicit uniform proposal: every bin equally probable.
        print("[hist] --mc not given: assuming a FLAT (uniform) proposal g. "
              "This is only valid if the generator samples (x,y) uniformly.")
        G = np.full((args.nx, args.ny), 1.0 / (args.nx * args.ny))
        # An analytic proposal has no sampling error, so no bin is low-stat.
        G_counts = np.full((args.nx, args.ny), np.inf)

    # --- rejection weight w = d / g, guarding the 0/0 in empty MC bins ---
    # An empty MC bin means the proposal never reaches that region, so no
    # amount of rejection can populate it -> w = 0 there.
    W = np.where(G > 0.0, D / G, 0.0)
    if W.max() <= 0:
        raise SystemExit("Weight surface is all zero (check ranges / inputs).")

    W = apply_ratio_guards(W, G_counts, args)

    x_edges = np.linspace(x_lo, x_hi, args.nx + 1)
    y_edges = np.linspace(y_lo, y_hi, args.ny + 1)
    finalize_and_write(W, name, cfg, args, x_edges, y_edges,
                       label=f"range {cfg['xlabel']}=[{x_lo},{x_hi}] "
                             f"{cfg['ylabel']}=[{y_lo},{y_hi}]")


if __name__ == "__main__":
    main()
