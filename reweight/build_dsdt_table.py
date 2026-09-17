"""
Tabulate dsigma/dt(s, t) as the TH2D the generator's ratio weight reads.

The generator (EventWeighter.h, `ratio_weight:`) carries a per-event weight

    w_ratio = dsigma/dt(s_pbarp, t) / dsigma/dt(s_pp, t)

for e p -> e' p pbar p: ONE parametrization of the elastic dsigma/dt,
evaluated at the pbar-p and the p-p sub-energies of the same event, at the
event's t. This script turns your parametrization into that TH2D. Nothing
here is normalized -- any overall constant cancels in the ratio.

Four ways to give dsigma/dt(s, t)
---------------------------------
    --from-gen truth.root
        NOT a model: the generator's own (s, t) density D_gen, i.e. the
        truth ntuple of an UNWEIGHTED run histogrammed in (s, t) on the
        grid below. The generator divides each hypothesis model by it
        (`ratio_weight_gen:`), so that the weighted sample follows the
        model instead of model x generator shape -- the same model /
        generated construction as every other weight here. It is looked
        up PER BIN (no interpolation, no --log), which makes the closure
        exact cell by cell. --var picks which s fills it: s_pbarp, s_pp
        or both (default; the two are identically distributed in the
        generator, so this is the same density with twice the entries).
        --smooth <bins> Gaussian-filters the counts (costs exactness).
        Output name defaults to dgen_s_t.
    --formula "exp((4.0 + 0.5*log(s))*t) * pow(s,-2)"
        a numpy expression in s and t (plus the functions in xsec.py's
        whitelist). Use `pow` and `log` (natural), so the identical string
        also works as `ratio_weight_formula:` in the input card.
    --xsec-py model.py:dsdt
        a Python function f(s, t) -> array, e.g. a fit to the Ambats et al.
        (small |t|) and White et al. (large |t|) pbar-p / p-p data.
    --table dsdt.csv --cols s,t,dsdt
        scattered (s, t, dsigma/dt) points, linearly interpolated onto the
        grid (scipy griddata); grid points outside the table get 0.

Grid
----
x axis = s [GeV^2], y axis = t [GeV^2] with t_lo < t_hi <= 0 (write it as
`--t-range=-16,0`: a value starting with `-` needs the `=`). The generator
looks the table up with TH2::Interpolate, i.e. bilinearly BETWEEN BIN
CENTERS -- so the function is evaluated AT the centers, not averaged over
the bin (unlike the accept-reject builders, whose per-bin ratio must be a
bin integral). Make the grid fine enough that a bilinear patch follows the
function: check with --gen, which reports the interpolation error on real
events.

--log stores ln(dsigma/dt) instead. The generator then interpolates in log
space and exponentiates (`ratio_weight_mode: log`). For an exponential in t
that turns a curved function into a plane, so the interpolation error drops
by orders of magnitude at the same binning. Recommended.

--gen truth.root
----------------
Reads the truth ntuple of a previous run (branches s_pbarp, s_pp, t, and
w_ratio) and reports the ranges you need to cover, the fraction of events
inside the grid (the generator gives w = 0 outside), and -- with a model on
the command line -- the interpolation error the table will incur, without
a second generator run. If the ntuple came from a run that already carried
a ratio weight, its w_ratio is compared to the model too.

Usage
-----
    # 1. unweighted run with `truth_ntuple: gen_truth.root` in the card
    # 2. choose the grid from the ranges it reports, build the table
    cd reweight
    python build_dsdt_table.py --formula "exp((4.0 + 0.5*log(s))*t) * pow(s,-2)" \\
        --s-range 3.5,20.5 --ns 100 --t-range=-16,0 --nt 200 --log \\
        --gen gen_truth.root --out ../dsdt_table.root
    # 3. add the printed `ratio_weight:` / `ratio_weight_mode:` lines and rerun
"""

import argparse
import os
import sys
from array import array as _darr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xsec                                                      # noqa: E402

LOG_FLOOR = -700.0   # ln(dsigma/dt) for dsigma/dt <= 0: exp() underflows to 0


# --------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------- #
def parse_edges(rng, n, edges, tag):
    if edges:
        e = np.array([float(v) for v in edges.split(",")])
        if len(e) < 2 or np.any(np.diff(e) <= 0):
            raise SystemExit(f"--{tag}-edges must be increasing, got {edges}")
        return e
    if not rng:
        raise SystemExit(f"need --{tag}-range lo,hi (or --{tag}-edges)")
    lo, hi = (float(v) for v in rng.split(","))
    if hi <= lo:
        raise SystemExit(f"--{tag}-range: hi must exceed lo, got {rng}")
    return np.linspace(lo, hi, n + 1)


def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------- #
# the model, evaluated at (s, t) arrays of any (matching) shape
# --------------------------------------------------------------------- #
def make_model(args):
    """Return f(S, T) -> dsigma/dt array, from whichever input was given."""
    given = [k for k in ("formula", "xsec_py", "table") if getattr(args, k)]
    if len(given) != 1:
        raise SystemExit("give exactly one of --formula / --xsec-py / --table.")

    if args.formula:
        expr = args.formula

        def f(S, T):
            ns = dict(xsec._SAFE_FUNCS)
            ns.update({"s": S, "t": T, "x": S, "y": T, "M_P": xsec.M_P})
            try:
                val = eval(expr, {"__builtins__": {}}, ns)      # noqa: S307
            except Exception as err:
                raise SystemExit(f"--formula failed: {err}\n"
                                 "Available: s, t (aliases x, y), M_P and "
                                 "the numpy functions (exp, log, pow, ...).")
            return np.broadcast_to(np.asarray(val, float), S.shape).astype(float)
        return f, f"formula: {expr}"

    if args.xsec_py:
        fn = xsec.load_callable(args.xsec_py)

        def f(S, T):
            return np.broadcast_to(np.asarray(fn(S, T), float), S.shape).astype(float)
        return f, f"python: {args.xsec_py}"

    from scipy.interpolate import griddata
    cols = tuple(args.cols.split(","))
    if len(cols) != 3:
        raise SystemExit("--cols expects three names: s,t,dsdt")
    ts, tt, td = xsec._read_table(args.table, cols)
    if len(ts) < 3:
        raise SystemExit(f"--table {args.table}: need at least 3 points.")
    pts = np.column_stack([ts, tt])
    print(f"[table] {args.table}: {len(ts)} points, "
          f"s=[{ts.min():g},{ts.max():g}] t=[{tt.min():g},{tt.max():g}]")

    def f(S, T):
        out = griddata(pts, td, (S, T), method="linear")
        return np.where(np.isfinite(out), out, 0.0)
    return f, f"table: {args.table}"


# --------------------------------------------------------------------- #
# --from-gen: the generator's own (s, t) distribution as the model
# --------------------------------------------------------------------- #
def hist_from_gen(path, tree, var, s_edges, t_edges, smooth):
    import uproot
    if not os.path.exists(path):
        raise SystemExit(f"--from-gen: file not found: {path}")
    t = uproot.open(path)[tree]
    need = ["s_pbarp", "s_pp", "t"]
    missing = [b for b in need if b not in t.keys()]
    if missing:
        raise SystemExit(f"--from-gen {path}:{tree} lacks branches {missing}.")
    a = t.arrays(need, library="np")
    cols = ["s_pbarp", "s_pp"] if var == "both" else [var]
    S = np.concatenate([a[c] for c in cols])
    T = np.concatenate([a["t"] for _ in cols])
    ok = np.isfinite(S) & np.isfinite(T)
    H, _, _ = np.histogram2d(S[ok], T[ok], bins=[s_edges, t_edges])
    n_in = int(H.sum())
    print(f"[gen] {path}:{tree}: {len(a['t'])} events, filled with "
          f"{'+'.join(cols)} -> {n_in} entries on the grid "
          f"({100.0*n_in/max(ok.sum(),1):.1f}% of the fills), "
          f"{int(np.sum(H == 0))}/{H.size} empty cells")
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=smooth, mode="nearest")
        print(f"[gen] Gaussian-smoothed with sigma = {smooth:g} bins")
    # density: divide by the bin area so a non-uniform grid stays honest
    area = np.outer(np.diff(s_edges), np.diff(t_edges))
    D = H / area
    return D, "from-gen: generated density D_gen(s, t) = dN/ds dt (per-bin lookup)"


# --------------------------------------------------------------------- #
# ROOT output
# --------------------------------------------------------------------- #
def write_th2(D, name, title, s_edges, t_edges, out):
    import ROOT
    nx, ny = D.shape
    xe = _darr('d', [float(v) for v in s_edges])
    ye = _darr('d', [float(v) for v in t_edges])
    h = ROOT.TH2D(name, title, nx, xe, ny, ye)
    for i in range(nx):
        for j in range(ny):
            h.SetBinContent(i + 1, j + 1, float(D[i, j]))
    tf = ROOT.TFile(out, "RECREATE")
    h.Write()
    tf.Close()


# --------------------------------------------------------------------- #
# --gen: coverage and interpolation error on real events
# --------------------------------------------------------------------- #
def table_lookup(D, s_edges, t_edges, S, T):
    """What the generator computes: bilinear between bin centers, clamped
    into the center hull (EventWeighter::Surface2D::interpolateClamped)."""
    from scipy.interpolate import RegularGridInterpolator
    sc, tc = centers(s_edges), centers(t_edges)
    rgi = RegularGridInterpolator((sc, tc), D, method="linear",
                                  bounds_error=False, fill_value=None)
    Sq = np.clip(S, sc[0], sc[-1])
    Tq = np.clip(T, tc[0], tc[-1])
    return rgi(np.column_stack([Sq, Tq]))


def report_gen(path, tree, model, D, s_edges, t_edges, use_log):
    import uproot
    if not os.path.exists(path):
        raise SystemExit(f"--gen: file not found: {path}")
    t = uproot.open(path)[tree]
    need = ["s_pbarp", "s_pp", "t"]
    missing = [b for b in need if b not in t.keys()]
    if missing:
        raise SystemExit(f"--gen {path}:{tree} lacks branches {missing}; "
                         "rerun the generator (they were added with the "
                         "ratio weight).")
    a = t.arrays(need + (["w_ratio"] if "w_ratio" in t.keys() else []),
                 library="np")
    ok = np.isfinite(a["s_pbarp"]) & np.isfinite(a["s_pp"]) & np.isfinite(a["t"])
    n = int(ok.sum())
    print(f"[gen] {path}:{tree}: {len(ok)} rows, {n} with finite (s_pbarp, s_pp, t)")
    if n == 0:
        return
    s1, s2, tt = a["s_pbarp"][ok], a["s_pp"][ok], a["t"][ok]

    q = (0, 1, 50, 99, 100)
    print("      percentile   " + "".join(f"{v:>10d}%" for v in q))
    for nm, v in (("s_pbarp", s1), ("s_pp", s2), ("t", tt)):
        print(f"      {nm:10s}  " + "".join(f"{x:11.3f}" for x in np.percentile(v, q)))

    # The generator's covers() test: strict inequalities on the axis RANGE,
    # for BOTH s values and t.
    def inside(v, e):
        return (v > e[0]) & (v < e[-1])
    cov = inside(s1, s_edges) & inside(s2, s_edges) & inside(tt, t_edges)
    print(f"      grid s=[{s_edges[0]:g},{s_edges[-1]:g}] t=[{t_edges[0]:g},{t_edges[-1]:g}]: "
          f"{100.0*cov.mean():.2f}% of events inside (the rest get w_ratio = 0)")

    # What the generator will compute on the covered events.
    num = table_lookup(D, s_edges, t_edges, s1[cov], tt[cov])
    den = table_lookup(D, s_edges, t_edges, s2[cov], tt[cov])
    if use_log:
        num, den = np.exp(num), np.exp(den)
    w_tab = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    if model is None:
        # --from-gen: D_gen is looked up per bin. Report how many events
        # would fall in an empty cell (they get w = 0 in the generator).
        def cell(S_, T_):
            i = np.clip(np.searchsorted(s_edges, S_, side="right") - 1, 0, D.shape[0] - 1)
            j = np.clip(np.searchsorted(t_edges, T_, side="right") - 1, 0, D.shape[1] - 1)
            return D[i, j]
        empty = (cell(s1[cov], tt[cov]) <= 0) | (cell(s2[cov], tt[cov]) <= 0)
        print(f"      D_gen per-bin lookup: {100.0*empty.mean():.3f}% of covered events "
              f"hit an empty cell for s_pbarp or s_pp (w = 0 there); "
              f"min occupied cell = {D[D > 0].min() * np.diff(s_edges).min() * np.diff(t_edges).min():.0f} entries")
        return
    exact = model(s1[cov], tt[cov]) / model(s2[cov], tt[cov])
    good = np.isfinite(exact) & (exact > 0) & np.isfinite(num) & (den > 0)
    if good.any():
        rel = np.abs(num[good] / den[good] / exact[good] - 1.0)
        print(f"      table vs model on {int(good.sum())} events: "
              f"|w_table/w_exact - 1|  median {np.median(rel):.2e}  "
              f"99% {np.percentile(rel, 99):.2e}  max {rel.max():.2e}"
              + ("" if use_log else "   (try --log if this is too large)"))

    # If the run already carried a ratio weight, check the generator's own
    # numbers against the model (this is the table-vs-formula closure when
    # the ntuple came from a `ratio_weight:` run).
    if "w_ratio" in a:
        w = a["w_ratio"][ok][cov]
        if not np.all(w == 1.0):
            g2 = good & np.isfinite(w) & (w > 0)
            rel = np.abs(w[g2] / exact[g2] - 1.0)
            print(f"      generator w_ratio vs model on {int(g2.sum())} events: "
                  f"median {np.median(rel):.2e}  99% {np.percentile(rel, 99):.2e}  "
                  f"max {rel.max():.2e}   (mean w = {w[g2].mean():.5f})")


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Tabulate dsigma/dt(s, t) as the TH2D behind the "
                    "generator's carried ratio weight.")
    m = ap.add_argument_group("dsigma/dt model (exactly one)")
    m.add_argument("--formula", default=None,
                   help='numpy expression in s, t, e.g. '
                        '"exp((4.0 + 0.5*log(s))*t) * pow(s,-2)"')
    m.add_argument("--xsec-py", default=None,
                   help="path/to/model.py:func with func(s, t) -> array")
    m.add_argument("--table", default=None,
                   help="CSV/npz of scattered (s, t, dsigma/dt) points")
    m.add_argument("--from-gen", default=None,
                   help="truth ntuple: histogram the generated (s, t) on the "
                        "grid and use that as D(s, t)")
    m.add_argument("--var", choices=("both", "s_pbarp", "s_pp"), default="both",
                   help="which s fills the --from-gen histogram (default both)")
    m.add_argument("--smooth", type=float, default=0.0,
                   help="--from-gen: Gaussian filter sigma in bins (default 0)")
    m.add_argument("--cols", default="s,t,dsdt",
                   help="column names in --table (default s,t,dsdt)")

    g = ap.add_argument_group("grid (x = s, y = t, both GeV^2)")
    g.add_argument("--s-range", default=None, help='"lo,hi"')
    g.add_argument("--ns", type=int, default=100)
    g.add_argument("--s-edges", default=None, help="comma list, overrides --s-range/--ns")
    g.add_argument("--t-range", default=None,
                   help='"lo,hi" with lo < hi <= 0; write --t-range=-16,0 '
                        '(the leading minus needs the =)')
    g.add_argument("--nt", type=int, default=200)
    g.add_argument("--t-edges", default=None, help="comma list, overrides --t-range/--nt")

    o = ap.add_argument_group("output")
    o.add_argument("--out", required=True, help="output ROOT file")
    o.add_argument("--name", default=None,
                   help="TH2D name (default dsdt_s_t, or log_dsdt_s_t with --log)")
    o.add_argument("--log", action="store_true",
                   help="store ln(dsigma/dt); pair with `ratio_weight_mode: log`")
    o.add_argument("--gen", default=None,
                   help="truth ntuple of a previous run: report coverage and "
                        "interpolation error on its events")
    o.add_argument("--tree", default="truth")
    args = ap.parse_args()

    s_edges = parse_edges(args.s_range, args.ns, args.s_edges, "s")
    t_edges = parse_edges(args.t_range, args.nt, args.t_edges, "t")
    if t_edges[-1] > 0:
        print(f"[warn] t axis reaches {t_edges[-1]:g} > 0; physical t is <= 0.")

    if args.from_gen:
        for k in ("formula", "xsec_py", "table"):
            if getattr(args, k):
                raise SystemExit("--from-gen cannot be combined with "
                                 "--formula / --xsec-py / --table.")
        if args.log:
            raise SystemExit("--from-gen is looked up per bin; --log does not apply.")
        D, label = hist_from_gen(args.from_gen, args.tree, args.var, s_edges,
                                 t_edges, args.smooth)
        model = None                       # a density, not a model
        if args.gen is None:
            args.gen = args.from_gen       # report coverage on the same events
    else:
        model, label = make_model(args)
        S, T = np.meshgrid(centers(s_edges), centers(t_edges), indexing="ij")
        D = model(S, T)
    print(f"[dsdt] {label}")
    n_bad = int(np.sum(~np.isfinite(D)))
    D = np.where(np.isfinite(D), D, 0.0)
    D = np.clip(D, 0.0, None)
    n_zero = int(np.sum(D <= 0))
    print(f"[dsdt] grid {D.shape[0]}x{D.shape[1]} at bin centers; "
          f"range [{D[D > 0].min() if n_zero < D.size else 0:.3e}, {D.max():.3e}]"
          + (f"; {n_bad} non-finite -> 0" if n_bad else "")
          + (f"; {n_zero} zero cells (w_ratio = 0 for events whose "
             f"denominator lands there)" if n_zero else ""))
    if D.max() <= 0:
        raise SystemExit("dsigma/dt is zero everywhere on the grid.")

    name = args.name or ("dgen_s_t" if args.from_gen else
                         "log_dsdt_s_t" if args.log else "dsdt_s_t")
    stored = np.where(D > 0, np.log(np.where(D > 0, D, 1.0)), LOG_FLOOR) if args.log else D
    title = ("ln d#sigma/dt(s,t)" if args.log else "d#sigma/dt(s,t)") + \
            ";s [GeV^{2}];t [GeV^{2}]"

    if args.gen:
        report_gen(args.gen, args.tree, model, stored, s_edges, t_edges, args.log)

    write_th2(stored, name, title, s_edges, t_edges, args.out)
    if args.from_gen:
        print(f"[write] {args.out}:{name}  (D_gen = dN/ds dt, x = s, y = t)")
        print("\nAdd to the generator's input card (next to ratio_weight / _formula):")
        print(f"    ratio_weight_gen: {args.out} {name}")
    else:
        print(f"[write] {args.out}:{name}  ({'ln ' if args.log else ''}dsigma/dt, "
              f"x = s, y = t)")
        print("\nAdd to the generator's input card:")
        print(f"    ratio_weight: {args.out} {name}")
        print(f"    ratio_weight_mode: {'log' if args.log else 'linear'}")
        print("    ratio_weight_sidecar: w_resc.txt       # optional, next to the LUND")


if __name__ == "__main__":
    main()
