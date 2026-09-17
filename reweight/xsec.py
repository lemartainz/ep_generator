"""
Cross-section models for the standalone weight builder.

This module knows nothing about ROOT or about the generator; it turns a
user-supplied cross section into an array of dsigma/dQ2 dE' values on a
grid of (Q2, E') points. `build_xsec_weight.py` does the rest (divide by
the proposal density, normalize, write the TH2D).

Three ways to specify the cross section
---------------------------------------
    analytic   -- a numpy expression in the kinematic variables below
                  (`eval_formula`)
    callable   -- a Python function in your own file (`load_callable`)
    tabulated  -- a CSV/npz of (Q2, E', sigma) points (`load_table`)

Kinematic variables available to a formula (all numpy arrays on the grid,
except the scalars E, M):

    Q2        four-momentum transfer [GeV^2]
    Ep        scattered-electron energy E'  [GeV]
    E         beam energy [GeV]              (scalar)
    M         target mass [GeV]              (scalar)
    nu        E - E'  [GeV]
    y         nu / E
    x         Bjorken x = Q2 / (2 M nu)
    W         hadronic invariant mass [GeV]
    W2        W^2 [GeV^2]
    theta     scattered-electron polar angle [rad]
    theta_deg the same in degrees
    tan2_half tan^2(theta/2)
    eps       virtual-photon transverse polarization
    Gamma     virtual-photon flux, Hand convention  (alias Gamma_H)
    Gamma_G   virtual-photon flux, Gilman convention
    Mott      Mott cross section dsigma/dOmega [GeV^-2]

theta is not independent: the generator fixes it from (Q2, E') through
cos(theta) = 1 - Q2 / (2 E E'), so the pair (Q2, E') determines the whole
electron kinematics. Points where that relation has no solution get
`phys = False` from `physical_mask` and are dropped.

UNITS DO NOT MATTER. The weight is renormalized to max = 1 before the
generator ever sees it, so any overall constant (nb, GeV^-2, arbitrary)
cancels. Only the SHAPE of the cross section over the grid is used.
"""

import importlib.util
import inspect
import os

import numpy as np

# Keep in sync with the PDG:: constants in runEventGenerator.cpp.
ALPHA = 1.0 / 137.035999084
M_P = 0.9382720813
M_N = 0.9395654133
M_E = 0.0005109989461

TARGET_MASS = {2212: M_P, 2112: M_N}


def target_mass(pid):
    """Target mass [GeV] for a PDG code, matching the generator's getMass()."""
    if pid not in TARGET_MASS:
        raise SystemExit(f"no target mass known for PDG {pid}; "
                         "pass --target-mass explicitly.")
    return TARGET_MASS[pid]


def kinematics(Q2, Ep, Ebeam, Mt):
    """Every kinematic variable a cross-section formula may reference.

    Q2 and Ep are arrays of the same shape (typically a meshgrid of bin
    centers); Ebeam and Mt are scalars. Unphysical points are not filtered
    here -- they come back as nan/inf and are masked by `physical_mask` --
    so that a formula can be evaluated on the full grid in one shot.
    """
    Q2 = np.asarray(Q2, dtype=float)
    Ep = np.asarray(Ep, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        nu = Ebeam - Ep
        y = nu / Ebeam
        # cos(theta) from the generator's own relation, and the half-angle
        # form that the Mott / flux factors are written in.
        cos_th = 1.0 - Q2 / (2.0 * Ebeam * Ep)
        sin2_half = Q2 / (4.0 * Ebeam * Ep)
        theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
        tan2_half = sin2_half / np.clip(1.0 - sin2_half, 1e-300, None)

        W2 = Mt * Mt + 2.0 * Mt * nu - Q2
        W = np.sqrt(np.clip(W2, 0.0, None))
        x = Q2 / (2.0 * Mt * nu)

        # Transverse polarization of the virtual photon.
        eps = 1.0 / (1.0 + 2.0 * (1.0 + nu * nu / Q2) * tan2_half)

        # Virtual-photon flux. Hand uses the equivalent real-photon energy
        # K = (W^2 - M^2) / 2M; Gilman uses K = nu. Both are standard; the
        # choice only matters if you multiply by a sigma_gamma*p that was
        # extracted with a particular convention.
        K_hand = (W2 - Mt * Mt) / (2.0 * Mt)
        pref = ALPHA / (2.0 * np.pi * np.pi) * (Ep / Ebeam) / (1.0 - eps)
        Gamma_H = pref * K_hand / Q2
        Gamma_G = pref * nu / Q2

        # Mott cross section dsigma/dOmega for a point charge [GeV^-2].
        Mott = (ALPHA * ALPHA * (1.0 - sin2_half)
                / (4.0 * Ebeam * Ebeam * sin2_half * sin2_half))

    return {
        "Q2": Q2, "Ep": Ep, "E": float(Ebeam), "M": float(Mt),
        "nu": nu, "y": y, "x": x, "xB": x,
        "W": W, "W2": W2,
        "theta": theta, "theta_deg": np.degrees(theta),
        "cos_theta": cos_th, "sin2_half": sin2_half, "tan2_half": tan2_half,
        "eps": eps, "epsilon": eps,
        "Gamma": Gamma_H, "Gamma_H": Gamma_H, "Gamma_G": Gamma_G,
        "Mott": Mott,
    }


def physical_mask(kin, theta_min_deg, theta_max_deg, W_min, Ebeam):
    """Points the generator could actually produce, as a boolean array.

    Mirrors the accept conditions in eventGenerator::generateScatteredElectron:
    a solvable cos(theta), theta inside theta_range, and W >= W_min (plus
    0 < E' <= E). Everything else is set to w = 0 so it cannot inflate the
    max=1 normalization -- a 1/Q^4 cross section evaluated in the unphysical
    corner would otherwise set the scale for the whole surface and collapse
    the keep fraction.
    """
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(kin["cos_theta"])
        ok &= np.abs(kin["cos_theta"]) <= 1.0
        ok &= (kin["Ep"] > 0.0) & (kin["Ep"] <= Ebeam)
        ok &= kin["theta_deg"] >= theta_min_deg
        ok &= kin["theta_deg"] <= theta_max_deg
        ok &= kin["W"] >= W_min
    return ok


# --------------------------------------------------------------------- #
# Differential measure -> dsigma/dQ2 dE'
# --------------------------------------------------------------------- #
# The generator samples Q2 and E' uniformly and derives theta from them, so
# the accept probability must be proportional to the density in exactly
# those two variables. A cross section quoted in any other pair needs the
# Jacobian of the change of variables (at fixed Q2, since Q2 is common to
# all of them):
#
#   Q2Ep    dsigma/dQ2 dE'     -> 1                    (already there)
#   Q2nu    dsigma/dQ2 dnu     -> |dnu/dE'| = 1        (nu = E - E')
#   OmegaEp dsigma/dOmega dE'  -> phi-integrated: dOmega = dQ2 dphi/(2 E E')
#                                 so the factor is pi/(E E')
#   xQ2     dsigma/dx dQ2      -> |dx/dE'|_Q2 = x/nu
#   WQ2     dsigma/dW dQ2      -> |dW/dE'|_Q2 = M/W
#
# OmegaEp assumes the cross section is phi-independent, which it is for an
# unpolarized beam and target.
DIFF_MEASURES = ("Q2Ep", "Q2nu", "OmegaEp", "xQ2", "WQ2")


def jacobian(measure, kin):
    """Multiplicative factor turning `measure` into dsigma/dQ2 dE'."""
    if measure in ("Q2Ep", "Q2nu"):
        return np.ones_like(kin["Q2"])
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "OmegaEp":
            return np.full_like(kin["Q2"], np.pi / kin["E"]) / kin["Ep"]
        if measure == "xQ2":
            return kin["x"] / kin["nu"]
        if measure == "WQ2":
            return kin["M"] / np.clip(kin["W"], 1e-300, None)
    raise SystemExit(f"unknown --diff {measure!r}; "
                     f"choose from {', '.join(DIFF_MEASURES)}")


# --------------------------------------------------------------------- #
# 1. analytic formula
# --------------------------------------------------------------------- #
_SAFE_FUNCS = {
    n: getattr(np, n) for n in (
        "exp", "log", "log10", "log2", "sqrt", "abs", "sign",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
        "sinh", "cosh", "tanh", "power", "minimum", "maximum", "clip",
        "where", "heaviside", "floor", "ceil", "hypot",
    )
}
_SAFE_FUNCS.update({"pi": np.pi, "e": np.e, "inf": np.inf,
                    "min": np.minimum, "max": np.maximum,
                    # so one string works both here and in ROOT's TFormula
                    "pow": np.power})


def eval_formula(expr, kin):
    """Evaluate a numpy expression in the kinematic variables.

    The namespace is restricted to `kin` plus a whitelist of numpy
    functions -- no builtins, no imports. This is a convenience guard
    against typos reaching into the interpreter, not a security sandbox:
    only run formulas you wrote.
    """
    ns = dict(_SAFE_FUNCS)
    ns.update(kin)
    try:
        val = eval(expr, {"__builtins__": {}}, ns)          # noqa: S307
    except NameError as err:
        known = ", ".join(sorted(k for k in kin if not k.startswith("_")))
        raise SystemExit(f"--formula: {err}.\nAvailable variables: {known}")
    except Exception as err:
        raise SystemExit(f"--formula failed to evaluate: {err}")
    out = np.asarray(val, dtype=float)
    if out.shape != kin["Q2"].shape:
        out = np.broadcast_to(out, kin["Q2"].shape).copy()
    return out


# --------------------------------------------------------------------- #
# 2. Python callable
# --------------------------------------------------------------------- #
def load_callable(spec):
    """Import `path/to/file.py:func` (or `module:func`) and return it.

    The function is called as f(Q2, Ep) with two numpy arrays, or
    f(Q2, Ep, kin) if it accepts a third positional argument (kin is the
    dict from `kinematics`, so you get nu/W/x/eps/Gamma for free). It must
    return an array broadcastable to Q2's shape.
    """
    if ":" not in spec:
        raise SystemExit("--xsec-py expects 'path/to/file.py:function' "
                         f"(got {spec!r})")
    path, func = spec.rsplit(":", 1)
    if path.endswith(".py"):
        if not os.path.exists(path):
            raise SystemExit(f"--xsec-py: no such file: {path}")
        mod_name = os.path.splitext(os.path.basename(path))[0]
        s = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path)
    if not hasattr(mod, func):
        raise SystemExit(f"--xsec-py: {path} has no '{func}'")
    fn = getattr(mod, func)
    if not callable(fn):
        raise SystemExit(f"--xsec-py: {path}:{func} is not callable")
    return fn


def eval_callable(fn, kin):
    try:
        npar = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        npar = 2
    val = fn(kin["Q2"], kin["Ep"], kin) if npar >= 3 else fn(kin["Q2"], kin["Ep"])
    return np.broadcast_to(np.asarray(val, dtype=float),
                           kin["Q2"].shape).astype(float)


# --------------------------------------------------------------------- #
# 3. tabulated cross section
# --------------------------------------------------------------------- #
def _read_table(path, cols):
    """Return (x, y, sigma) columns from a CSV or npz table."""
    xcol, ycol, scol = cols
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        missing = [c for c in cols if c not in z]
        if missing:
            raise SystemExit(f"--table {path}: missing key(s) {missing}; "
                             f"has {list(z.keys())}")
        return (np.asarray(z[xcol], float).ravel(),
                np.asarray(z[ycol], float).ravel(),
                np.asarray(z[scol], float).ravel())
    import csv
    rows = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        missing = [c for c in cols if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"--table {path}: missing column(s) {missing}; "
                             f"has {r.fieldnames}")
        for row in r:
            for c in cols:
                rows[c].append(float(row[c]))
    return tuple(np.asarray(rows[c], float) for c in cols)


def load_table(path, cols, kin, fill=0.0):
    """Interpolate a tabulated cross section onto the (Q2, E') grid.

    The table is a list of (Q2, E', sigma) points -- scattered or on a
    rectangular grid, either way. Values are linearly interpolated
    (scipy.interpolate.griddata); grid points outside the table's convex
    hull get `fill`, which defaults to 0 so the generator simply never
    produces them rather than extrapolating a cross section you did not
    provide.
    """
    from scipy.interpolate import griddata

    tx, ty, ts = _read_table(path, cols)
    if not (len(tx) == len(ty) == len(ts)):
        raise SystemExit(f"--table {path}: columns have different lengths.")
    if len(tx) < 3:
        raise SystemExit(f"--table {path}: need at least 3 points, got {len(tx)}.")

    pts = np.column_stack([tx, ty])
    out = griddata(pts, ts, (kin["Q2"], kin["Ep"]), method="linear")
    n_out = int(np.sum(~np.isfinite(out)))
    out = np.where(np.isfinite(out), out, fill)
    print(f"[table] {path}: {len(tx)} points, "
          f"x=[{tx.min():g},{tx.max():g}] y=[{ty.min():g},{ty.max():g}]; "
          f"{n_out} grid points outside the table -> {fill:g}")
    return out


# --------------------------------------------------------------------- #
# Binned cross sections: density vs per-bin integral
# --------------------------------------------------------------------- #
def bin_volume(edges):
    """Product of bin widths over all axes, broadcast to the full grid."""
    vol = None
    for i, e in enumerate(edges):
        w = np.diff(np.asarray(e, dtype=float))
        shape = [1] * len(edges)
        shape[i] = len(w)
        w = w.reshape(shape)
        vol = w if vol is None else vol * w
    return vol


def to_bin_yield(D, edges, units, tag="xsec"):
    """Turn a binned cross section into the expected YIELD per bin.

    The weight is a ratio of per-bin numbers, so the numerator has to be
    the number of events the cross section predicts in that bin -- its
    INTEGRAL over the bin -- not its average height. The two differ by the
    bin volume, which only matters when the binning is non-uniform, and
    then it matters a lot: Q2 edges of 1,2,3,4.5,7 span a factor 2.5, so a
    density fed in as if it were an integral under-populates the wide bins
    by that factor.

        units='integral'  already a per-bin yield (or a cross section
                          already integrated over each bin) -- used as is.
        units='density'   dsigma/dQ2 dW dM sampled in the bin -- multiplied
                          by the bin volume.

    There is no way to tell the two apart from the numbers alone, so say
    which you have. `make_pseudo_xsec.py` records 'density' in its npz.
    """
    if units not in ("integral", "density"):
        raise SystemExit(f"{tag}: units must be 'integral' or 'density', "
                         f"got {units!r}")
    if units == "integral":
        return D
    vol = bin_volume(edges)
    print(f"[{tag}] units=density: multiplied by bin volume "
          f"(spread {vol.max()/vol.min():.2f}x) to get a per-bin yield")
    return D * vol


def warn_if_ambiguous(edges, units_given, tag="xsec"):
    """Shout when the binning is non-uniform and nobody said which it is."""
    vol = bin_volume(edges)
    spread = float(vol.max() / vol.min())
    if units_given or spread < 1.001:
        return
    print(f"[warn] {tag}: bin volumes span {spread:.2f}x and the file does "
          "not say whether the values are a DENSITY (dsigma/dQ2 dW dM) or a "
          "per-bin INTEGRAL. Assuming integral. If they are a density, pass "
          "--xsec-units density -- otherwise the wide bins come out low by "
          f"up to {spread:.2f}x.")
