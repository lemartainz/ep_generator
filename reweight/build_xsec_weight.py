"""
Build the generator's weight surface from a CROSS SECTION, standalone.

This script never runs the event generator. It evaluates a cross section
you supply on a (Q2, E') grid, turns it into an accept-reject probability,
and writes it as a TH2D. The generator picks that file up through one line
in input.txt:

    weight_func: weight_func.root w_Q2_Ep

and evaluates it with TH2::Interpolate -- bilinear interpolation between
bin centers, i.e. a continuous w(Q2, E') -- keeping each sampled electron
with probability w. So the "interpolated function" handed to the generator
is exactly the TH2D this script writes; nothing in runEventGenerator.cpp
needs to change.

Why w is proportional to the cross section
------------------------------------------
The generator draws Q2 ~ U(Q2_range) and E' ~ U(E_range) independently and
derives theta from them (cos(theta) = 1 - Q2/(2 E E')), so its proposal
density g(Q2, E') is FLAT. Rejection sampling with probability

        w(Q2, E') = (1/C) * d(Q2, E') / g(Q2, E'),   C = max(d/g)

turns the proposal into the target d. With g flat, w is proportional to
d = dsigma/dQ2 dE' itself: evaluate the cross section, divide by its own
maximum, done. Pass --mc <lund> if the proposal is NOT flat (for example a
run that already had a weight applied) and g is binned from that file
instead.

Cross sections quoted in some other pair of variables are converted with
--diff (see xsec.DIFF_MEASURES): dsigma/dOmega dE', dsigma/dx dQ2 and
dsigma/dW dQ2 all carry a Jacobian into dQ2 dE'.

Overall units and constants cancel in the max=1 rescale -- only the shape
of the cross section matters.

Phase space
-----------
Pass --input-card input.txt and the beam energy, Q2/E'/theta ranges, W_min
and target are read from the same card the generator uses, so the weight
grid matches the run it will steer.

Bins the generator cannot reach (no solution for theta, theta outside
theta_range, W < W_min) get special treatment through --outside, because a
1/Q^4 cross section evaluated in the unphysical corner would otherwise set
the max=1 scale and crater the keep fraction. The default, `clip`, keeps
the cross section there but caps it at the largest physical value. It is
NOT the same as zeroing: the generator re-applies the theta and W_min cuts
itself after the accept-reject, so no unreachable event is produced either
way, but TH2::Interpolate's bilinear stencil straddles the acceptance
edge -- a ring of zeros drags down the weight of genuinely physical points
next to it. On a 100x100 grid that bias is visible (a few percent, and up
to 25% in the tails of the E' projection); capping instead removes it.
`--outside zero` restores the strict behaviour, and is unbiased too once
the grid is fine enough that the boundary bins are narrow.

Usage
-----
    # analytic: a 1/Q^4-ish electroproduction shape
    python build_xsec_weight.py \\
        --input-card ../input.txt \\
        --formula "Gamma * exp(-2.0 * (W - 2.85))" \\
        --out ../weight_func.root

    # your own python function, f(Q2, Ep) or f(Q2, Ep, kin)
    python build_xsec_weight.py --input-card ../input.txt \\
        --xsec-py my_model.py:sigma --out ../weight_func.root

    # a table of measured points (CSV columns Q2,Ep,sigma)
    python build_xsec_weight.py --input-card ../input.txt \\
        --table xsec.csv --out ../weight_func.root

Then look at what you built before running the generator:

    python plot_weight.py ../weight_func.root w_Q2_Ep
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xsec                                                  # noqa: E402
from build_weight_func import (MODES, apply_ratio_guards,    # noqa: E402
                               finalize_and_write, parse_range)


def read_input_card(path):
    """Pull the generator's phase-space settings out of its own input.txt.

    Only the keys this script needs are parsed (beam_energy, Q2_range,
    E_range, theta_range, W_min, target_pid); everything else, and every
    '#' comment, is ignored. Reading the same card the generator reads is
    the point: the weight grid then covers exactly the phase space the run
    will sample.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--input-card: no such file: {path}")
    want = {"beam_energy", "W_min", "target_pid",
            "Q2_range", "E_range", "theta_range"}
    card = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            if key in want:
                card[key] = val.split()
    out = {}
    if "beam_energy" in card:
        out["beam_energy"] = float(card["beam_energy"][0])
    if "W_min" in card:
        out["W_min"] = float(card["W_min"][0])
    if "target_pid" in card:
        out["target_pid"] = int(card["target_pid"][0])
    for key, name in (("Q2_range", "q2_range"), ("E_range", "e_range"),
                      ("theta_range", "theta_range")):
        if key in card:
            v = card[key]
            if len(v) < 2:
                raise SystemExit(f"--input-card: '{key}' needs two numbers.")
            out[name] = (float(v[0]), float(v[1]))
    print(f"[card] {path}: " + "  ".join(f"{k}={v}" for k, v in out.items()))
    return out


def proposal_density(mc_path, x_edges, y_edges):
    """Bin a generator LUND into g(Q2, E') on the weight grid.

    Only needed when the proposal is not flat -- e.g. building a second,
    corrective surface on top of a run that already used a weight. Returns
    (probability-normalized density, raw counts); the counts feed the
    low-statistics guards.
    """
    from kinematics import compute_kinematics_batch
    k = compute_kinematics_batch(mc_path)
    gx = np.asarray(k["Q2"], dtype=float)
    gy = np.asarray(k["Ep"], dtype=float)
    finite = np.isfinite(gx) & np.isfinite(gy)
    counts, _, _ = np.histogram2d(gx[finite], gy[finite],
                                  bins=[x_edges, y_edges])
    if counts.sum() <= 0:
        raise SystemExit(f"--mc {mc_path}: no events inside the weight grid.")
    print(f"[mc] {mc_path}: {int(counts.sum())} events inside the grid "
          "(non-flat proposal g)")
    return counts / counts.sum(), counts


def evaluate_xsec(args, kin):
    """Dispatch to the requested cross-section source -> raw sigma array."""
    if args.formula:
        sigma = xsec.eval_formula(args.formula, kin)
        print(f"[xsec] formula: {args.formula}")
    elif args.xsec_py:
        fn = xsec.load_callable(args.xsec_py)
        sigma = xsec.eval_callable(fn, kin)
        print(f"[xsec] callable: {args.xsec_py}")
    else:
        cols = tuple(c.strip() for c in args.table_cols.split(","))
        if len(cols) != 3:
            raise SystemExit("--table-cols wants three names: 'x,y,sigma'")
        sigma = xsec.load_table(args.table, cols, kin, fill=args.table_fill)
    return sigma


def grid_centers(x_edges, y_edges, supersample):
    """Points at which to evaluate sigma, plus the axis over which to average.

    supersample == 1 gives the bin centers, which is what TH2::Interpolate
    reads back, so it is the faithful default. supersample = n evaluates an
    n x n sub-grid inside every bin and averages -- worth it when sigma
    varies strongly across one bin (a steep 1/Q^4 rise near the low-Q2
    edge), at the cost of no longer being exactly the center value.
    """
    n = max(1, int(supersample))
    # offsets of n equally spaced sub-points inside a unit bin
    frac = (np.arange(n) + 0.5) / n
    xw = np.diff(x_edges)[:, None] * (frac[None, :] - 0.5)
    yw = np.diff(y_edges)[:, None] * (frac[None, :] - 0.5)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])[:, None] + xw     # [nx, n]
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])[:, None] + yw     # [ny, n]
    # [nx, n, ny, n] meshgrid, collapsed to [nx*n, ny*n]
    X = np.repeat(xc.ravel()[:, None], yc.size, axis=1)
    Y = np.repeat(yc.ravel()[None, :], xc.size, axis=0)
    return X, Y, n


def collapse(vals, nx, ny, n):
    """Average an [nx*n, ny*n] sub-grid array back down to [nx, ny]."""
    if n == 1:
        return vals
    return vals.reshape(nx, n, ny, n).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(
        description="Build a generator weight surface from a cross section.")

    src = ap.add_argument_group("cross section (exactly one)")
    src.add_argument("--formula", default=None,
                     help="numpy expression in Q2, Ep, nu, x, y, W, theta, "
                          "eps, Gamma, Mott, ... e.g. \"Gamma * exp(-2*W)\". "
                          "Overall constants are irrelevant.")
    src.add_argument("--xsec-py", default=None,
                     help="'path/to/file.py:func'; func(Q2, Ep) or "
                          "func(Q2, Ep, kin) returning an array.")
    src.add_argument("--table", default=None,
                     help="CSV/npz of tabulated cross-section points, "
                          "linearly interpolated onto the grid.")
    src.add_argument("--table-cols", default="Q2,Ep,sigma",
                     help="column/key names in --table (default Q2,Ep,sigma)")
    src.add_argument("--table-fill", type=float, default=0.0,
                     help="value for grid points outside the --table hull "
                          "(default 0: the generator just never goes there)")
    src.add_argument("--diff", default="Q2Ep", choices=xsec.DIFF_MEASURES,
                     help="what the cross section is differential in; the "
                          "Jacobian into dsigma/dQ2 dE' is applied for you. "
                          "Default Q2Ep (no Jacobian).")

    ps = ap.add_argument_group("phase space")
    ps.add_argument("--input-card", default=None,
                    help="the generator's input.txt: beam_energy, Q2_range, "
                         "E_range, theta_range, W_min and target_pid are read "
                         "from it so the surface matches the run it steers.")
    ps.add_argument("--beam-energy", type=float, default=None)
    ps.add_argument("--target-pid", type=int, default=None)
    ps.add_argument("--target-mass", type=float, default=None,
                    help="overrides --target-pid")
    ps.add_argument("--w-min", type=float, default=None)
    ps.add_argument("--theta-range", default=None,
                    help='"lo,hi" in degrees (generator units)')
    ps.add_argument("--x-range", default=None,
                    help='"lo,hi" Q2 grid range (default: the card Q2_range)')
    ps.add_argument("--y-range", default=None,
                    help="\"lo,hi\" E' grid range (default: the card E_range)")
    ps.add_argument("--nx", type=int, default=100, help="Q2 bins (default 100)")
    ps.add_argument("--ny", type=int, default=100, help="E' bins (default 100)")
    ps.add_argument("--supersample", type=int, default=1,
                    help="evaluate sigma on an NxN sub-grid per bin and "
                         "average, instead of at the bin center (default 1). "
                         "Use 3-5 for a cross section that varies fast "
                         "within one bin.")
    ps.add_argument("--outside", default="clip",
                    choices=("clip", "zero", "keep"),
                    help="what to do with bins the generator cannot reach "
                         "(no solution for theta, theta outside theta_range, "
                         "W < W_min). clip (default): keep the cross section "
                         "but cap it at the physical maximum; zero: set w=0; "
                         "keep: leave it alone. See the note in the module "
                         "docstring -- the generator re-applies both cuts "
                         "itself, so this only changes how the surface "
                         "interpolates NEAR the acceptance edge.")

    out = ap.add_argument_group("output and guards")
    out.add_argument("--out", required=True, help="output ROOT file")
    out.add_argument("--name", default="w_Q2_Ep",
                    help="TH2D name; must match the second field of the "
                         "'weight_func:' line in input.txt (default w_Q2_Ep)")
    out.add_argument("--mc", default=None,
                     help="LUND of a generator run to bin as the proposal g. "
                          "Omit when the generator samples Q2/E' uniformly "
                          "(the normal case) -- g is then flat.")
    out.add_argument("--min-rec", type=float, default=0.0,
                     help="with --mc, zero bins whose proposal holds fewer "
                          "than this many events (default 0 = off; inert for "
                          "a flat g, which has no sampling error)")
    out.add_argument("--wmax", type=float, default=None,
                     help="cap the raw sigma/g ratio before the max=1 rescale."
                          " Tames a divergence (1/Q^4 at low Q2) that would "
                          "otherwise leave every other bin at w << 1.")
    out.add_argument("--wclip-pct", type=float, default=None,
                     help="same, but cap at this percentile of the nonzero "
                          "bins (e.g. 99). Ignored when --wmax is set.")
    out.add_argument("--prev", default=None,
                     help="previous cumulative weight ROOT file (same name and "
                          "binning); the new surface is multiplied into it.")
    out.add_argument("--archive", default=None,
                     help="directory for a versioned copy w_xsec_iter<N>.root")

    args = ap.parse_args()
    args.mode = "xsec"                     # archive naming in finalize_and_write

    n_src = sum(bool(s) for s in (args.formula, args.xsec_py, args.table))
    if n_src != 1:
        raise SystemExit("give exactly one of --formula / --xsec-py / --table.")

    # ---- phase space: card first, explicit flags win -------------------
    card = read_input_card(args.input_card) if args.input_card else {}
    Ebeam = args.beam_energy if args.beam_energy is not None \
        else card.get("beam_energy")
    if Ebeam is None:
        raise SystemExit("need a beam energy: --input-card or --beam-energy.")

    if args.target_mass is not None:
        Mt = args.target_mass
    else:
        pid = args.target_pid if args.target_pid is not None \
            else card.get("target_pid", 2212)
        Mt = xsec.target_mass(pid)

    W_min = args.w_min if args.w_min is not None else card.get("W_min", 0.0)
    if args.theta_range:
        th_lo, th_hi = parse_range(args.theta_range)
    else:
        th_lo, th_hi = card.get("theta_range", (0.0, 180.0))

    x_spec = args.x_range
    y_spec = args.y_range
    x_lo, x_hi = parse_range(x_spec) if x_spec else card.get("q2_range", (None, None))
    y_lo, y_hi = parse_range(y_spec) if y_spec else card.get("e_range", (None, None))
    if x_lo is None or y_lo is None:
        raise SystemExit("need a grid: --x-range/--y-range or --input-card "
                         "(Q2_range / E_range).")

    # The generator drops anything at or outside the TH2 edges (Interpolate
    # returns 0 on the boundary), so a grid narrower than the sampled range
    # is a silent extra cut.
    for lbl, grid, samp in (("Q2", (x_lo, x_hi), card.get("q2_range")),
                            ("E'", (y_lo, y_hi), card.get("e_range"))):
        if samp and (grid[0] > samp[0] or grid[1] < samp[1]):
            print(f"[warn] {lbl} grid [{grid[0]:g},{grid[1]:g}] is inside the "
                  f"sampled range [{samp[0]:g},{samp[1]:g}]: the generator "
                  "will reject everything outside the grid.")

    # A Q2 or E' edge at exactly 0 makes 1/Q2 and E'/E blow up on the first
    # bin center; nudge off zero rather than emitting inf.
    if x_lo <= 0.0:
        x_lo = max(x_lo, 1e-4)
        print(f"[grid] Q2 low edge moved off zero -> {x_lo:g} "
              "(1/Q2 factors are singular at Q2=0).")
    if y_lo <= 0.0:
        y_lo = max(y_lo, 1e-4)
        print(f"[grid] E' low edge moved off zero -> {y_lo:g}.")

    x_edges = np.linspace(x_lo, x_hi, args.nx + 1)
    y_edges = np.linspace(y_lo, y_hi, args.ny + 1)
    print(f"[grid] {args.nx}x{args.ny}  Q2=[{x_lo:g},{x_hi:g}]  "
          f"E'=[{y_lo:g},{y_hi:g}]  E={Ebeam:g}  M={Mt:.4f}  "
          f"W_min={W_min:g}  theta=[{th_lo:g},{th_hi:g}] deg")

    # ---- evaluate the cross section ------------------------------------
    X, Y, nsub = grid_centers(x_edges, y_edges, args.supersample)
    kin = xsec.kinematics(X, Y, Ebeam, Mt)

    sigma = evaluate_xsec(args, kin)
    sigma = sigma * xsec.jacobian(args.diff, kin)
    if args.diff != "Q2Ep":
        print(f"[xsec] --diff {args.diff}: Jacobian applied -> dsigma/dQ2 dE'")

    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 0.0, None)      # a density cannot be negative

    phys = xsec.physical_mask(kin, th_lo, th_hi, W_min, Ebeam)
    if not phys.any():
        raise SystemExit("no point on the grid is reachable: check W_min, "
                         "theta_range and the beam energy against the "
                         "Q2 / E' ranges.")
    if args.outside == "zero":
        sigma = np.where(phys, sigma, 0.0)
    elif args.outside == "clip":
        # Cap the unreachable bins at the largest PHYSICAL value. They then
        # cannot set the max=1 scale (a 1/Q^4 spike in the unphysical corner
        # would), but they also do not drag their physical neighbours toward
        # zero through TH2::Interpolate's bilinear stencil. Nothing outside
        # the acceptance is produced either way: the generator applies the
        # same theta and W_min cuts after the accept-reject.
        cap = float(sigma[phys].max())
        sigma = np.where(phys, sigma, np.minimum(sigma, cap))
    print(f"[mask] outside={args.outside}: "
          f"{float(np.mean(~phys)):.3f} of the grid is unreachable")

    D = collapse(sigma, args.nx, args.ny, nsub)
    phys_frac = collapse(phys.astype(float), args.nx, args.ny, nsub)
    if D.max() <= 0:
        raise SystemExit(
            "Cross section is zero everywhere on the physical grid. Check "
            "--formula / --table against the Q2 and E' ranges, and that "
            "W_min and theta_range leave any phase space at all.")
    print(f"[xsec] evaluated on {args.nx*nsub}x{args.ny*nsub} points "
          f"(supersample={nsub}); physical fraction="
          f"{float(np.mean(phys_frac > 0)):.3f}")

    # ---- w = sigma / g --------------------------------------------------
    D = D / D.sum()
    if args.mc:
        G, G_counts = proposal_density(args.mc, x_edges, y_edges)
        W = np.divide(D, G, out=np.zeros_like(D), where=G > 0.0)
        # Only reachable bins count as holes: with --outside clip the
        # unphysical ones carry a nonzero cross section too, and they are
        # supposed to be empty in the proposal.
        n_holes = int(np.sum((G_counts <= 0) & (D > 0) & (phys_frac > 0)))
        if n_holes:
            print(f"[mc] {n_holes} reachable bins have cross section but no "
                  "proposal events -> w=0. Throw more events into --mc, or "
                  "coarsen --nx/--ny.")
    else:
        # Flat proposal: every bin equally probable, so w is the cross
        # section itself. Infinite "counts" -> the --min-rec floor is inert,
        # since an analytic g has no sampling error.
        print("[prop] flat proposal g (generator samples Q2 and E' "
              "uniformly): w is proportional to the cross section.")
        G_counts = np.full_like(D, np.inf)
        W = D * (args.nx * args.ny)

    W = apply_ratio_guards(W, G_counts, args)

    # Keep fraction the generator will see. Two different numbers matter:
    # the accept rate for a sample that already passed the theta / W_min
    # cuts, and the overall yield per uniform (Q2, E') draw -- which also
    # pays the reachable fraction of the grid. Reported before
    # finalize_and_write does the same max=1 rescale.
    w_norm = W / W.max()
    denom = phys_frac.sum()
    eff_phys = float((w_norm * phys_frac).sum() / denom) if denom else 0.0
    eff_total = float((w_norm * phys_frac).mean())
    print(f"[eff] accept probability over the reachable region = "
          f"{eff_phys:.4f}; per uniform (Q2,E') draw = {eff_total:.4f} "
          f"(~1 kept event per "
          f"{(1.0/eff_total if eff_total > 0 else float('inf')):.1f} samples)")
    if eff_total < 0.01:
        capped = args.wmax is not None or args.wclip_pct is not None
        advice = ("tighten the cap (a lower --wclip-pct, or an explicit "
                  "--wmax)" if capped else
                  "--wclip-pct 99 (or --wmax) caps the few extreme bins that "
                  "set the scale")
        print(f"[eff] that is a slow run. Steep cross sections spend their "
              f"whole budget on the peak: {advice}, or narrow --x-range / "
              "--y-range to the region you actually want.")

    cfg = dict(MODES["q2ep"])
    cfg["title"] = "w(Q^{2},E') from cross section;Q^{2} [GeV^{2}];E' [GeV]"
    finalize_and_write(W, args.name, cfg, args, x_edges, y_edges,
                       label=f"(cross section, diff={args.diff})")

    print(f"\nAdd this line to the generator's input card:\n"
          f"    weight_func: {args.out} {args.name}")


if __name__ == "__main__":
    main()
