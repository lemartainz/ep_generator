"""
Build the generator's PAIR-MASS weights w(Q2, W, M_AB), one TH3D per species
pair, fitted jointly so that the accepted events reproduce several measured
pair-mass distributions at once.

Why not xsec_weight
-------------------
xsec_weight reshapes M_X, the mass of the intermediate from the first vertex.
In the generator that is exact (it reads the truth 4-vector). In DATA it is
not observable: e p -> e' p p pbar has two protons and nothing says which one
came from X, so the measured M(p pbar) is either one pairing (a guess) or
BOTH pairings pooled -- and the "true" M_X would have to be unfolded from the
pooled one by subtracting the wrong-pairing shape, which goes negative in
the tails.

The pair weights sidestep this. The generator evaluates each surface on
EVERY (A, B) pair in the final state and multiplies the factors (see
EventWeighter::acceptPairs). For `2212 -2212` that is two factors per event,
M(p1 pbar) and M(p2 pbar) -- exactly how the pooled histogram is filled --
and for `2212 2212` one, M(p1 p2). Both observables are symmetric under
p1 <-> p2, so the target is a directly measured, sideband-subtracted
distribution and no pairing is ever chosen. Together the pooled M(p pbar)
and M(p p) pin down the three-body Dalitz distribution in each (Q2, W) cell.

Why the surfaces have to be fitted together
-------------------------------------------
With several entries per event a per-bin ratio d/g is NOT the answer: the
two M(p pbar) entries of one event share one accept probability, and
M(p p) is kinematically tied to them. So the per-event weight

    w_e = u(Q2, W, m_1) * u(Q2, W, m_2) * v(Q2, W, m_pp)

is FITTED on the generator's own sample: the log-factors of every fitted
cell of every surface are the parameters, and

    chi2 = sum_s sum_cells (H_s - lambda_s T_s)^2 / sigma^2 + prior

is minimized (L-BFGS, analytic gradient), where H_s is the sample's
weighted histogram, T_s the target, lambda_s a per-surface scale (the
absolute rate is not the generator's business) and sigma combines the
target's error with the sample's own. The closure printed at the end is
the fit residual; the generator, drawing from the same proposal, inherits
it up to statistics.

A chi2 fit rather than iterative proportional fitting (raking) because the
two targets are measured with noise and are not guaranteed to be jointly
reachable by a product of per-pairing factors -- near the kinematic edge
they routinely are not, by a few sigma -- and raking then diverges (one
factor to infinity, its partner to zero, the product finite), which wrecks
the max = 1 normalization. The fit settles on the compromise the errors
justify. The weak prior (--prior) pins the one flat direction of the
product form, u -> c u, v -> v / c^2, at "least correction".

Truth-level vs reco-level targets
---------------------------------
    --target 2212,-2212 D.npz          truth-level: the accepted TRUTH
                                       distribution is driven onto D
    --target 2212,-2212 D.npz R.npz    reco-level: R is the RECONSTRUCTED
                                       sim of the same run as --gen, on
                                       D's grid; the truth distribution is
                                       driven onto (D / R) x (its own
                                       histogram), i.e. the per-cell data /
                                       sim correction is applied at truth
                                       level. Iterate with --prev, as
                                       build_weight_func.py does, until
                                       D / R -> 1.

D and R are the npz files the analysis notebook writes (keys 'counts',
'edges', 'varnames'; 3-D, axes Q2, W, M). Negative sideband-subtracted
bins are clipped to zero. Each target may sit on its own grid.

Workflow
--------
    1. run the generator UNWEIGHTED -> events_unweighted.lund. The LUND is
       all the proposal sample this builder needs: both p pbar pairings and
       the p p mass are symmetric in the two protons, and Q2, W come from
       the electron. (A `truth_ntuple:` .root works too and is smaller and
       faster to read; it is only REQUIRED for xsec_weight, which needs the
       truth M_X.)
    2. python build_pair_weight.py \\
           --target 2212,-2212 ../subtracted_Mppbar_pooled.npz \\
           --target 2212,2212  ../subtracted_Mpp.npz \\
           --gen ../events_unweighted.lund --out ../pair_weight.root
    3. add the two `pair_weight:` lines it prints to input.txt, rerun
       -> events_pair.lund
    4. python plot_xsec_closure.py --gen ../events_pair.lund \\
           --xsec ../subtracted_Mppbar_pooled.npz --pair 2212,-2212 \\
           --weight ../pair_weight.root --weight-name w_pair_2212_-2212_fitted \\
           --out ../pair_closure_ppbar.pdf
       (and the same for 2212,2212)

Interpolated, and fitted that way
---------------------------------
By default (--mode interp) the generator reads each surface with
TH3::Interpolate -- trilinear between bin centers, clamped into the hull
of the centers -- so the weight is continuous in Q2, W and M and the
accepted events carry no steps at the cell edges. A per-bin surface would
imprint the (Q2, W) grid on every projection. Because the surfaces are
FITTED, the interpolation is part of the model: the parameters are the
factors at the bin centers, and every entry's factor in the chi2 is the
same blend of its eight surrounding nodes the generator will form. The
closure is therefore exact for the interpolated lookup, which is not the
case for a per-bin ratio read interpolated afterwards. --mode bin fits
and reads per bin; the generator's `pair_weight_mode` must match.

What happens outside the fitted cells
-------------------------------------
With several entries per event, "no information" and "nothing there" have
to be kept apart, because a zero on ONE pairing kills the whole event --
including its other pairing, which the data's pooled histogram still
counts. So:

    target == 0 in a cell     -> factor 0: the data says nothing lives
                                 there, and rejecting the event is right
    M outside the grid, or a  -> factor 1 (fixed, never updated): the
    cell below --min-gen /       surface has nothing to say, the entry is
    --min-rec                    passed through and only the event's OTHER
                                 factors weight it. These cells are
                                 written as unfitted (a <name>_fitted mask
                                 TH3D sits next to each surface) and the
                                 share of the target they hold is printed.
    Q2 or W outside the grid  -> factor 0: outside the analysis domain,
                                 as for xsec_weight.

The pass-through factor is stored in the TH3D's under/overflow bins along
M, so the generator reads it with the same per-bin lookup. --min-gen
defaults from --target-accuracy as in build_xsec_weight3d.py. Ratio clips
are deliberately not offered: they break the closure permanently.
"""

import argparse
import os
import sys

import numpy as np
import ROOT
from array import array as _darr
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xsec                                                         # noqa: E402
from build_weight_func import _archive_copy                         # noqa: E402
from build_xsec_weight3d import read_prev_th3                       # noqa: E402
from truth_ntuple import load_truth, pair_masses, parse_pids, pair_label  # noqa: E402


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def read_prev_outside(path, name):
    """The previous surface's pass-through factor (M under/overflow bins)."""
    tf = ROOT.TFile(path)
    h = tf.Get(name)
    v = float(h.GetBinContent(1, 1, 0)) if h else 1.0
    tf.Close()
    return v


def load_hist3(path, tag, units_arg=None):
    """A 3-D npz (counts, edges[, varnames, units]) -> (bin yield, edges)."""
    if not os.path.exists(path):
        raise SystemExit(f"{tag}: file not found: {path}")
    z = np.load(path, allow_pickle=True)
    for k in ("counts", "edges"):
        if k not in z:
            raise SystemExit(f"{tag} {path}: missing '{k}' (has {list(z.keys())})")
    edges = [np.asarray(e, dtype=float) for e in z["edges"]]
    if len(edges) != 3:
        raise SystemExit(f"{tag} {path}: expected a 3-D histogram (Q2, W, M), "
                         f"got {len(edges)}-D.")
    H = np.asarray(z["counts"], dtype=float)
    if H.shape != tuple(len(e) - 1 for e in edges):
        raise SystemExit(f"{tag} {path}: counts shape {H.shape} does not match "
                         f"edges {[len(e)-1 for e in edges]}.")
    names = [str(v) for v in z["varnames"]] if "varnames" in z else ["Q2", "W", "M"]
    n_neg = int((H < 0).sum())
    H = np.clip(H, 0.0, None)
    # per-bin error: the notebook's sideband propagation if present, else
    # Poisson on the (clipped) counts
    if "errors" in z and np.shape(z["errors"]) == H.shape:
        E = np.abs(np.asarray(z["errors"], dtype=float))
        err_src = "file"
    else:
        E = np.sqrt(H)
        err_src = "sqrt(N)"
    units = units_arg or (str(z["units"]) if "units" in z else None)
    print(f"[{tag}] {path}: vars={names} grid="
          f"{'x'.join(str(len(e)-1) for e in edges)} total={H.sum():.0f}"
          + (f" (clipped {n_neg} negative bins)" if n_neg else "")
          + f" errors={err_src}" + (f" units={units}" if units else ""))
    return H, E, edges, units


def cell_index(x, edges):
    """Bin index of x on `edges`, -1 outside. Mirrors Surface::inRange (strict
    at both ends) plus TAxis::FindBin."""
    i = np.searchsorted(edges, x, side="right") - 1
    ok = (x > edges[0]) & (x < edges[-1])
    return np.where(ok, i, -1)


def interp_nodes_1d(x, edges):
    """Lower node index and fraction for linear interpolation between bin
    CENTERS, clamped into the hull -- what TH3::Interpolate does per axis
    once the generator has clamped the point (PairSurface::lookup)."""
    c = 0.5 * (edges[:-1] + edges[1:])
    n = len(c)
    if n == 1:
        return np.zeros(len(x), int), np.zeros(len(x))
    xc = np.clip(x, c[0], c[-1])
    j = np.clip(np.searchsorted(c, xc, side="right") - 1, 0, n - 2)
    t = np.clip((xc - c[j]) / (c[j + 1] - c[j]), 0.0, 1.0)
    return j, t


def interp_nodes_3d(q2, w, m, edges, shape):
    """(n, 8) flat node indices and (n, 8) trilinear weights."""
    idx, frac = zip(*(interp_nodes_1d(v, e) for v, e in zip((q2, w, m), edges)))
    nodes, lam = [], []
    for dq in (0, 1):
        for dw in (0, 1):
            for dm in (0, 1):
                i = np.minimum(idx[0] + dq, shape[0] - 1)
                j = np.minimum(idx[1] + dw, shape[1] - 1)
                k = np.minimum(idx[2] + dm, shape[2] - 1)
                nodes.append(np.ravel_multi_index((i, j, k), shape))
                lam.append((frac[0] if dq else 1 - frac[0])
                           * (frac[1] if dw else 1 - frac[1])
                           * (frac[2] if dm else 1 - frac[2]))
    return np.stack(nodes, axis=1), np.stack(lam, axis=1)


class Target:
    """One pair-mass target: its grid, its data, and the sample's entries."""

    def __init__(self, spec, gen, units_arg, min_gen, min_rec, interp=True):
        # spec = [pids, D.npz, (R.npz), (name)]
        if len(spec) < 2:
            raise SystemExit("--target needs at least '<pidA,pidB> <D.npz>'")
        self.pid_a, self.pid_b = parse_pids(spec[0])
        self.label = pair_label(self.pid_a, self.pid_b)
        self.name = f"w_pair_{self.pid_a}_{self.pid_b}"
        d_path, r_path = spec[1], None
        for tok in spec[2:]:
            if tok.endswith(".npz"):
                r_path = tok
            else:
                self.name = tok

        tag = f"target {self.label}"
        D, E, self.edges, units = load_hist3(d_path, tag, units_arg)
        # A measured histogram is per-bin COUNTS (an integral); only a
        # cross-section-like density needs the bin volume folded in.
        self.D = xsec.to_bin_yield(D, self.edges, units or "integral", tag)
        self.E = xsec.to_bin_yield(E, self.edges, units or "integral", tag)
        self.shape = self.D.shape
        self.ncell = self.D.size

        self.R = None
        if r_path:
            R, _, r_edges, _ = load_hist3(r_path, f"rec {self.label}")
            if any(not np.allclose(a, b) for a, b in zip(r_edges, self.edges)):
                raise SystemExit(f"rec {r_path}: bin edges differ from the "
                                 f"target {d_path}; D and R must share a grid.")
            self.R = R

        # --- the sample's entries on this grid ---
        # Every entry gets an index into the EXTENDED factor array
        # [cells..., OUT_M, OUT_QW]: in-grid entries point at their cell,
        # M-outside entries at the fixed pass-through factor, Q2/W-outside
        # entries at the fixed zero.
        ev, m, per_event = pair_masses(gen, self.pid_a, self.pid_b)
        if len(ev) == 0:
            raise SystemExit(f"no ({self.pid_a}, {self.pid_b}) pairs in the "
                             "generated final state -- wrong reaction?")
        self.k = int(per_event[per_event > 0].max())
        n_ev = len(gen["Q2"])
        iq = cell_index(np.asarray(gen["Q2"], float)[ev], self.edges[0])
        iw = cell_index(np.asarray(gen["W"], float)[ev], self.edges[1])
        im = cell_index(m, self.edges[2])
        qw_in = (iq >= 0) & (iw >= 0)
        inside = qw_in & (im >= 0)
        self.OUT_M, self.OUT_QW = self.ncell, self.ncell + 1
        flat = np.ravel_multi_index(
            (np.clip(iq, 0, None), np.clip(iw, 0, None), np.clip(im, 0, None)),
            self.shape)
        flat = np.where(inside, flat, np.where(qw_in, self.OUT_M, self.OUT_QW))
        self.ev, self.flat, self.inside = ev, flat, inside
        self.has_pair = per_event > 0
        self.interp = interp

        # The lookup each entry's factor is formed from: its own cell (bin
        # mode), or the eight bin-center nodes around it with trilinear
        # weights (interp mode) -- the same blend TH3::Interpolate returns
        # after the generator clamps into the hull. Out-of-grid entries
        # keep a single node at their fixed extended slot.
        if interp:
            nodes, lam = interp_nodes_3d(np.asarray(gen["Q2"], float)[ev],
                                         np.asarray(gen["W"], float)[ev], m,
                                         self.edges, self.shape)
        else:
            nodes, lam = flat[:, None].copy(), np.ones((len(ev), 1))
        out = ~inside
        nodes[out] = flat[out, None]
        lam[out] = 0.0
        lam[out, 0] = 1.0
        self.nodes, self.lam = nodes, lam
        n_out_m = int((qw_in & ~inside).sum())
        n_out_qw = int((~qw_in).sum())
        print(f"[gen] {self.label}: {len(ev)} entries from {int(self.has_pair.sum())} "
              f"of {n_ev} events ({self.k} per event); {n_out_m} entries "
              f"({100.0*n_out_m/len(ev):.1f}%) have M outside the grid -> passed "
              f"through unweighted; {n_out_qw} ({100.0*n_out_qw/len(ev):.1f}%) "
              "have Q2 or W outside -> event rejected")

        # raw sample counts per cell (unweighted): the denominator statistics
        self.G0 = np.bincount(flat[inside], minlength=self.ncell).astype(float)
        self.G0 = self.G0.reshape(self.shape)

        # --- cell classes: zero (data empty), fitted, or free (pass-through)
        self.zero = self.D <= 0
        fitted = ~self.zero
        n_lowgen = int((fitted & (self.G0 < min_gen)).sum())
        fitted &= self.G0 >= min_gen
        n_lowrec = 0
        if self.R is not None:
            n_lowrec = int((fitted & (self.R < min_rec)).sum())
            fitted &= self.R >= min_rec
        self.fitted = fitted
        free = ~self.zero & ~fitted
        print(f"[cells] {self.label}: {int(fitted.sum())} fitted, "
              f"{int(free.sum())} pass-through ({n_lowgen} below --min-gen {min_gen:g}"
              + (f", {n_lowrec} below --min-rec {min_rec:g}" if self.R is not None else "")
              + f"), {int(self.zero.sum())} empty in the data, of {self.ncell}; "
              f"{100*self.D[free].sum()/self.D.sum():.1f}% of the target sits in "
              "pass-through cells (delivered uncorrected)")

        # per-cell target for the fit and its error: D itself, or the
        # per-cell data / sim correction applied to the sample's own
        # histogram, (D / R) x G0, with the three Poisson terms combined.
        if self.R is None:
            self.T = np.where(fitted, self.D, 0.0)
            self.T_err = np.where(fitted, self.E, 0.0)
        else:
            ratio = np.divide(self.D, self.R, out=np.zeros_like(self.D),
                              where=self.R > 0)
            self.T = np.where(fitted, ratio * self.G0, 0.0)
            rel2 = np.zeros_like(self.D)
            rel2[fitted] = ((self.E[fitted] / self.D[fitted]) ** 2
                            + 1.0 / self.R[fitted] + 1.0 / self.G0[fitted])
            self.T_err = self.T * np.sqrt(rel2)
            print(f"[rec] {self.label}: reco-level target = (D / R) x truth; "
                  f"D/R over fitted cells: median {np.median(ratio[fitted]):.3f}, "
                  f"range [{ratio[fitted].min():.3f}, {ratio[fitted].max():.3f}]")

        # extended factors: 0 where the data is empty or Q2/W is outside,
        # 1 everywhere else; the fitted cells are what the fit moves
        self.u = np.concatenate([np.where(self.zero, 0.0, 1.0).ravel(), [1.0, 0.0]])
        self.fit_cells = np.nonzero(fitted.ravel())[0]
        self.param_of_cell = np.full(self.ncell + 2, -1, dtype=int)
        self.param_of_cell[self.fit_cells] = np.arange(len(self.fit_cells))

    def factors(self):
        """Per-entry factor: the blend of its nodes' current values."""
        return (self.lam * self.u[self.nodes]).sum(axis=1)

    def dead_events(self, n_ev):
        """Events this surface vetoes outright: no pair, or an entry whose
        factor is zero -- data-empty cells (all its nodes, in interp mode)
        or outside the Q2/W domain. Fixed for the whole fit, since fitted
        nodes stay positive."""
        dead = ~self.has_pair.copy()
        bad = self.factors() <= 0.0
        np.logical_or.at(dead, self.ev, bad)
        return dead

    def set_live(self, alive):
        """Restrict the entry lists to live events (all their factors > 0),
        so the fit can work in log space."""
        keep = alive[self.ev]
        self.ev, self.flat, self.inside = self.ev[keep], self.flat[keep], self.inside[keep]
        self.nodes, self.lam = self.nodes[keep], self.lam[keep]
        # entries whose CELL is fitted (they enter the chi2 through H) ...
        self.in_fit = self.param_of_cell[self.flat] >= 0
        # ... and (entry, node) pairs whose NODE is a parameter (they carry
        # the gradient -- an entry in a pass-through cell still depends on
        # the fitted nodes it interpolates from)
        self.node_param = self.param_of_cell[self.nodes]
        self.node_sel = self.node_param >= 0

    def log_factors(self):
        return np.log(self.factors())

    def grad_accumulate(self, wG, n_par):
        """sum over entries of w_e G_e * d log f_i / d theta_c, per parameter:
        d log f / d theta_c = lam_c u_c / f."""
        coef = (wG[self.ev] / self.factors())[:, None] * self.lam
        sel = self.node_sel
        g = np.bincount(self.node_param[sel], weights=coef[sel], minlength=n_par)
        return g * self.u[self.fit_cells]

    def hist(self, w):
        """Current weighted histogram of the sample's in-grid entries."""
        sel = self.inside
        H = np.bincount(self.flat[sel], weights=w[self.ev[sel]], minlength=self.ncell)
        return H.reshape(self.shape)

    def grid(self):
        """The in-grid factors as a [nq, nw, nm] array."""
        return self.u[:self.ncell].reshape(self.shape)


# ---------------------------------------------------------------------------
# raking
# ---------------------------------------------------------------------------
def event_weights(targets, n_ev, alive):
    """w_e = prod of every factor of every entry, in log space (live only)."""
    logw = np.zeros(n_ev)
    for t in targets:
        logw += np.bincount(t.ev, weights=t.log_factors(), minlength=n_ev)
    return np.where(alive, np.exp(logw), 0.0)


class Fit:
    """chi2 of the weighted sample against every target, with gradient in
    the log-factors of the fitted cells, for scipy's L-BFGS-B.

    Per surface s, with H_s the weighted histogram over the fitted cells,
    lambda_s = sum H / sum T (the absolute rate is not the generator's
    business, so the NORMALIZED histogram h = H / lambda is what is fitted)
    and a relative variance rv = T_err^2 + sum(w^2) / lambda^2 (target error
    plus the sample's own, the latter held fixed between outer rounds):

        chi2 = sum_cells (h - T)^2 / rv  +  prior * sum theta^2

    This is invariant under a common rescaling of all the weights, so the
    fit cannot lower chi2 by drifting along the flat direction of the
    product form; only the prior sees that direction. Gradient:

        d chi2 / d theta_{s,c} = sum over entries i of s of  w_e * G_e * lam_ic u_c / f_i,
        (lam_ic the entry's interpolation weight on node c, f_i its factor;
         in bin mode lam is 1 on the entry's own cell)
        G_e = sum over the event's fitted entries of (g_k - gbar) / lambda,
        g_k = 2 (h_k - T_k) / rv_k,   gbar = sum_k g_k h_k / sum_k T_k,

    the gbar term being lambda's own dependence on the weights.
    """

    def __init__(self, targets, n_ev, alive, prior):
        self.targets, self.n_ev, self.alive, self.prior = targets, n_ev, alive, prior
        self.offsets = np.cumsum([0] + [len(t.fit_cells) for t in targets])
        self.n_par = int(self.offsets[-1])
        self.rel_var = [None] * len(targets)

    def theta(self):
        return np.concatenate([np.log(t.u[t.fit_cells]) for t in self.targets])

    def set_theta(self, theta):
        for i, t in enumerate(self.targets):
            t.u[t.fit_cells] = np.exp(theta[self.offsets[i]:self.offsets[i + 1]])

    def update_sigma(self):
        """Re-estimate the sample's own variance term at the current weights."""
        w = event_weights(self.targets, self.n_ev, self.alive)
        for i, t in enumerate(self.targets):
            var_w = np.bincount(t.flat[t.inside], weights=w[t.ev[t.inside]] ** 2,
                                minlength=t.ncell).reshape(t.shape)
            H = t.hist(w)
            f = t.fitted
            lam = H[f].sum() / t.T[f].sum()
            rv = np.where(f, t.T_err ** 2 + var_w / lam ** 2, 1.0)
            self.rel_var[i] = np.maximum(rv, 1e-12 * t.T.max() ** 2 + 1e-30)

    def normalized(self, t, w):
        H = t.hist(w)
        f = t.fitted
        lam = H[f].sum() / t.T[f].sum()
        return H / lam, lam

    def __call__(self, theta):
        self.set_theta(theta)
        w = event_weights(self.targets, self.n_ev, self.alive)
        chi2 = self.prior * float(theta @ theta)
        G = np.zeros(self.n_ev)
        for i, t in enumerate(self.targets):
            h, lam = self.normalized(t, w)
            f = t.fitted
            rv = self.rel_var[i]
            res = np.where(f, h - t.T, 0.0)
            chi2 += float((res[f] ** 2 / rv[f]).sum())
            g = np.where(f, 2.0 * res / rv, 0.0)
            gbar = (g[f] * h[f]).sum() / t.T[f].sum()
            g = np.where(f, (g - gbar) / lam, 0.0)
            sel = t.in_fit
            G += np.bincount(t.ev[sel], weights=g.ravel()[t.flat[sel]], minlength=self.n_ev)
        grad = 2.0 * self.prior * theta
        wG = w * G
        for i, t in enumerate(self.targets):
            grad[self.offsets[i]:self.offsets[i + 1]] += t.grad_accumulate(wG, len(t.fit_cells))
        return chi2, grad


def fit_surfaces(targets, n_ev, prior, rounds, maxiter, bound):
    """Minimize the joint chi2; the surfaces are updated in place.
    Returns the event weights on the sample."""
    dead = np.zeros(n_ev, dtype=bool)
    for t in targets:
        dead |= t.dead_events(n_ev)
    alive = ~dead
    print(f"[fit] {int(alive.sum())}/{n_ev} events ({100.0*alive.mean():.1f}%) "
          "survive the data-empty / outside-domain cells")
    if not alive.any():
        raise SystemExit("no event survives; check the grids against the "
                         "generated ranges.")
    for t in targets:
        t.set_live(alive)

    fit = Fit(targets, n_ev, alive, prior)
    print(f"[fit] {fit.n_par} parameters (" + " + ".join(
        f"{len(t.fit_cells)} {t.label}" for t in targets) + f"), prior {prior:g}")
    theta = fit.theta()
    lb = np.log(bound)
    chi2_prev = None
    for r in range(1, rounds + 1):
        fit.update_sigma()
        res = minimize(fit, theta, jac=True, method="L-BFGS-B",
                       bounds=[(-lb, lb)] * fit.n_par,
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
        theta = res.x
        fit.set_theta(theta)
        ndf = max(fit.n_par, 1)
        print(f"[fit] round {r}: chi2 = {res.fun:.1f} for {ndf} parameters "
              f"({res.nit} iterations, {res.message if isinstance(res.message, str) else res.message.decode()})")
        if chi2_prev is not None and abs(chi2_prev - res.fun) < 1e-3 * max(res.fun, 1.0):
            break
        chi2_prev = res.fun
    at_bound = int((np.abs(theta) > 0.999 * lb).sum())
    if at_bound:
        print(f"[fit] WARNING: {at_bound} factors sit at the --bound {bound:g}; "
              "those cells are not reachable from this proposal.")
    return event_weights(targets, n_ev, alive)


def closure(targets, w, fit):
    """Fit residuals per surface: how well the weighted sample matches."""
    for t, rv in zip(targets, fit.rel_var):
        h, _ = fit.normalized(t, w)
        f = t.fitted
        dev = h[f] / t.T[f] - 1.0
        pull = (h[f] - t.T[f]) / np.sqrt(rv[f])
        print(f"[closure] {t.label}: over {int(f.sum())} fitted cells "
              f"chi2/cell = {(pull**2).mean():.2f}, rms pull {np.sqrt((pull**2).mean()):.2f}, "
              f"max |pull| {np.abs(pull).max():.1f}; median |gen/target - 1| = "
              f"{np.median(np.abs(dev)):.3f}")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_surfaces(targets, out):
    """One TH3D per surface, plus a <name>_fitted 0/1 mask.

    The pass-through factor for M outside the grid goes into the
    under/overflow bins along M of every in-range (Q2, W) column, which is
    what TAxis::FindBin returns there -- so EventWeighter reads it with the
    same per-bin lookup and no special case.
    """
    tf = ROOT.TFile(out, "RECREATE")
    for t in targets:
        u = t.grid()
        out_m = float(t.u[t.OUT_M])
        nx, ny, nz = u.shape
        xe, ye, ze = (_darr('d', [float(v) for v in e]) for e in t.edges)
        title = (f"w(Q^{{2}},W,M_{{{t.label}}}) fitted for {'interp' if t.interp else 'bin'};"
                 f"Q^{{2}} [GeV^{{2}}];W [GeV];M({t.label}) [GeV]")
        h = ROOT.TH3D(t.name, title, nx, xe, ny, ye, nz, ze)
        hm = ROOT.TH3D(t.name + "_fitted", "1 = cell fitted to the target;"
                       + title.split(";", 1)[1], nx, xe, ny, ye, nz, ze)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    h.SetBinContent(i + 1, j + 1, k + 1, float(u[i, j, k]))
                    hm.SetBinContent(i + 1, j + 1, k + 1, float(t.fitted[i, j, k]))
                h.SetBinContent(i + 1, j + 1, 0, out_m)
                h.SetBinContent(i + 1, j + 1, nz + 1, out_m)
        h.Write()
        hm.Write()
    tf.Close()


def main():
    ap = argparse.ArgumentParser(
        description="Fit joint pair-mass weights w(Q2, W, M_AB) to several "
                    "measured pair-mass distributions at once.")
    ap.add_argument("--target", action="append", nargs="+", required=True,
                    metavar="ARG",
                    help="'<pidA,pidB> <D.npz> [<R.npz>] [<hist name>]', "
                         "repeatable. D: the measured (sideband-subtracted) "
                         "3-D histogram of M(A B) in (Q2, W, M); R: the "
                         "reconstructed sim of the --gen run on the same grid, "
                         "for a reco-level correction. Name defaults to "
                         "w_pair_<pidA>_<pidB>.")
    ap.add_argument("--gen", required=True,
                    help="the run to correct: its LUND file, or its truth "
                         "ntuple (.root). Unweighted for iteration 0, the "
                         "previous surfaces' run with --prev.")
    ap.add_argument("--tree", default="truth", help="tree name for a .root --gen")
    ap.add_argument("--out", required=True, help="output ROOT file (all surfaces)")
    ap.add_argument("--mode", choices=("interp", "bin"), default="interp",
                    help="how the generator will read the surfaces, fitted "
                         "accordingly: interp (default) trilinear between bin "
                         "centers -- continuous in Q2, W, M; bin -- per-cell "
                         "steps. Set pair_weight_mode in the card to match.")
    ap.add_argument("--target-units", choices=("integral", "density"), default=None,
                    help="how to read the target values; default: what the npz "
                         "records, else integral -- i.e. per-bin counts, which "
                         "is what a measured histogram is. Pass density for a "
                         "cross section quoted as dsigma/dQ2 dW dM.")
    ap.add_argument("--min-gen", type=float, default=None,
                    help="zero cells the sample visits fewer times than this. "
                         "An accuracy floor: a delivered cell with N entries has "
                         "its weight known to 1/sqrt(N). Default from "
                         "--target-accuracy.")
    ap.add_argument("--target-accuracy", type=float, default=0.05,
                    help="sets --min-gen = 1/accuracy^2 when --min-gen is not "
                         "given (default 0.05 -> 400)")
    ap.add_argument("--min-rec", type=float, default=25.0,
                    help="with an R.npz: zero cells whose reconstructed count "
                         "is below this (default 25)")
    ap.add_argument("--prior", type=float, default=1.0,
                    help="weight of the sum(log u)^2 term in the chi2 "
                         "(default 1: a factor known to 10%% is pulled toward "
                         "1 by ~1%%). It fixes the flat direction of the product "
                         "form and keeps poorly measured cells from wandering.")
    ap.add_argument("--bound", type=float, default=1e3,
                    help="hard bound on every factor, [1/bound, bound] (default 1e3)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="outer rounds re-estimating the sample's own variance "
                         "term between L-BFGS runs (default 4)")
    ap.add_argument("--max-iter", type=int, default=2000,
                    help="L-BFGS iterations per round (default 2000)")
    ap.add_argument("--prev", default=None,
                    help="previous cumulative pair_weight ROOT file (same "
                         "names and grids); the new corrections are multiplied "
                         "into it so --out holds the running product")
    ap.add_argument("--archive", default=None,
                    help="directory for a versioned copy w_pair_iter<N>.root")
    args = ap.parse_args()

    if args.min_gen is None:
        acc = max(args.target_accuracy, 1e-6)
        args.min_gen = float(np.ceil(1.0 / (acc * acc)))
        print(f"[guard] --target-accuracy {acc:g} -> --min-gen {args.min_gen:g}")

    gen = load_truth(args.gen, args.tree, final_state=True)
    n_ev = len(gen["Q2"])
    print(f"[gen] {args.gen}:{args.tree}: {n_ev} events")

    targets = [Target(spec, gen, args.target_units, args.min_gen, args.min_rec,
                      interp=(args.mode == "interp"))
               for spec in args.target]
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate surface names {names}; give each --target "
                         "its own name.")

    w = fit_surfaces(targets, n_ev, args.prior, args.rounds, args.max_iter, args.bound)
    fit_state = Fit(targets, n_ev, w > 0, args.prior)
    fit_state.update_sigma()
    closure(targets, w, fit_state)

    if args.prev:
        for t in targets:
            prev = read_prev_th3(args.prev, t.name, t.edges)
            prev_out = read_prev_outside(args.prev, t.name)
            g = t.grid() * prev
            if g.max() <= 0:
                raise SystemExit(f"{t.name}: cumulative weight is all zero after --prev.")
            t.u[:t.ncell] = g.ravel()
            t.u[t.OUT_M] *= prev_out
        print(f"[prev] multiplied in {args.prev} -> cumulative surfaces")

    # Normalize each surface so the largest factor any sample entry actually
    # RECEIVES is 1 (pass-through included). Every factor the generator
    # forms is then <= 1 up to what the sample did not visit, so the product
    # is a valid accept probability. Not the largest NODE value: with
    # interpolation a node at a kinematic edge is only ever seen with a
    # small weight (every event in its cell sits on one side of its center)
    # and the fit inflates it by 1 / weight to reproduce the data there --
    # a value no event reaches, which would set the scale for nothing.
    # The generator counts pair factors above 1 (accepted outright) so a
    # region the sample missed shows up in its summary.
    for t in targets:
        realized = max(float(t.factors().max()), float(t.u[t.OUT_M]))
        node_max = float(t.u[:t.ncell].max())
        t.u /= realized
        if node_max > 1.5 * realized:
            print(f"[norm] {t.name}: largest node {node_max/realized:.1f}x the largest "
                  "realized factor -- an edge node seen only with a small "
                  "interpolation weight; normalized to the realized maximum.")
        u = t.grid()
        top = np.argsort(u.ravel())[::-1][:3]
        cells = []
        for c in top:
            i, j, k = np.unravel_index(c, u.shape)
            cells.append(f"Q2[{t.edges[0][i]:.2f},{t.edges[0][i+1]:.2f}] "
                         f"W[{t.edges[1][j]:.2f},{t.edges[1][j+1]:.2f}] "
                         f"M[{t.edges[2][k]:.2f},{t.edges[2][k+1]:.2f}] "
                         f"u={u[i,j,k]:.3f} n_gen={t.G0[i,j,k]:.0f}")
        print(f"[norm] {t.name}: max -> 1; pass-through factor "
              f"{t.u[t.OUT_M]:.3f}; zero-fraction {float(np.mean(u <= 0)):.3f}; "
              "cells setting the scale:\n      " + "\n      ".join(cells))

    # expected keep fraction on this sample, with the normalized surfaces
    eff = float(event_weights(targets, n_ev, w > 0).mean())
    print(f"[eff] mean accept probability = {eff:.4f} "
          f"(~1 kept event per {1.0/max(eff,1e-12):.1f} fully built events)")
    if eff < 0.02:
        print("[eff] that is expensive -- every rejection here has been through "
              "the full decay chain. Raise --min-gen (the cells setting the "
              "scale are listed above) or widen the proposal (mass_9999 width) "
              "where the target has support the generator rarely visits.")

    write_surfaces(targets, args.out)
    print(f"[write] {args.out}: " + ", ".join(
        f"{t.name} ({'x'.join(str(n) for n in t.shape)})" for t in targets))
    if args.archive:
        print(f"[archive] stored -> {_archive_copy(args.out, args.archive, 'pair')}")

    print("\nAdd these lines to the generator's input card (all of them: the "
          "surfaces were fitted together):")
    for t in targets:
        print(f"    pair_weight: {args.out} {t.name} {t.pid_a} {t.pid_b}")
    print(f"    pair_weight_mode: {args.mode}")


if __name__ == "__main__":
    main()
