"""
Toy Monte Carlo: does ONE pooled pair weight close on TWO pair masses?

Standalone -- no ROOT, no generator, no files from the rest of the repo.
It rebuilds the p p pbar problem in a single (Q2, W) cell, where it is
exact kinematics rather than an approximation: the hadronic final state of
e p -> e' p p pbar IS the three-body system of mass W, so at fixed W every
event is one point on its Dalitz plot, fixed by any two of

    M1 = M(p1 pbar),   M2 = M(p2 pbar),   M_pp = M(p1 p2),
    M1^2 + M2^2 + M_pp^2 = W^2 + 3 m_p^2.

The question
------------
Data cannot tell the protons apart, so the measured p pbar mass is POOLED:
every event fills M1 AND M2. The generator weights each event with

    w = u(M1) * u(M2) * v(M_pp)

-- the SAME surface u evaluated on two different variables of one event,
times v on a third -- and keeps it with probability w / max(w). The claim
tested here: with u and v FITTED jointly (as reweight/build_pair_weight.py
does), the kept events reproduce the pooled M(p pbar) and the M(p p)
distributions at once, including their relative normalization (2 entries
to 1 per event), on an event sample the fit never saw.

What the toy does
-----------------
1. TRUTH ("data"). A model the generator knows nothing about: 60% of events
   go through a narrow X(2.25, 0.10) -> p pbar with the other proton a
   spectator, 40% are three-body phase space; all of it multiplied by a
   p p final-state enhancement 1 + A exp(-(M_pp - 2 m_p) / b) near the p p
   threshold. Filled POOLED, as a measurement would be.
2. PROPOSAL. What the real generator does: the "X" mass from a broad
   Breit-Wigner (2.0, 0.4), isotropic decays -- so it is not flat, and has
   the wrong resonance, the wrong width and no p p enhancement.
3. FIT u, v on one proposal sample (chi2 of the weighted pooled and p p
   histograms against the data, analytic gradient, L-BFGS-B), per-bin
   lookup.
4. CLOSURE on an INDEPENDENT proposal sample, by accept-reject -- exactly
   what the generator does.
5. The same closure for the shortcuts that do NOT work, so it is clear
   what the fit buys:
     A  u = pooled d/g ratio, applied to both pairings  (no fit)
     B  pooled d/g ratio applied to the TRUE pairing     (the old xsec_weight
                                                         with a pooled target)
     C  fitted u, but on pooled p pbar only             (no v)
     D  fitted u and v jointly                           (the method)
6. What the two targets do NOT fix: M_lo = min(M1, M2) and M_hi =
   max(M1, M2) separately. The pooled histogram is their sum, so they are
   free to trade entries; the product form is a model there, and the toy
   shows how far it lands from the truth.

Reading the numbers
-------------------
chi2/ndf VS DATA is the check you can run on real data. It comes out
below 1: the fit absorbed the data's own fluctuations, so the generated
points track the data more closely than their combined errors say.
chi2/ndf VS TRUTH (a 20x larger truth sample, only possible in a toy) is
the honest statistic: the data's noise that the fit absorbed now counts
as error, and a working method gives ~1 and a pull distribution of width
~1. Its error has four parts: the closure sample, the data the fit
absorbed, the finite fit sample the surfaces were built from (comparable
to the data's at a ~12% keep rate), and the truth sample itself.

Usage
-----
    python pooled_weight_toy.py                    # defaults, ~10 s
    python pooled_weight_toy.py --W 3.2 --n-data 30000 --seed 7
writes pooled_weight_toy.pdf next to this file.
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

M_P = 0.938272

# Reference categorical palette, slots 1-3 (validated all-pairs); data in ink.
GEN, ALT, AUX = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#8a8983", "#e4e3df"


# ---------------------------------------------------------------------------
# Three-body kinematics at fixed W (all three masses = m_p)
# ---------------------------------------------------------------------------
def s2_range(s1, W):
    """Allowed M2^2 = M(p2 pbar)^2 for given s1 = M(p1 pbar)^2, from the
    energies of p2 and pbar in the (p1 pbar) rest frame."""
    m = M_P
    rs = np.sqrt(s1)
    e3 = rs / 2.0                                 # pbar
    p3 = np.sqrt(np.maximum(e3**2 - m * m, 0.0))
    e2 = (W * W - s1 - m * m) / (2.0 * rs)        # spectator proton
    p2 = np.sqrt(np.maximum(e2**2 - m * m, 0.0))
    return (e2 + e3)**2 - (p2 + p3)**2, (e2 + e3)**2 - (p2 - p3)**2


def event(s1, s2, W):
    """(M1, M2, M_pp) from the two p pbar invariants."""
    spp = W * W + 3 * M_P**2 - s1 - s2
    return np.sqrt(s1), np.sqrt(s2), np.sqrt(np.maximum(spp, 4 * M_P**2))


def bw_trunc(n, m0, g, lo, hi, rng):
    """Breit-Wigner (Cauchy) in mass, truncated to [lo, hi], by inversion."""
    a, b = np.arctan(2 * (lo - m0) / g), np.arctan(2 * (hi - m0) / g)
    return m0 + 0.5 * g * np.tan(rng.uniform(a, b, n))


def sequential(mx, W, rng):
    """X(mx) -> p1 pbar with p2 a spectator, isotropic: for fixed s1, M2^2 is
    uniform over its range."""
    s1 = mx * mx
    lo, hi = s2_range(s1, W)
    return s1, lo + (hi - lo) * rng.uniform(size=s1.size)


def phase_space(n, W, rng):
    """Uniform on the Dalitz plot (= three-body phase space)."""
    smin, smax = 4 * M_P**2, (W - M_P)**2
    out1, out2, have = [], [], 0
    while have < n:
        k = 2 * (n - have) + 1000
        s1 = rng.uniform(smin, smax, k)
        s2 = rng.uniform(smin, smax, k)
        lo, hi = s2_range(s1, W)
        ok = (s2 >= lo) & (s2 <= hi)
        out1.append(s1[ok]); out2.append(s2[ok]); have += ok.sum()
    return np.concatenate(out1)[:n], np.concatenate(out2)[:n]


# ---------------------------------------------------------------------------
# The two samples
# ---------------------------------------------------------------------------
def truth_sample(n, W, rng, f_res=0.6, mres=2.25, gres=0.10, fsi_a=3.0, fsi_b=0.08):
    """The 'data' model. Returns (M1, M2, Mpp)."""
    lo, hi = 2 * M_P, W - M_P
    parts, have = [], 0
    while have < n:
        k = 2 * (n - have) + 1000
        nres = rng.binomial(k, f_res)
        s1r, s2r = sequential(bw_trunc(nres, mres, gres, lo, hi, rng), W, rng)
        s1p, s2p = phase_space(k - nres, W, rng)
        s1, s2 = np.concatenate([s1r, s1p]), np.concatenate([s2r, s2p])
        m1, m2, mpp = event(s1, s2, W)
        fsi = 1.0 + fsi_a * np.exp(-(mpp - 2 * M_P) / fsi_b)
        keep = rng.uniform(size=k) * (1.0 + fsi_a) < fsi
        # which proton is "p1" is unknowable in data -- shuffle the labels
        swap = rng.uniform(size=k) < 0.5
        a = np.where(swap, m2, m1)[keep]; b = np.where(swap, m1, m2)[keep]
        parts.append((a, b, mpp[keep])); have += keep.sum()
    return tuple(np.concatenate([p[i] for p in parts])[:n] for i in range(3))


def proposal_sample(n, W, rng, mx0=2.0, gx=0.4):
    """What the generator throws: X from a broad BW, isotropic decays.
    Returns (M1, M2, Mpp, M_X) with M1 the TRUE pairing (= M_X)."""
    mx = bw_trunc(n, mx0, gx, 2 * M_P, W - M_P, rng)
    s1, s2 = sequential(mx, W, rng)
    m1, m2, mpp = event(s1, s2, W)
    return m1, m2, mpp, mx


# ---------------------------------------------------------------------------
# Histograms and the fit
# ---------------------------------------------------------------------------
def binidx(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def pooled(i1, i2, w, nb):
    return np.bincount(i1, w, nb) + np.bincount(i2, w, nb)


def fill(idxs, w, nb):
    """Histogram of one or two fills per event and its variance. The
    variance is per EVENT: an event whose two pooled entries land in the
    same bin adds (2w)^2 there, not 2 w^2."""
    w = np.ones(idxs[0].size) if w is None else w
    H = sum(np.bincount(i, w, nb) for i in idxs)
    V = sum(np.bincount(i, w * w, nb) for i in idxs)
    if len(idxs) == 2:
        same = idxs[0] == idxs[1]
        V = V + 2.0 * np.bincount(idxs[0][same], (w * w)[same], nb)
    return H.astype(float), V.astype(float)


def fit_surfaces(i1, i2, ipp, k, T_ab, T_pp, nb, use_pp=True, prior=1e-3, bound=15.0):
    """Fit log u (nb) and log v (nb) so that the weighted proposal sample,
    w = k u(M1) u(M2) v(Mpp), matches the pooled and p p targets.

    chi2 = sum (H - T)^2 / T over both targets + prior |theta|^2. k puts the
    proposal on the data's scale up front, so theta = 0 means 'no change'
    and the weak prior only acts on the flat direction u -> c u,
    v -> v / c^2 and on bins with no events."""
    sig_ab, sig_pp = np.maximum(T_ab, 1.0), np.maximum(T_pp, 1.0)

    def f(theta):
        tu = theta[:nb]
        tv = theta[nb:] if use_pp else np.zeros(nb)
        w = k * np.exp(tu[i1] + tu[i2] + tv[ipp])
        d_ab = pooled(i1, i2, w, nb) - T_ab
        r_ab = d_ab / sig_ab
        chi = (r_ab * d_ab).sum()
        r = 2.0 * (r_ab[i1] + r_ab[i2])
        if use_pp:
            d_pp = np.bincount(ipp, w, nb) - T_pp
            r_pp = d_pp / sig_pp
            chi += (r_pp * d_pp).sum()
            r = r + 2.0 * r_pp[ipp]
        wr = w * r
        g = [pooled(i1, i2, wr, nb)]
        if use_pp:
            g.append(np.bincount(ipp, wr, nb))
        g = np.concatenate(g) + 2.0 * prior * theta
        return chi + prior * (theta**2).sum(), g

    n = 2 * nb if use_pp else nb
    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B",
                   bounds=[(-bound, bound)] * n,
                   options=dict(maxiter=20000, maxfun=40000))
    tu = res.x[:nb]
    tv = res.x[nb:] if use_pp else np.zeros(nb)
    return np.exp(tu), np.exp(tv), res


def accept(w, wmax, rng):
    """The generator's accept-reject: keep with probability w / wmax."""
    p = w / wmax
    return rng.uniform(size=w.size) < p, int((p > 1).sum())


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9, color=MUTED)


def overlay(fig, spec, edges, T, VT, G, VG, xl, title, rr=(0.8, 1.2), prop=None,
            gl="Weighted generator", tl="Data"):
    """Upper: reference points and the generated histogram; lower: their
    ratio. G is already on the reference's scale; VT, VG the variances."""
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                    height_ratios=[3, 1.2], hspace=0.22)
    ax = fig.add_subplot(inner[0]); rx = fig.add_subplot(inner[1], sharex=ax)
    c = 0.5 * (edges[1:] + edges[:-1])
    if prop is not None:
        ax.stairs(prop, edges, color=MUTED, lw=1.4, ls="--", label="Proposal (unweighted)")
    ax.stairs(G, edges, color=GEN, lw=2.0, label=gl)
    ax.errorbar(c, T, yerr=np.sqrt(VT), fmt="o", ms=4, color=INK, lw=1.0,
                label=tl, zorder=5)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("Entries / bin", fontsize=10)
    ax.set_ylim(0, 1.18 * max(T.max(), G.max(), prop.max() if prop is not None else 0))
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    style(ax)

    with np.errstate(divide="ignore", invalid="ignore"):
        ok = T > 0
        r = np.where(ok, G / T, np.nan)
        e = np.where(ok, np.sqrt(VG + VT) / T, np.nan)
    rx.axhline(1.0, color=INK, lw=1.0)
    rx.errorbar(c, np.clip(r, *rr), yerr=e, fmt="o", ms=3.5, color=GEN, lw=1.0)
    out = np.isfinite(r) & ((r < rr[0]) | (r > rr[1]))
    for ci, ri in zip(c[out], r[out]):
        rx.plot(ci, rr[1] if ri > rr[1] else rr[0], "^" if ri > rr[1] else "v",
                ms=6, color=ALT, clip_on=False, zorder=6)
    rx.set_ylim(*rr)
    rx.set_ylabel("Gen / data", fontsize=9)
    rx.set_xlabel(xl, fontsize=10)
    style(rx)
    return ax, rx


def chi2(G, R, var):
    ok = ((G > 0) | (R > 0)) & (var > 0)
    return float(((G - R)**2 / np.where(ok, var, 1.0))[ok].sum()), int(ok.sum())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--W", type=float, default=3.6, help="p p pbar mass = W [GeV]")
    ap.add_argument("--n-data", type=int, default=100_000, help="data events")
    ap.add_argument("--n-gen", type=int, default=1_000_000,
                    help="proposal events, for the fit AND again for the closure")
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "pooled_weight_toy.pdf"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    W, nb = a.W, a.nbins
    edges = np.linspace(2 * M_P, W - M_P, nb + 1)
    lab_ab, lab_pp = r"$M(p\bar{p})$ [GeV]", r"$M(pp)$ [GeV]"

    # ---- samples ----
    d1, d2, dpp = truth_sample(a.n_data, W, rng)
    b1, b2, bpp = truth_sample(20 * a.n_data, W, rng)            # toy-only truth
    f1, f2, fpp, fx = proposal_sample(a.n_gen, W, rng)            # fit sample
    c1, c2, cpp, cx = proposal_sample(a.n_gen, W, rng)            # closure sample

    I = lambda x: binidx(x, edges)                                # noqa: E731
    lo_hi = lambda x1, x2: (I(np.minimum(x1, x2)), I(np.maximum(x1, x2)))  # noqa: E731
    obs = ("ab", "pp", "lo", "hi")                  # pooled, p p, M_lo, M_hi

    def fills(i1, i2, ipp, ilo, ihi):
        return dict(ab=[i1, i2], pp=[ipp], lo=[ilo], hi=[ihi])

    dF = fills(I(d1), I(d2), I(dpp), *lo_hi(d1, d2))
    bF = fills(I(b1), I(b2), I(bpp), *lo_hi(b1, b2))
    T, VT, E, VE = {}, {}, {}, {}
    for o in obs:
        T[o], VT[o] = fill(dF[o], None, nb)
        E[o], VE[o] = fill(bF[o], None, nb)
        E[o], VE[o] = E[o] / 20.0, VE[o] / 400.0    # truth on the data scale
    T_ab, T_pp = T["ab"], T["pp"]
    fi1, fi2, fipp, fix = I(f1), I(f2), I(fpp), I(fx)
    ci1, ci2, cipp, cix = I(c1), I(c2), I(cpp), I(cx)
    fF = fills(fi1, fi2, fipp, *lo_hi(f1, f2))
    k = a.n_data / a.n_gen
    print(f"[toy] W = {W} GeV, {nb} bins on [{edges[0]:.3f}, {edges[-1]:.3f}]; "
          f"{a.n_data} data events, 2 x {a.n_gen} proposal events")

    # ---- weights: the method and the shortcuts ----
    methods = {}
    u, v, res = fit_surfaces(fi1, fi2, fipp, k, T_ab, T_pp, nb)
    print(f"[fit D] joint u, v: chi2 {res.fun:.1f} on {2*nb} target bins, "
          f"{res.nit} iterations, {res.message}")
    methods["D"] = ("Fitted u and v, jointly  (the method)",
                    lambda i1, i2, ipp, ix: u[i1] * u[i2] * v[ipp])
    uC, _, resC = fit_surfaces(fi1, fi2, fipp, k, T_ab, T_pp, nb, use_pp=False)
    print(f"[fit C] u only: chi2 {resC.fun:.1f} on {nb} pooled bins")
    methods["C"] = ("Fitted u on pooled p pbar only  (no v)",
                    lambda i1, i2, ipp, ix: uC[i1] * uC[i2])
    G0_ab = pooled(fi1, fi2, None, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rA = np.where(G0_ab > 0, (T_ab / T_ab.sum()) / (G0_ab / G0_ab.sum()), 0.0)
        g_true = np.bincount(fix, None, nb)
        rB = np.where(g_true > 0, (T_ab / T_ab.sum()) / (g_true / g_true.sum()), 0.0)
    methods["A"] = ("Pooled d/g ratio on both pairings  (no fit)",
                    lambda i1, i2, ipp, ix: rA[i1] * rA[i2])
    methods["B"] = ("Pooled d/g ratio on the true pairing  (old xsec_weight)",
                    lambda i1, i2, ipp, ix: rB[ix])

    # ---- closure: accept-reject on the independent sample ----
    closure = {}
    for key, (name, wf) in methods.items():
        wF = wf(fi1, fi2, fipp, fix)
        wmax = wF.max()                                # normalized on the FIT sample
        keep, over = accept(wf(ci1, ci2, cipp, cix), wmax, rng)
        n_acc = int(keep.sum())
        s = n_acc / a.n_data                           # one scale, from event counts
        wF = wF * a.n_data / wF.sum()                  # fit sample on the data scale
        r = dict(name=name, n_acc=n_acc, eff=n_acc / a.n_gen, over=over)
        cF = fills(ci1[keep], ci2[keep], cipp[keep], *lo_hi(c1[keep], c2[keep]))
        for o in obs:
            G, VG = fill(cF[o], None, nb)
            r[o], r["V" + o] = G / s, VG / s**2
            # Noise the surfaces inherit from the finite FIT sample: the
            # variance of its weighted histogram.
            VA = fill(fF[o], wF, nb)[1]
            # vs data: what real data allows (data + generator statistics)
            r["chi_data_" + o] = chi2(r[o], T[o], r["V" + o] + VT[o])
            # vs truth: every noise source -- the closure sample, the data
            # the fit absorbed, the fit sample, the truth sample itself
            vt = r["V" + o] + VT[o] + VA + VE[o]
            r["chi_true_" + o] = chi2(r[o], E[o], vt)
            r["Vtrue_" + o] = vt
            ok = ((r[o] > 0) | (E[o] > 0)) & (vt > 0)
            r["pull_" + o] = ((r[o] - E[o]) / np.sqrt(np.where(ok, vt, 1.0)))[ok]
        closure[key] = r

    print("\n  method                                                   eff    "
          "chi2/ndf vs data      chi2/ndf vs truth")
    print("                                                                  "
          "pooled    p p         pooled    p p")
    for key in "ABCD":
        r = closure[key]
        f = lambda c: f"{c[0]/c[1]:7.2f}"                        # noqa: E731
        print(f"  {key} {r['name']:<55s} {r['eff']:5.3f}  {f(r['chi_data_ab'])} "
              f"{f(r['chi_data_pp'])}     {f(r['chi_true_ab'])} {f(r['chi_true_pp'])}"
              + (f"   ({r['over']} events with w > max)" if r["over"] else ""))
    D = closure["D"]
    pulls = np.concatenate([D["pull_ab"], D["pull_pp"]])
    print(f"\n[D] pulls vs truth over {pulls.size} bins: mean {pulls.mean():+.2f}, "
          f"rms {pulls.std():.2f}")

    # ---- the PDF ----
    pdf = PdfPages(a.out)
    fs = 4.2

    # page 1: the problem
    fig = plt.figure(figsize=(12, 5.2))
    gs = GridSpec(1, 2, figure=fig, left=0.07, right=0.98, top=0.8, bottom=0.12, wspace=0.25)
    for j, (o, P, xl, tt) in enumerate((("ab", G0_ab * k, lab_ab, "Pooled p pbar: 2 entries per event"),
                                        ("pp", np.bincount(fipp, None, nb) * k, lab_pp,
                                         "p p: 1 entry per event"))):
        ax = fig.add_subplot(gs[j])
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.stairs(P, edges, color=MUTED, lw=1.6, ls="--", label="Proposal (unweighted)")
        ax.errorbar(c, T[o], yerr=np.sqrt(VT[o]), fmt="o", ms=4, color=INK, lw=1.0, label="Data")
        ax.set_xlabel(xl, fontsize=10); ax.set_ylabel("Entries / bin", fontsize=10)
        ax.set_title(tt, fontsize=11, loc="left"); ax.legend(fontsize=9, frameon=False)
        style(ax)
    fig.suptitle(f"The problem: generator proposal vs measured pair masses   (W = {W} GeV)\n"
                 "data: X(2.25, 0.10) -> p pbar + spectator p, 40% phase space, p p threshold "
                 "enhancement    |    proposal: X from BW(2.0, 0.4)", fontsize=11.5)
    pdf.savefig(fig); plt.close(fig)

    # page 2: closure of the method
    fig = plt.figure(figsize=(12, 9.4))
    gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.98, top=0.9, bottom=0.07,
                  hspace=0.32, wspace=0.25, height_ratios=[1.35, 1])
    for j, (t, xl) in enumerate((("ab", lab_ab), ("pp", lab_pp))):
        cd, ct = D[f"chi_data_{t}"], D[f"chi_true_{t}"]
        overlay(fig, gs[0, j], edges, T[t], VT[t], D[t], D["V" + t], xl,
                ("Pooled p pbar" if t == "ab" else "p p")
                + f"   $\\chi^2$/ndf vs data {cd[0]:.0f}/{cd[1]}, vs truth {ct[0]:.0f}/{ct[1]}",
                prop=(G0_ab if t == "ab" else np.bincount(fipp, None, nb)) * k)
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-4, 4, 17)
    ax.hist(D["pull_ab"], bins, histtype="stepfilled", color=GEN, alpha=0.35,
            label=f"pooled p pbar ({D['pull_ab'].size} bins)")
    ax.hist(D["pull_pp"], bins, histtype="step", color=AUX, lw=2.0,
            label=f"p p ({D['pull_pp'].size} bins)")
    xg = np.linspace(-4, 4, 200)
    ax.plot(xg, pulls.size * 0.5 * np.exp(-xg**2 / 2) / np.sqrt(2 * np.pi) * (bins[1] - bins[0]),
            color=INK, lw=1.2, label="N(0, 1), per target")
    ax.set_title(f"Pulls vs truth model: mean {pulls.mean():+.2f}, rms {pulls.std():.2f}",
                 fontsize=11, loc="left")
    ax.set_xlabel("(generated - truth) / $\\sigma$", fontsize=10); ax.set_ylabel("Bins", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False); style(ax)
    ax = fig.add_subplot(gs[1, 1])
    c = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(c, u / u.max(), "-", color=GEN, lw=2.0, label="u, applied to M(p1 pbar) AND M(p2 pbar)")
    ax.plot(c, v / v.max(), "-", color=AUX, lw=2.0, label="v, applied to M(p p)")
    ax.set_yscale("log"); ax.set_xlabel("pair mass [GeV]", fontsize=10)
    ax.set_ylabel("factor / max", fontsize=10)
    ax.set_title("The fitted surfaces", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left"); style(ax)
    fig.suptitle("Closure: one weight  w = u(M1) u(M2) v(M_pp)  on an independent sample\n"
                 f"accept-reject exactly as the generator does it; efficiency {D['eff']:.1%}, "
                 f"{D['n_acc']} events kept, one normalization for both targets (from the event count)",
                 fontsize=11.5)
    pdf.savefig(fig); plt.close(fig)

    # page 3: the shortcuts
    fig = plt.figure(figsize=(12, 4 * fs))
    gs = GridSpec(4, 2, figure=fig, left=0.07, right=0.98, top=0.93, bottom=0.04,
                  hspace=0.42, wspace=0.25)
    for i, key in enumerate("ABCD"):
        r = closure[key]
        for j, (t, xl) in enumerate((("ab", lab_ab), ("pp", lab_pp))):
            ct = r[f"chi_true_{t}"]
            overlay(fig, gs[i, j], edges, T[t], VT[t], r[t], r["V" + t], xl,
                    f"{key}  {r['name']}" if j == 0 else
                    f"$\\chi^2$/ndf vs truth: pooled {r['chi_true_ab'][0]/r['chi_true_ab'][1]:.1f}, "
                    f"p p {ct[0]/ct[1]:.1f}",
                    rr=(0.5, 1.5))
    fig.suptitle("Why the surfaces are fitted: the same closure for the shortcuts "
                 "(ratio axis 0.5-1.5; triangles = off scale)", fontsize=12)
    pdf.savefig(fig); plt.close(fig)

    # page 4: what the targets do not fix
    fig = plt.figure(figsize=(12, 5.6))
    gs = GridSpec(1, 2, figure=fig, left=0.07, right=0.98, top=0.8, bottom=0.1, wspace=0.25)
    for j, (t, xl) in enumerate((("lo", r"$M_{lo}$ = min(M1, M2) [GeV]"),
                                 ("hi", r"$M_{hi}$ = max(M1, M2) [GeV]"))):
        cc = D["chi_true_" + t]
        # truth points carry every error except the generator's own
        ax, rx = overlay(fig, gs[j], edges, E[t], D["Vtrue_" + t] - D["V" + t], D[t], D["V" + t], xl,
                         f"$\\chi^2$/ndf vs truth {cc[0]:.0f}/{cc[1]}", rr=(0.7, 1.3),
                         gl="Weighted generator (D)", tl="Truth")
        rx.set_ylabel("Gen / truth", fontsize=9)
    fig.suptitle("Not fixed by the targets: the pooled histogram is M_lo + M_hi, so the two can trade "
                 "entries.\nHere the product form is a model -- this is how far it lands from the "
                 "truth.", fontsize=11.5)
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
