"""
Pseudo 3-D cross sections for the pooled M(p pbar) and M(p p) pair masses,
in exactly the npz layout ppbar_acceptance.ipynb's `save_xsec3d` writes --
a closure test of the pair_weight chain with a KNOWN answer.

An unweighted run (LUND or truth ntuple), INDEPENDENT of the one the
builder will fit on, is reweighted by a function of the Dalitz plane that
is deliberately not of product form and not symmetric in the two pairings,

    w = exp(-1.5 (M(p_lead pbar) - 2)) (1 + 0.6 cos 4 M_pp) exp(-0.3 (Q2 - 2)),

binned on the analysis grid, Poisson-fluctuated, divided by a luminosity
and the accessible bin volume (closed bins NaN, as accessible_bin_volume()
does), and written with `counts` = the per-bin integral. Then:

    python make_pseudo_pair_xsec.py --gen ../events_pseudodata.lund
    python build_pair_weight.py \\
        --target 2212,-2212 ../pseudo_xsec3d_Mppbar_pooled.npz \\
        --target 2212,2212  ../pseudo_xsec3d_Mpp.npz \\
        --gen ../events_unweighted.lund --out ../pair_weight.root
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truth_ntuple import load_truth, pair_masses   # noqa: E402

M_P = 0.9382720813


def accessible_volume(edges, e_beam, n=2000, min_frac=0.01, seed=0):
    """Open fraction x rectangle per bin, NaN where < min_frac (mirrors the
    notebook's accessible_bin_volume)."""
    rng = np.random.default_rng(seed)
    shape = tuple(len(e) - 1 for e in edges)
    frac = np.zeros(shape)
    for idx in np.ndindex(shape):
        (q0, q1), (w0, w1), (m0, m1) = ((edges[d][idx[d]], edges[d][idx[d] + 1])
                                        for d in range(3))
        q, w, m = (rng.uniform(lo, hi, n) for lo, hi in ((q0, q1), (w0, w1), (m0, m1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            nu = (w ** 2 + q - M_P ** 2) / (2 * M_P)
            ep = e_beam - nu
            s2 = q / (4 * e_beam * ep)
            ok = (ep > 0) & (s2 > 0) & (s2 < 1) & (m <= w - M_P)
        frac[idx] = ok.mean()
    rect = (np.diff(edges[0])[:, None, None] * np.diff(edges[1])[None, :, None]
            * np.diff(edges[2])[None, None, :])
    return np.where(frac >= min_frac, frac * rect, np.nan)


def save_xsec3d(xs, path, varnames, label=""):
    """Verbatim copy of the notebook cell: counts = sigma integrated over the
    bin (N x V_used), NaN-blanked bins -> 0, density kept as `xsec`."""
    V = np.asarray(xs["bin_volume_used"], float)
    dens = np.asarray(xs["N"], float)
    err = np.asarray(xs["sigma_tot"], float)
    counts = np.nan_to_num(dens * V, nan=0.0)
    errors = np.nan_to_num(err * V, nan=0.0)
    edges = [np.asarray(e, float) for e in xs["edges"]]
    np.savez(path, counts=counts, errors=errors,
             xsec=np.nan_to_num(dens, nan=0.0), xsec_err=np.nan_to_num(err, nan=0.0),
             volume=np.nan_to_num(V, nan=0.0),
             edges=np.array(edges, dtype=object), varnames=np.array(varnames),
             units="integral", xsec_units=str(xs["units"]))
    filled = counts > 0
    print(f"saved {path}{' (' + label + ')' if label else ''}: grid "
          f"{'x'.join(str(len(e)-1) for e in edges)}, {int(filled.sum())} filled bins, "
          f"integral {counts.sum():.4g} {xs['units']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True,
                    help="LUND or truth ntuple of an unweighted run, INDEPENDENT "
                         "of the one build_pair_weight.py will fit on")
    ap.add_argument("--grid", default="/Users/leo/Research/Python/ppbar/Analysis/q_factors_data.npz",
                    help="npz with Q2_cuts, W_cuts, IM_cuts (the analysis grid)")
    ap.add_argument("--beam-energy", type=float, default=10.2)
    ap.add_argument("--lumi", type=float, default=1e-3, help="pseudo luminosity [1/pb]")
    ap.add_argument("--out-dir", default="..")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    g = np.load(args.grid)
    edges = [np.asarray(g[k], float) for k in ("Q2_cuts", "W_cuts", "IM_cuts")]
    a = load_truth(args.gen, final_state=True)
    ev, m_pool, _ = pair_masses(a, 2212, -2212)
    ev2, m_pp, _ = pair_masses(a, 2212, 2212)
    q2, w = (np.asarray(a[k], float) for k in ("Q2", "W"))
    n_ev = len(q2)

    # the "physics": a weight on the Dalitz plane through M(p_lead pbar) --
    # the pairing with the faster proton, well defined from the LUND alone
    # and asymmetric in the two pairings -- and M(p p)
    pid, valid = a["fs_pid"], np.arange(a["fs_pid"].shape[1])[None, :] < a["nfs"][:, None]
    pmag = np.sqrt(a["fs_px"] ** 2 + a["fs_py"] ** 2 + a["fs_pz"] ** 2)
    pmag_p = np.where(valid & (pid == 2212), pmag, -1.0)
    lead = pmag_p.argmax(axis=1)
    ipb = np.where(valid & (pid == -2212), 1, 0).argmax(axis=1)
    rows = np.arange(n_ev)
    e_ = a["fs_E"][rows, lead] + a["fs_E"][rows, ipb]
    p2 = sum((a[k][rows, lead] + a[k][rows, ipb]) ** 2 for k in ("fs_px", "fs_py", "fs_pz"))
    m_lead = np.sqrt(np.clip(e_ ** 2 - p2, 0, None))
    w_true = (np.exp(-1.5 * (m_lead - 2.0)) * (1 + 0.6 * np.cos(4 * m_pp))
              * np.exp(-0.3 * (q2 - 2)))
    w_true /= w_true.max()
    print(f"[truth] {n_ev} events, mean known weight {w_true.mean():.3f}")

    V = accessible_volume(edges, args.beam_energy)
    print(f"[volume] {int(np.isnan(V).sum())} of {V.size} bins closed -> NaN")
    rng = np.random.default_rng(args.seed)

    def xs(ev_, m_):
        pts = np.column_stack([q2[ev_], w[ev_], m_])
        Y, _ = np.histogramdd(pts, bins=edges, weights=w_true[ev_])
        Y = rng.poisson(Y).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(N=Y / (args.lumi * V), sigma_tot=np.sqrt(Y) / (args.lumi * V),
                        bin_volume_used=V, edges=edges, units="pb")

    save_xsec3d(xs(ev, m_pool), os.path.join(args.out_dir, "pseudo_xsec3d_Mppbar_pooled.npz"),
                ["Q2", "W", "M_ppbar"], "pooled p pbar, ep-level")
    save_xsec3d(xs(ev2, m_pp), os.path.join(args.out_dir, "pseudo_xsec3d_Mpp.npz"),
                ["Q2", "W", "M_pp"], "p1 p2, ep-level")


if __name__ == "__main__":
    main()
