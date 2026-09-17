"""
Convert a real 3-D cross section into the npz the weight builder reads.

build_xsec_weight3d.py wants one file:

    counts    [nQ2, nW, nM] array of cross-section values
    edges     the three bin-edge arrays (object array, may be non-uniform)
    varnames  ["Q2", "W", "M"]
    units     "density" (dsigma/dQ2 dW dM) or "integral" (per-bin yield)

`units` is not decoration. The weight is a ratio of per-bin numbers, so a
density has to be multiplied by the bin volume first. With uniform binning
that is a constant and cancels; with Q2 edges like 1,2,3,4.5,7 it is a
factor 2.5 between the narrowest and widest bin, and getting it wrong
under-populates the wide bins by exactly that.

Sources
-------
    --th3 file.root:hist     a ROOT TH3 with axes (Q2, W, M)
    --csv file.csv           rows of Q2,W,M,sigma at BIN CENTERS
    --csv-edges file.csv     rows with explicit lo/hi per axis -- the
                             reliable form for non-uniform binning

Or roll your own; the format is five lines of numpy:

    np.savez("my_xsec.npz",
             counts=sigma,                      # [nQ2, nW, nM]
             edges=np.array([q2_edges, w_edges, m_edges], dtype=object),
             varnames=np.array(["Q2", "W", "M"]),
             units="density")

Usage
-----
    python import_xsec.py --th3 xsec.root:h_xsec --units density \\
        --out my_xsec.npz
    python import_xsec.py --csv xsec.csv --units integral --out my_xsec.npz
"""

import argparse
import csv as _csv

import numpy as np


def from_th3(spec):
    """Read a ROOT TH3 as (counts, edges). Axis order is X=Q2, Y=W, Z=M."""
    import ROOT
    if ":" not in spec:
        raise SystemExit("--th3 expects 'file.root:histname'")
    path, name = spec.rsplit(":", 1)
    tf = ROOT.TFile(path)
    if tf.IsZombie():
        raise SystemExit(f"--th3: cannot open {path}")
    h = tf.Get(name)
    if not h:
        tf.Close()
        raise SystemExit(f"--th3: no histogram '{name}' in {path}")
    nx, ny, nz = h.GetNbinsX(), h.GetNbinsY(), h.GetNbinsZ()
    edges = []
    for ax in (h.GetXaxis(), h.GetYaxis(), h.GetZaxis()):
        n = ax.GetNbins()
        edges.append(np.array([ax.GetBinLowEdge(i + 1) for i in range(n)]
                              + [ax.GetBinUpEdge(n)], dtype=float))
    counts = np.array([[[h.GetBinContent(i + 1, j + 1, k + 1)
                         for k in range(nz)]
                        for j in range(ny)]
                       for i in range(nx)], dtype=float)
    tf.Close()
    print(f"[th3] {path}:{name}  grid={nx}x{ny}x{nz}")
    return counts, edges


def _edges_from_centers(c, axis):
    """Bin edges implied by sorted unique centers.

    Interior edges are midpoints, which is exact. The two OUTER edges have
    to be guessed by reflecting the first and last spacing -- fine for a
    uniform axis, a guess otherwise. Use --csv-edges when the binning is
    non-uniform and the outer edges matter.
    """
    c = np.asarray(sorted(set(np.asarray(c, float))), dtype=float)
    if len(c) < 2:
        raise SystemExit(f"--csv: need >= 2 distinct {axis} values")
    mid = 0.5 * (c[:-1] + c[1:])
    lo = c[0] - (mid[0] - c[0])
    hi = c[-1] + (c[-1] - mid[-1])
    e = np.concatenate([[lo], mid, [hi]])
    if len(set(np.round(np.diff(e), 9))) > 1:
        print(f"[warn] {axis} centers are not uniformly spaced; the two "
              "OUTER edges are inferred by reflection. Use --csv-edges to "
              "state them exactly.")
    return c, e


def from_csv(path, cols):
    """Rows of (Q2, W, M, sigma) at bin centers -> (counts, edges)."""
    q, w, m, s = cols
    rows = {c: [] for c in cols}
    with open(path) as f:
        r = _csv.DictReader(f)
        missing = [c for c in cols if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"--csv {path}: missing {missing}; "
                             f"has {r.fieldnames}")
        for row in r:
            for c in cols:
                rows[c].append(float(row[c]))

    qc, qe = _edges_from_centers(rows[q], "Q2")
    wc, we = _edges_from_centers(rows[w], "W")
    mc, me = _edges_from_centers(rows[m], "M")
    counts = np.zeros((len(qc), len(wc), len(mc)), dtype=float)
    seen = np.zeros_like(counts, dtype=bool)
    qi = {v: i for i, v in enumerate(qc)}
    wi = {v: i for i, v in enumerate(wc)}
    mi = {v: i for i, v in enumerate(mc)}
    for a, b, c_, v in zip(rows[q], rows[w], rows[m], rows[s]):
        i, j, k = qi[a], wi[b], mi[c_]
        counts[i, j, k] = v
        seen[i, j, k] = True
    n_missing = int((~seen).sum())
    if n_missing:
        print(f"[warn] {n_missing}/{counts.size} grid cells had no row in "
              "the CSV and are set to 0 -- the generator will never populate "
              "them. Fill them in if that is not what you mean.")
    print(f"[csv] {path}  grid={counts.shape[0]}x{counts.shape[1]}"
          f"x{counts.shape[2]}  ({len(rows[q])} rows)")
    return counts, [qe, we, me]


def from_csv_edges(path, cols):
    """Rows with explicit lo/hi per axis -- exact for non-uniform binning."""
    need = list(cols)
    rows = {c: [] for c in need}
    with open(path) as f:
        r = _csv.DictReader(f)
        missing = [c for c in need if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"--csv-edges {path}: missing {missing}; "
                             f"has {r.fieldnames}")
        for row in r:
            for c in need:
                rows[c].append(float(row[c]))

    def axis_edges(lo_key, hi_key, label):
        pairs = sorted(set(zip(rows[lo_key], rows[hi_key])))
        e = [pairs[0][0]]
        for lo, hi in pairs:
            if abs(lo - e[-1]) > 1e-9:
                raise SystemExit(f"--csv-edges: {label} bins are not "
                                 f"contiguous at {lo} (expected {e[-1]})")
            e.append(hi)
        return pairs, np.array(e, dtype=float)

    qp, qe = axis_edges(cols[0], cols[1], "Q2")
    wp, we = axis_edges(cols[2], cols[3], "W")
    mp, me = axis_edges(cols[4], cols[5], "M")
    idx = ({p: i for i, p in enumerate(qp)},
           {p: i for i, p in enumerate(wp)},
           {p: i for i, p in enumerate(mp)})
    counts = np.zeros((len(qp), len(wp), len(mp)), dtype=float)
    for n in range(len(rows[cols[0]])):
        i = idx[0][(rows[cols[0]][n], rows[cols[1]][n])]
        j = idx[1][(rows[cols[2]][n], rows[cols[3]][n])]
        k = idx[2][(rows[cols[4]][n], rows[cols[5]][n])]
        counts[i, j, k] = rows[cols[6]][n]
    print(f"[csv-edges] {path}  grid={counts.shape[0]}x{counts.shape[1]}"
          f"x{counts.shape[2]}")
    return counts, [qe, we, me]


def main():
    ap = argparse.ArgumentParser(
        description="Convert a 3-D cross section into the weight builder's npz.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--th3", help="'file.root:histname', axes (Q2, W, M)")
    src.add_argument("--csv", help="rows of Q2,W,M,sigma at bin centers")
    src.add_argument("--csv-edges",
                     help="rows with explicit lo/hi per axis (exact for "
                          "non-uniform binning)")
    ap.add_argument("--csv-cols", default="Q2,W,M,sigma",
                    help="column names for --csv (default Q2,W,M,sigma)")
    ap.add_argument("--csv-edge-cols",
                    default="Q2_lo,Q2_hi,W_lo,W_hi,M_lo,M_hi,sigma",
                    help="column names for --csv-edges")
    ap.add_argument("--units", choices=("density", "integral"), required=True,
                    help="'density' if the values are dsigma/dQ2 dW dM; "
                         "'integral' if each value is already the cross "
                         "section integrated over its bin. Required -- the "
                         "numbers alone cannot tell you, and getting it "
                         "wrong skews every non-uniform bin.")
    ap.add_argument("--out", required=True, help="output npz")
    args = ap.parse_args()

    if args.th3:
        counts, edges = from_th3(args.th3)
    elif args.csv:
        cols = [c.strip() for c in args.csv_cols.split(",")]
        if len(cols) != 4:
            raise SystemExit("--csv-cols wants four names")
        counts, edges = from_csv(args.csv, cols)
    else:
        cols = [c.strip() for c in args.csv_edge_cols.split(",")]
        if len(cols) != 7:
            raise SystemExit("--csv-edge-cols wants seven names")
        counts, edges = from_csv_edges(args.csv_edges, cols)

    n_neg = int((counts < 0).sum())
    if n_neg:
        print(f"[warn] {n_neg} negative values clipped to 0 (a cross section "
              "cannot be negative, and an accept probability certainly not).")
        counts = np.clip(counts, 0.0, None)
    if counts.sum() <= 0:
        raise SystemExit("cross section is zero everywhere.")

    vol = np.ones_like(counts)
    for i, e in enumerate(edges):
        w = np.diff(e)
        shape = [1] * 3
        shape[i] = len(w)
        vol = vol * w.reshape(shape)
    print(f"[grid] Q2 {[round(float(v), 4) for v in edges[0]]}")
    print(f"[grid] W  {len(edges[1])-1} bins [{edges[1][0]:g},{edges[1][-1]:g}]"
          f"   M {len(edges[2])-1} bins [{edges[2][0]:g},{edges[2][-1]:g}]")
    print(f"[grid] bin-volume spread {vol.max()/vol.min():.2f}x  -> "
          f"units={args.units} "
          + ("matters here" if vol.max() / vol.min() > 1.001
             else "makes no difference on a uniform grid"))

    np.savez(args.out, counts=counts,
             edges=np.array(edges, dtype=object),
             varnames=np.array(["Q2", "W", "M"]),
             units=args.units)
    print(f"[write] {args.out}")
    print(f"\nNext:\n"
          f"    python build_xsec_weight3d.py --xsec {args.out} \\\n"
          f"        --gen gen_truth_unweighted.root --scan --out /dev/null")


if __name__ == "__main__":
    main()
