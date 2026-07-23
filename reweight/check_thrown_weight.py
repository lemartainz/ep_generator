"""
Thrown-level validation of the reco-matching weight, BEFORE GEMC.

Checks (see the analysis notes):
  0. weight_func.root really is  N_sub/N_rec  (matches the npzs).
  1. thrown_weighted / thrown_unweighted  ==  weight surface w.
  2. thrown_weighted  ~  N_sub / eps   (eps = N_rec / thrown_unweighted).
  4. expected accept-reject keep fraction.
"""
import argparse, csv, numpy as np, uproot


def load_csv_xy(path, xcol="Q2", ycol="Ep"):
    X, Y = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            X.append(float(r[xcol])); Y.append(float(r[ycol]))
    return np.asarray(X), np.asarray(Y)


def hist_on(x, y, xe, ye):
    fin = np.isfinite(x) & np.isfinite(y)
    H, _, _ = np.histogram2d(x[fin], y[fin], bins=[xe, ye])
    return H


def corr(a, b, m):
    a, b = a[m], b[m]
    if a.size < 2: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unw", required=True, help="unweighted thrown CSV (Q2,Ep)")
    ap.add_argument("--wtd", required=True, help="weighted thrown CSV (Q2,Ep)")
    ap.add_argument("--wfile", required=True, help="weight_func.root")
    ap.add_argument("--wname", default="w_Q2_Ep")
    ap.add_argument("--sub", required=True, help="subtracted_q2ep.npz (N_sub)")
    ap.add_argument("--rec", required=True, help="rec_q2ep.npz (N_rec)")
    ap.add_argument("--mincount", type=float, default=20,
                    help="min thrown counts per bin to include in comparisons")
    args = ap.parse_args()

    # weight surface from the ROOT file (the thing the generator actually used)
    h = uproot.open(args.wfile)[args.wname]
    W = h.values()
    xe = h.axis(0).edges(); ye = h.axis(1).edges()

    # npzs
    sub = np.load(args.sub, allow_pickle=True)
    rec = np.load(args.rec, allow_pickle=True)
    N_sub = np.clip(np.asarray(sub["counts"], float), 0.0, None)
    N_rec = np.asarray(rec["counts"], float)
    se = [np.asarray(e, float) for e in sub["edges"]]
    if not (np.allclose(se[0], xe) and np.allclose(se[1], ye)):
        raise SystemExit("npz edges != weight_func edges")

    # ---- Check 0: is weight_func.root == normalized N_sub/N_rec? ----
    Dn = N_sub / N_sub.sum()
    Rn = N_rec / N_rec.sum()
    W_expect = np.divide(Dn, Rn, out=np.zeros_like(Dn), where=Rn > 0)
    W_expect /= W_expect.max()
    both = (W > 0) & (W_expect > 0)
    print(f"[0] weight_func vs N_sub/N_rec : corr={corr(W, W_expect, both):.4f}  "
          f"max|diff|={np.max(np.abs(W - W_expect)[both]):.3e}  "
          f"(nonzero bins: file={int((W>0).sum())}, expected={int((W_expect>0).sum())})")

    # ---- thrown histograms ----
    ux, uy = load_csv_xy(args.unw)
    wx, wy = load_csv_xy(args.wtd)
    Hu = hist_on(ux, uy, xe, ye)
    Hw = hist_on(wx, wy, xe, ye)
    print(f"    thrown: unweighted N={int(Hu.sum())}, weighted N={int(Hw.sum())}")

    # ---- Check 1: Hw/Hu ~ W ----
    m = (Hu >= args.mincount) & (W > 0)
    ratio = np.zeros_like(Hu)
    ratio[m] = (Hw[m] / Hw.sum()) / (Hu[m] / Hu.sum())
    if ratio[m].max() > 0:
        ratio_n = ratio / ratio[m].max()
    else:
        ratio_n = ratio
    print(f"[1] (Hw/Hu) vs W : corr={corr(ratio_n, W, m):.4f}  "
          f"median(ratio_n/W)={np.median((ratio_n[m]/W[m])[W[m]>0]):.3f}  "
          f"(bins compared: {int(m.sum())})")

    # ---- Check 2: Hw ~ N_sub/eps,  eps = N_rec/Hu ----
    me = (Hu >= args.mincount) & (N_rec > 0)
    eps = np.zeros_like(Hu); eps[me] = N_rec[me] / Hu[me]
    pred = np.zeros_like(Hu); pred[me] = N_sub[me] / eps[me]      # = N_sub*Hu/N_rec
    m2 = me & (pred > 0)
    print(f"[2] Hw vs N_sub/eps : corr={corr(Hw, pred, m2):.4f}  (bins: {int(m2.sum())})")

    # ---- Check 4: expected keep fraction ----
    p_thrown = Hu / Hu.sum()
    keep = float(np.sum(p_thrown * W))
    print(f"[4] expected accept-reject keep fraction ~ sum(p_thrown * W) = {keep:.4f}  "
          f"(mean W over grid = {W.mean():.4f})")

    # ---- 1-D projections for eyeballing ----
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    def proj(H, ax): return H.sum(axis=ax)
    print("\n[Q2 projection] col=Q2  unw  wtd  N_sub/eps(pred)")
    Pu, Pw, Pp = proj(Hu,1), proj(Hw,1), proj(pred,1)
    for i in range(0, len(xc), 10):
        print(f"  Q2={xc[i]:4.2f}  {Pu[i]:7.0f} {Pw[i]:7.0f} {Pp[i]:9.1f}")


if __name__ == "__main__":
    main()
