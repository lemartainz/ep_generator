"""
Visualize a weight TH2D exactly as the generator sees it.

Reads the saved ROOT file, pulls the TH2D by name, and draws the surface
(plus its 1-D projections) with matplotlib. Use it to confirm a weight is
generator-ready before feeding it to runEventGenerator:

    * values in [0, 1] with max == 1   (accept-reject probability)
    * no unexpected w == 0 holes       (empty-MC-bin rejection traps)
    * the ridge/structure you expect from the data

Usage:
    python plot_weight.py weight_func.root w_Q2_Ep
    python plot_weight.py mom_weight.root  w_pp
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import ROOT


def th2_to_numpy(h):
    """Return (W[nx,ny], x_edges, y_edges) from a TH2D (no under/overflow)."""
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    W = np.array([[h.GetBinContent(ix + 1, iy + 1) for iy in range(ny)]
                  for ix in range(nx)])
    xa, ya = h.GetXaxis(), h.GetYaxis()
    x_edges = np.array([xa.GetBinLowEdge(i + 1) for i in range(nx)]
                       + [xa.GetBinUpEdge(nx)])
    y_edges = np.array([ya.GetBinLowEdge(i + 1) for i in range(ny)]
                       + [ya.GetBinUpEdge(ny)])
    return W, x_edges, y_edges


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "weight_func.root"
    name = sys.argv[2] if len(sys.argv) > 2 else "w_Q2_Ep"

    tf = ROOT.TFile.Open(path, "READ")
    h = tf.Get(name)
    if not h:
        raise SystemExit(f"TH2D '{name}' not found in {path}")

    W, xe, ye = th2_to_numpy(h)
    # quick sanity numbers — the same properties the generator relies on
    print(f"{path}:{name}  grid={W.shape[0]}x{W.shape[1]}  "
          f"max={W.max():.3f}  mean={W.mean():.3f}  "
          f"zero-fraction={np.mean(W <= 0):.3f}")

    xt = h.GetXaxis().GetTitle() or "x"
    yt = h.GetYaxis().GetTitle() or "y"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # main surface — transpose so x is horizontal, origin lower-left
    im = ax[0].pcolormesh(xe, ye, W.T, cmap="viridis", vmin=0, vmax=1)
    ax[0].set(xlabel=xt, ylabel=yt, title=f"w = d/g  ({name})")
    fig.colorbar(im, ax=ax[0], label="accept prob w")

    # 1-D projections: average weight vs each axis (a rough shape check)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ax[1].plot(xc, W.mean(axis=1), drawstyle="steps-mid")
    ax[1].set(xlabel=xt, ylabel="mean w", title="mean weight vs x")
    ax[2].plot(yc, W.mean(axis=0), drawstyle="steps-mid")
    ax[2].set(xlabel=yt, ylabel="mean w", title="mean weight vs y")

    fig.tight_layout()
    out = path.replace(".root", f"_{name}.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
