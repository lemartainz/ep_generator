"""
Read the generator's truth ntuple (`truth_ntuple:` in the input card) and
form pair invariant masses from its final-state block.

The ntuple holds, per ACCEPTED event, the scalar kinematics (Q2, W, M, Ep,
theta_e) and a fixed-capacity final-state block:

    nfs                  number of final-state particles stored
    fs_pid[8]            PDG codes (0 past nfs)
    fs_px/py/pz/E[8]     lab 4-vectors [GeV]

Fixed-size branches read back as plain (N, 8) numpy arrays, so nothing here
needs awkward. Shared by build_pair_weight.py and plot_xsec_closure.py.
"""

import os

import numpy as np

FS_CAP = 8
SCALARS = ("Q2", "W", "M", "Ep", "theta_e")
FS_BRANCHES = ("nfs", "fs_pid", "fs_px", "fs_py", "fs_pz", "fs_E")


def load_truth(path, tree="truth", final_state=False):
    """Return a dict of numpy arrays for the scalar branches, plus the
    final-state block when `final_state` is set.

    `path` may be the generator's truth ntuple (.root) or its LUND file: the
    LUND holds the same final-state 4-vectors and the scattered electron, so
    everything that does not need the truth pairing (M_X) can be rebuilt
    from it -- Q2, W, E' and every pair mass. Only `M` is then NaN.
    """
    if path.endswith(".lund"):
        return load_lund(path)
    import uproot
    if not os.path.exists(path):
        raise SystemExit(f"truth ntuple not found: {path}")
    t = uproot.open(path)[tree]
    have = set(t.keys())
    want = [b for b in SCALARS if b in have]
    if final_state:
        missing = [b for b in FS_BRANCHES if b not in have]
        if missing:
            raise SystemExit(
                f"{path}:{tree} has no final-state block ({missing} missing). "
                "Re-run the generator: the block was added to the truth "
                "ntuple together with the pair_weight stage.")
        want += list(FS_BRANCHES)
    return t.arrays(want, library="np")


M_E = 0.000510999


def load_lund(path):
    """Read a LUND file into the same dict layout as the truth ntuple.

    Header:  nparticles 1 1 0 0 11 beam_energy target_pid target_mass 0
    Particle: idx lifetime type pid parent daughter px py pz E mass vx vy vz

    Q2 and W come from the pid-11 electron against the beam (header) and
    the target at rest; M (the truth M_X) is not recoverable and is NaN.
    A plain token scan -- ~10 s per million events -- no ROOT needed.
    """
    if not os.path.exists(path):
        raise SystemExit(f"LUND file not found: {path}")
    n_ev, cap = 0, FS_CAP
    nfs, ebeam, mtgt = [], [], []
    pid, px, py, pz, E = [], [], [], [], []
    cur = 0
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if len(t) == 10:                       # event header
                if n_ev:
                    nfs.append(cur)
                n_ev += 1
                cur = 0
                ebeam.append(float(t[6]))
                mtgt.append(float(t[8]))
                pid.append([0] * cap); px.append([0.0] * cap); py.append([0.0] * cap)
                pz.append([0.0] * cap); E.append([0.0] * cap)
            else:                                  # particle line
                if cur < cap:
                    pid[-1][cur] = int(t[3])
                    px[-1][cur] = float(t[6]); py[-1][cur] = float(t[7])
                    pz[-1][cur] = float(t[8]); E[-1][cur] = float(t[9])
                cur += 1
        if n_ev:
            nfs.append(cur)
    if not n_ev:
        raise SystemExit(f"{path}: no events")
    a = {"nfs": np.asarray(nfs), "fs_pid": np.asarray(pid), "fs_px": np.asarray(px),
         "fs_py": np.asarray(py), "fs_pz": np.asarray(pz), "fs_E": np.asarray(E)}
    if (a["nfs"] > cap).any():
        print(f"[lund] WARNING: events with more than {cap} particles; extra "
              "ones ignored")
    eb, mt = np.asarray(ebeam), np.asarray(mtgt)
    # scattered electron: first pid == 11
    is_e = (a["fs_pid"] == 11) & (np.arange(cap)[None, :] < a["nfs"][:, None])
    has_e = is_e.any(axis=1)
    ie = np.where(has_e, is_e.argmax(axis=1), 0)
    rows = np.arange(n_ev)
    ex, ey, ez, ee = (a[k][rows, ie] for k in ("fs_px", "fs_py", "fs_pz", "fs_E"))
    qE, qz = eb - ee, np.sqrt(np.maximum(eb ** 2 - M_E ** 2, 0.0)) - ez
    q2 = -(qE ** 2 - ex ** 2 - ey ** 2 - qz ** 2)
    w2 = (qE + mt) ** 2 - ex ** 2 - ey ** 2 - qz ** 2
    a["Q2"] = np.where(has_e, q2, np.nan)
    a["W"] = np.where(has_e, np.sqrt(np.maximum(w2, 0.0)), np.nan)
    a["Ep"] = np.where(has_e, ee, np.nan)
    a["M"] = np.full(n_ev, np.nan)
    a["theta_e"] = np.where(has_e, np.degrees(np.arctan2(np.hypot(ex, ey), ez)), np.nan)
    print(f"[lund] {path}: {n_ev} events ({int((~has_e).sum())} without a scattered "
          "electron)")
    return a


def pair_masses(a, pid_a, pid_b):
    """Invariant mass of every (pid_a, pid_b) pair in every event.

    Mirrors EventWeighter::acceptPairs exactly: each unordered pair once when
    the species are the same, each (A, B) combination once otherwise.

    Returns (event_index, mass) as flat arrays, one entry per pair, sorted by
    event; and `per_event`, the number of pairs each event contributed
    (0 for events without such a pair).
    """
    n = np.asarray(a["nfs"])
    pid = np.asarray(a["fs_pid"])
    E, px, py, pz = (np.asarray(a[k], float) for k in ("fs_E", "fs_px", "fs_py", "fs_pz"))
    N, cap = pid.shape
    valid = np.arange(cap)[None, :] < n[:, None]
    is_a = valid & (pid == pid_a)
    is_b = valid & (pid == pid_b)

    ev_list, m_list = [], []
    for i in range(cap):
        j0 = i + 1 if pid_a == pid_b else 0
        for j in range(j0, cap):
            if j == i:
                continue
            sel = is_a[:, i] & is_b[:, j]
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            e = E[idx, i] + E[idx, j]
            p2 = ((px[idx, i] + px[idx, j]) ** 2 + (py[idx, i] + py[idx, j]) ** 2
                  + (pz[idx, i] + pz[idx, j]) ** 2)
            ev_list.append(idx)
            m_list.append(np.sqrt(np.clip(e * e - p2, 0.0, None)))
    if not ev_list:
        return (np.zeros(0, int), np.zeros(0, float), np.zeros(N, int))
    ev = np.concatenate(ev_list)
    m = np.concatenate(m_list)
    order = np.argsort(ev, kind="stable")
    ev, m = ev[order], m[order]
    per_event = np.bincount(ev, minlength=N)
    return ev, m, per_event


def parse_pids(spec):
    """'2212,-2212' -> (2212, -2212)."""
    try:
        a, b = (int(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(f"bad species pair {spec!r}; expected e.g. 2212,-2212")
    return a, b


def pair_label(pid_a, pid_b):
    names = {2212: "p", -2212: "pbar", 211: "pi+", -211: "pi-", 111: "pi0",
             321: "K+", -321: "K-", 22: "gamma", 11: "e-", -11: "e+",
             2112: "n", -2112: "nbar"}
    return f"{names.get(pid_a, pid_a)} {names.get(pid_b, pid_b)}"
