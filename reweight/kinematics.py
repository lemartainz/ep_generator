"""
Per-event kinematics using the `vector` library, with the same
conventions as /ppbar/Analysis/ppbar_sim.ipynb:

    p_beam   = vec.obj(px=0, py=0, pz=E_BEAM, M=m_e)
    p_target = vec.obj(px=0, py=0, pz=0,      M=m_p)
    p_q      = p_beam - p_e
    Q2       = -p_q**2
    p_w      = p_q + p_target
    W        = p_w.mass

Two entry points are provided:

    compute_kinematics(ev)       -> dict for a SINGLE Event
    compute_kinematics_batch(path, extra=None)
                                 -> dict of numpy arrays for ALL events

`compute_kinematics_batch` is vectorized via `vector.array` and is the
fast path used by build_weights.py. It also returns a missing-mass
(MM = beam + target - p1 - p2 - e) when the event contains two protons
and one electron — matching the notebook's `p_miss` definition.

Requires: `pip install vector numpy`
"""

import numpy as np
import vector as vec

from lund_io import read_lund

# Keep in sync with PID_helpers.PDG_MASS / E_BEAM in the ppbar analysis.
M_P = 0.938272
M_E = 0.000511


# ------------------------------------------------------------------ #
# Single-event (used by reweight_lund.py in streaming mode)
# ------------------------------------------------------------------ #
def compute_kinematics(ev):
    """Return a dict of kinematic scalars for one lund_io.Event."""
    Ebeam = ev.beam_energy
    p_beam   = vec.obj(px=0.0, py=0.0, pz=Ebeam, M=M_E)
    p_target = vec.obj(px=0.0, py=0.0, pz=0.0,   M=M_P)

    # scattered electron = the pid==11 entry
    eprime_part = None
    protons = []
    for p in ev.particles:
        if p.pid == 11:
            eprime_part = p
        elif p.pid == 2212 or p.pid == -2212:
            protons.append(p)
    if eprime_part is None:
        raise ValueError("No scattered electron (pid=11) in event")

    p_e = vec.obj(px=eprime_part.px, py=eprime_part.py,
                  pz=eprime_part.pz, M=M_E)

    p_q = p_beam - p_e
    p_w = p_q + p_target
    Q2  = -(p_q.M2)
    W   = p_w.mass
    nu  = p_q.E
    xB  = Q2 / (2.0 * M_P * nu) if nu > 0 else 0.0
    y   = nu / Ebeam

    out = {
        "Q2": Q2,
        "W":  W,
        "xB": xB,
        "y":  y,
        "nu": nu,
        "theta_e": np.degrees(p_e.theta),
        "phi_e":   np.degrees(p_e.phi),
        "Ep":      p_e.E,
    }

    # Optional: missing mass using first two protons (matches notebook's p_miss)
    if len(protons) >= 2:
        p1 = vec.obj(px=protons[0].px, py=protons[0].py, pz=protons[0].pz, M=M_P)
        p2 = vec.obj(px=protons[1].px, py=protons[1].py, pz=protons[1].pz, M=M_P)
        p_miss = p_beam + p_target - p1 - p2 - p_e
        out["MM"]      = p_miss.mass
        out["Mppbar"]  = (p1 + p2).mass  # ppbar invariant mass
    return out


# ------------------------------------------------------------------ #
# Vectorized batch loader (used by build_weights.py / dump_kinematics.py)
# ------------------------------------------------------------------ #
def compute_kinematics_batch(path):
    """
    Stream a LUND file and build numpy arrays of every kinematic
    quantity using `vector.array` for speed.
    """
    px_e,  py_e,  pz_e  = [], [], []
    px_p1, py_p1, pz_p1 = [], [], []
    px_p2, py_p2, pz_p2 = [], [], []
    has_pp = []
    Ebeam_arr = []

    for ev in read_lund(path):
        Ebeam_arr.append(ev.beam_energy)
        e = None
        protons = []
        for p in ev.particles:
            if p.pid == 11:
                e = p
            elif p.pid == 2212 or p.pid == -2212:
                protons.append(p)
        px_e.append(e.px); py_e.append(e.py); pz_e.append(e.pz)
        if len(protons) >= 2:
            px_p1.append(protons[0].px); py_p1.append(protons[0].py); pz_p1.append(protons[0].pz)
            px_p2.append(protons[1].px); py_p2.append(protons[1].py); pz_p2.append(protons[1].pz)
            has_pp.append(True)
        else:
            px_p1.append(0.0); py_p1.append(0.0); pz_p1.append(0.0)
            px_p2.append(0.0); py_p2.append(0.0); pz_p2.append(0.0)
            has_pp.append(False)

    Ebeam = np.asarray(Ebeam_arr)
    n = len(Ebeam)
    ones = np.ones(n)

    p_beam = vec.array({"px": np.zeros(n), "py": np.zeros(n),
                        "pz": Ebeam,       "M":  ones * M_E})
    p_target = vec.array({"px": np.zeros(n), "py": np.zeros(n),
                          "pz": np.zeros(n), "M": ones * M_P})
    p_e = vec.array({"px": np.asarray(px_e), "py": np.asarray(py_e),
                     "pz": np.asarray(pz_e), "M":  ones * M_E})

    p_q = p_beam - p_e
    p_w = p_q + p_target
    Q2  = -(p_q.M2)
    W   = p_w.mass
    nu  = p_q.E
    xB  = np.where(nu > 0, Q2 / (2.0 * M_P * nu), 0.0)
    y   = nu / Ebeam

    out = {
        "Q2": Q2,
        "W":  W,
        "xB": xB,
        "y":  y,
        "nu": nu,
        "theta_e": np.degrees(p_e.theta),
        "phi_e":   np.degrees(p_e.phi),
        "Ep":      p_e.E,
    }

    has_pp = np.asarray(has_pp)
    if has_pp.any():
        p_p1 = vec.array({"px": np.asarray(px_p1), "py": np.asarray(py_p1),
                          "pz": np.asarray(pz_p1), "M": ones * M_P})
        p_p2 = vec.array({"px": np.asarray(px_p2), "py": np.asarray(py_p2),
                          "pz": np.asarray(pz_p2), "M": ones * M_P})
        p_miss  = p_beam + p_target - p_p1 - p_p2 - p_e
        out["MM"]     = np.where(has_pp, p_miss.mass, np.nan)
        out["Mppbar"] = np.where(has_pp, (p_p1 + p_p2).mass, np.nan)

    return out
