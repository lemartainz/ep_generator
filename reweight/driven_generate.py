"""
Data-driven driver for runEventGenerator.cpp
--------------------------------------------

Instead of reweighting a finished LUND file, this script reshapes the
generator's *input distribution* so that the raw output already matches
real data. It does so WITHOUT modifying runEventGenerator.cpp.

Strategy
========
1. Bin real data in the electron kinematic variables that the generator
   samples uniformly: (Q^2, E', theta_e).  These are the three ranges in
   input.txt — Q2_range, E_range, theta_range.
2. For each 3-D cell, count how many real events fall in it.
3. For each nonempty cell, write a temporary input.txt where Q2_range,
   E_range and theta_range are restricted to that single cell's edges,
   and num_events is proportional to the real-data count in that cell.
4. Run `root -l -b -q runEventGenerator.cpp` once per cell, renaming the
   LUND output to a per-cell file.
5. Concatenate all per-cell LUND files into a single output LUND.

Because the generator samples uniformly *within* each cell, the
piecewise-uniform ensemble approximates the real-data density as
bin widths shrink. It is the generator analog of a histogram sampler.

Other input keys (reaction, t_slope, target, beam_energy, W_min...)
are taken from a base input file that you provide.

Usage
=====
    python driven_generate.py \\
        --base      ../input.txt \\
        --data      real_data.csv \\
        --total     200000 \\
        --q2-bins   "1,6,20" \\
        --e-bins    "1,9,20" \\
        --theta-bins "5,35,20" \\
        --out       ../events_datadriven.lund \\
        --workdir   ./_gen_tmp \\
        --generator ../runEventGenerator.cpp

`real_data.csv` must have columns named `Q2`, `Ep`, `theta_e` (degrees),
matching the names in kinematics.py.

Notes
=====
- `--total` is the target total number of generated events. Cell yields
  are rounded, so the actual total may differ by a few tens.
- Cells with zero real-data counts are skipped entirely.
- If a cell's data count is very small, you can set `--min-per-cell`
  to force at least that many events per nonempty cell (useful when you
  want decent statistics everywhere).
- W_min from the base input is still enforced; cells that produce zero
  events (because their kinematics are below W_min) simply yield empty
  LUND pieces, which is fine.
- Cell binning should be fine enough that uniform-within-cell is a good
  approximation, but coarse enough that each cell has enough data events
  to be meaningful. 20 bins per axis is a reasonable starting point.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import numpy as np


def parse_bins(spec):
    lo, hi, n = spec.split(",")
    return np.linspace(float(lo), float(hi), int(n) + 1)


def load_data(path, cols=("Q2", "Ep", "theta_e")):
    out = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        missing = [c for c in cols if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"real_data.csv is missing columns: {missing}. "
                             f"Has: {r.fieldnames}")
        for row in r:
            for c in cols:
                out[c].append(float(row[c]))
    return np.stack([np.asarray(out[c]) for c in cols], axis=1)


def read_base_input(path):
    """Return a dict of key -> raw string value, preserving order."""
    entries = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                entries.append(("__raw__", line.rstrip("\n")))
                continue
            if ":" not in s:
                entries.append(("__raw__", line.rstrip("\n")))
                continue
            key, _, val = s.partition(":")
            entries.append((key.strip(), val.strip()))
    return entries


def write_input(entries, path, overrides):
    with open(path, "w") as f:
        seen = set()
        for k, v in entries:
            if k == "__raw__":
                f.write(v + "\n")
                continue
            if k in overrides:
                f.write(f"{k}: {overrides[k]}\n")
                seen.add(k)
            else:
                f.write(f"{k}: {v}\n")
        for k, v in overrides.items():
            if k not in seen:
                f.write(f"{k}: {v}\n")


def run_generator(generator_cpp, workdir, log_path):
    """
    Run `root -l -b -q runEventGenerator.cpp` inside workdir.
    input.txt and runEventGenerator.cpp must already be in workdir.
    Returns path to produced LUND (default: events.lund in workdir).
    """
    cmd = ["root", "-l", "-b", "-q", os.path.basename(generator_cpp)]
    with open(log_path, "a") as logf:
        logf.write(f"\n### {cmd} (cwd={workdir})\n")
        logf.flush()
        res = subprocess.run(cmd, cwd=workdir, stdout=logf, stderr=logf)
    if res.returncode != 0:
        print(f"  ! generator returned {res.returncode}, see {log_path}")
    produced = os.path.join(workdir, "events.lund")
    return produced if os.path.exists(produced) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",      required=True, help="Base input.txt")
    ap.add_argument("--data",      required=True, help="Real-data CSV")
    ap.add_argument("--generator", required=True, help="runEventGenerator.cpp path")
    ap.add_argument("--out",       required=True, help="Final concatenated LUND")
    ap.add_argument("--workdir",   default="./_gen_tmp")
    ap.add_argument("--total",     type=int, default=100000,
                    help="Approximate total events to generate")
    ap.add_argument("--q2-bins",    required=True, help='"lo,hi,n"')
    ap.add_argument("--e-bins",     required=True, help='"lo,hi,n"')
    ap.add_argument("--theta-bins", required=True, help='"lo,hi,n"')
    ap.add_argument("--min-per-cell", type=int, default=0,
                    help="Force at least this many events in nonempty cells")
    args = ap.parse_args()

    q2_edges    = parse_bins(args.q2_bins)
    e_edges     = parse_bins(args.e_bins)
    theta_edges = parse_bins(args.theta_bins)

    data = load_data(args.data)            # (N, 3) = (Q2, Ep, theta_e)
    H, _ = np.histogramdd(data, bins=[q2_edges, e_edges, theta_edges])
    total_data = H.sum()
    if total_data == 0:
        raise SystemExit("No real-data events fell inside the specified bins.")
    density = H / total_data                # fraction per cell

    # target events per cell
    n_cells = np.round(density * args.total).astype(int)
    if args.min_per_cell > 0:
        n_cells = np.where(H > 0, np.maximum(n_cells, args.min_per_cell), n_cells)
    planned_total = int(n_cells.sum())
    nonempty = int((n_cells > 0).sum())
    print(f"[driver] bins={H.shape}  nonempty cells={nonempty}  "
          f"planned total={planned_total}")

    # prepare workdir
    os.makedirs(args.workdir, exist_ok=True)
    gen_dst = os.path.join(args.workdir, os.path.basename(args.generator))
    shutil.copy2(args.generator, gen_dst)

    base_entries = read_base_input(args.base)
    log_path = os.path.join(args.workdir, "generator.log")
    open(log_path, "w").close()

    # iterate cells and dispatch jobs
    produced_files = []
    cell_id = 0
    for i in range(len(q2_edges) - 1):
        for j in range(len(e_edges) - 1):
            for k in range(len(theta_edges) - 1):
                n = int(n_cells[i, j, k])
                if n <= 0:
                    continue
                cell_id += 1
                overrides = {
                    "num_events":  str(n),
                    "Q2_range":    f"{q2_edges[i]:.6f} {q2_edges[i+1]:.6f}",
                    "E_range":     f"{e_edges[j]:.6f} {e_edges[j+1]:.6f}",
                    "theta_range": f"{theta_edges[k]:.6f} {theta_edges[k+1]:.6f}",
                    "write_lund":  "1",
                    "gen_plots":   "0",
                    "print_debug": "0",
                }
                write_input(base_entries,
                            os.path.join(args.workdir, "input.txt"),
                            overrides)
                # remove any stale events.lund from previous cell
                stale = os.path.join(args.workdir, "events.lund")
                if os.path.exists(stale):
                    os.remove(stale)

                out = run_generator(args.generator, args.workdir, log_path)
                if out is None:
                    print(f"  cell {cell_id} ({i},{j},{k}) n={n}: no output")
                    continue
                cell_out = os.path.join(
                    args.workdir, f"cell_{i:02d}_{j:02d}_{k:02d}.lund")
                shutil.move(out, cell_out)
                produced_files.append(cell_out)
                if cell_id % 25 == 0:
                    print(f"  ...{cell_id} cells done")

    # concatenate
    print(f"[driver] concatenating {len(produced_files)} LUND pieces -> {args.out}")
    with open(args.out, "wb") as wf:
        for p in produced_files:
            with open(p, "rb") as rf:
                shutil.copyfileobj(rf, wf)
    print(f"[driver] done: {args.out}")


if __name__ == "__main__":
    main()
