# Data-driven reweighting for the ep event generator

These scripts sit **alongside** `runEventGenerator.cpp` — they do not modify
the generator. The workflow is:

1. Run the generator as usual → `events.lund`
2. Prepare a CSV of your **real data** with columns for the kinematic
   variables you want to match (e.g. `Q2,W`).
3. Build a weight table (data/MC ratio in an N-D histogram).
4. Apply the weights to produce either a **reweighted LUND**
   (accept-reject, unweighted output — use directly in GEMC) or a
   **sidecar weight file** (original LUND + one weight per event).

All scripts are pure Python (numpy only). Run them from the `reweight/`
directory so the `lund_io` / `kinematics` imports resolve.

## Files

| Script | Purpose |
|---|---|
| `lund_io.py` | Stream-read/write LUND events |
| `kinematics.py` | Compute Q², W, xB, y, θₑ, φₑ, Eₑ′ per event |
| `dump_kinematics.py` | LUND → CSV of kinematics (for inspection / cross-checks) |
| `build_weights.py` | Build `weights.npz` from data CSV + MC LUND |
| `reweight_lund.py` | Apply weights: accept-reject or sidecar |

## Example 1 — 1-D reweight in Q²

```bash
cd reweight

# (optional) inspect MC kinematics
python dump_kinematics.py --in ../events.lund --out mc_kin.csv

# build weights: 25 Q² bins from 1 to 6 GeV²
python build_weights.py \
    --mc   ../events.lund \
    --data real_data.csv \
    --vars Q2 \
    --bins "1,6,25" \
    --out  weights_Q2.npz

# accept-reject → a new unweighted LUND that matches real-data Q²
python reweight_lund.py \
    --in ../events.lund \
    --weights weights_Q2.npz \
    --out ../events_reweighted.lund \
    --mode accept --seed 42
```

`real_data.csv` must have a header row and at least a `Q2` column:

```
Q2,W
2.31,2.10
1.88,1.95
...
```

## Example 2 — 2-D reweight in (Q², W)

```bash
python build_weights.py \
    --mc   ../events.lund \
    --data real_data.csv \
    --vars Q2 W \
    --bins "1,6,25" "1.6,3.2,20" \
    --out  weights_Q2W.npz

python reweight_lund.py \
    --in ../events.lund \
    --weights weights_Q2W.npz \
    --out ../events_reweighted.lund
```

## Example 3 — 3-D reweight in (Q², xB, W)

```bash
python build_weights.py \
    --mc   ../events.lund \
    --data real_data.csv \
    --vars Q2 xB W \
    --bins "1,6,20" "0.1,0.7,15" "1.6,3.2,15" \
    --wmax 30 \
    --out  weights_3d.npz

python reweight_lund.py \
    --in ../events.lund --weights weights_3d.npz \
    --out ../events_reweighted.lund
```

## Example 4 — Sidecar mode (keep all events, carry per-event weight)

Use this if your analysis handles weighted events directly:

```bash
python reweight_lund.py \
    --in ../events.lund \
    --weights weights_Q2W.npz \
    --out ../events_weighted_copy.lund \
    --weights-out ../events_weights.txt \
    --mode sidecar
```

`events_weights.txt` has one float per line, in the same order as the
events in the LUND file.

## Notes

- **Normalization.** `build_weights.py` normalizes the table so that the
  mean weight over the MC sample is 1. Accept-reject will therefore
  keep roughly `N_MC / wmax` events. If that fraction is too low, use
  coarser bins or a lower `--wmax` clip.
- **Out-of-range events** (kinematics outside the weight-table edges)
  get weight 0 and are dropped by accept-reject. Widen your binning if
  you want to keep them.
- **Adding new weighting variables.** Just add them to
  `kinematics.py → compute_kinematics()`. No other file needs to change.
- **Source of `real_data.csv`.** Anything that yields a CSV with the
  right column names works — a small ROOT macro dumping a TTree with
  `TTree::Scan`/`ROOT::RDataFrame::Snapshot`, a pandas script, uproot,
  etc. Only the columns named in `--vars` are read.
- **Closure test.** Dump kinematics from the reweighted LUND and
  compare to your real data:

  ```bash
  python dump_kinematics.py --in ../events_reweighted.lund --out rw_kin.csv
  # then plot rw_kin.csv vs real_data.csv in your favorite tool
  ```
