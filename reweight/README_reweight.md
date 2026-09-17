# Data-driven reweighting for the ep event generator

These scripts sit **alongside** `runEventGenerator.cpp` — they do not modify
the generator. On the generator side every weight surface is loaded and
applied by `EventWeighter.h` (a header-only class the macro includes);
the generation code itself never touches a weight. The workflow is:

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
| `xsec.py` | Cross-section models: formulas, callables, tables, Jacobians |
| `build_xsec_weight.py` | Cross section &rarr; `weight_func.root` for the generator |
| `build_weight_func.py` | Data/MC ratio &rarr; `weight_func.root` for the generator |
| `plot_weight.py` | Draw a weight TH2D as the generator sees it |
| `make_pseudo_xsec.py` | Pseudo 3-D &sigma;(Q&sup2;, W, M) on a bin grid &rarr; multi-page PDF + npz |
| `import_xsec.py` | Your real &sigma;(Q&sup2;, W, M) (TH3 / CSV) &rarr; the npz the builder reads |
| `build_xsec_weight3d.py` | 3-D &sigma;(Q&sup2;, W, M) &rarr; `xsec_weight.root` (TH3D) for the generator |
| `plot_xsec_closure.py` | Generated vs cross section with ratio panels &rarr; closure PDF |
| `build_dsdt_table.py` | d&sigma;/dt(s, t) &rarr; `dsdt_table.root` (TH2D) for the generator's carried ratio weight |
| `plot_ratio_weight.py` | t distribution with / without the carried ratio weight, and &lang;w_ratio&rang;(t) |

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

## Cross-section weighting (no generator run needed)

`build_xsec_weight.py` is the standalone path: give it a **cross section**
and it writes the weight surface the generator reads. It never runs the
generator, never needs a LUND file, and never needs real data.

```bash
cd reweight
python build_xsec_weight.py \
    --input-card ../input.txt \
    --formula "Gamma * exp(-2.0 * (W - 2.85))" \
    --out ../weight_func.root
```

Then add one line to `input.txt` and run the generator as usual:

```
weight_func: weight_func.root w_Q2_Ep
```

`EventWeighter` (in `EventWeighter.h`) loads that TH2D once and evaluates
it with `TH2::Interpolate` — bilinear interpolation between bin centers,
i.e. a continuous `w(Q2, E')` — keeping each sampled electron with
probability `w`. That interpolated function *is* the handoff; nothing in
`runEventGenerator.cpp` changes.

### Why w is just the cross section

The generator draws `Q2 ~ U(Q2_range)` and `E' ~ U(E_range)` independently
and derives `theta` from them, so its proposal density `g(Q2, E')` is
**flat**. Rejection sampling with `w = (1/C) * d/g` turns a proposal into
a target; with `g` flat, `w` is proportional to `d = dsigma/dQ2 dE'`
itself. Evaluate, divide by the maximum, write. Overall units and
constants cancel in that rescale — only the *shape* matters.

Pass `--mc <lund>` if the proposal is **not** flat (a run that already had
a weight applied); `g` is then binned from that file instead.

### Three ways to give the cross section

| Flag | Source |
|---|---|
| `--formula "..."` | numpy expression in the kinematic variables |
| `--xsec-py file.py:func` | your own `f(Q2, Ep)` or `f(Q2, Ep, kin)` |
| `--table xsec.csv` | tabulated `(Q2, Ep, sigma)` points, interpolated |

Variables available to `--formula` (and in the `kin` dict passed to a
callable): `Q2`, `Ep`, `E`, `M`, `nu`, `y`, `x`, `W`, `W2`, `theta`,
`theta_deg`, `eps`, `Gamma` (virtual-photon flux, Hand; `Gamma_G` for
Gilman), `Mott`, plus the usual numpy functions.

```bash
# flux times a sigma_gamma*p that falls with W
--formula "Gamma * exp(-1.5 * (W - 2.85))"

# a pure 1/Q^4 shape
--formula "1.0 / Q2**2"

# your own model, with the kinematics handed to you
cat > my_model.py <<'EOF'
import numpy as np
def sigma(Q2, Ep, kin):
    return kin["Gamma"] * np.exp(-1.5 * (kin["W"] - 2.85))
EOF
python build_xsec_weight.py --input-card ../input.txt \
    --xsec-py my_model.py:sigma --out ../weight_func.root
```

### Cross sections in other variables

`--diff` applies the Jacobian into `dsigma/dQ2 dE'` for you:

| `--diff` | your cross section is | factor applied |
|---|---|---|
| `Q2Ep` (default) | `dsigma/dQ2 dE'` | 1 |
| `Q2nu` | `dsigma/dQ2 dnu` | 1 |
| `OmegaEp` | `dsigma/dOmega dE'` | `pi / (E E')` |
| `xQ2` | `dsigma/dx dQ2` | `x / nu` |
| `WQ2` | `dsigma/dW dQ2` | `M / W` |

`OmegaEp` assumes a phi-independent cross section (unpolarized beam and
target), which is what the generator samples.

### Phase space and the acceptance edge

`--input-card` reads `beam_energy`, `Q2_range`, `E_range`, `theta_range`,
`W_min` and `target_pid` from the same card the generator uses, so the
surface matches the run it steers. Override any of them with the explicit
flags (`--beam-energy`, `--theta-range`, `--x-range`, ...).

Bins the generator cannot reach (no solution for `theta`, `theta` outside
`theta_range`, `W < W_min`) are handled by `--outside`:

- **`clip` (default)** — keep the cross section there, capped at the
  largest physical value.
- `zero` — set `w = 0`.
- `keep` — leave it alone.

This is not about producing unphysical events: the generator re-applies
both cuts *after* the accept-reject, so none are produced either way. It
is about the interpolation. `TH2::Interpolate`'s bilinear stencil
straddles the acceptance edge, so a ring of zeros drags down the weight of
genuinely physical points next to it. On a 100x100 grid that bias is
visible — a closure test of `--formula "Gamma"` gives rms pulls of 3.0
(Q2) and 7.2 (E') with `zero`, versus 1.1 and 0.8 with `clip`, and up to
25% deviation in the tails of the E' projection. `zero` is unbiased too
once the grid is fine enough that the boundary bins are narrow (300x300
was, here).

### Keep fraction

The `[eff]` line predicts what the generator will see before you run it:

```
[eff] accept probability over the reachable region = 0.0813;
      per uniform (Q2,E') draw = 0.0191 (~1 kept event per 52.3 samples)
```

A steep cross section spends its whole budget on the peak. `--wclip-pct
99` (or `--wmax`) caps the few extreme bins that set the `max = 1` scale
and usually buys back an order of magnitude, at the cost of flattening
those bins. Narrowing `--x-range` / `--y-range` to the region you care
about works too.

Note that the generator drops anything at or outside the TH2 edges
(`Interpolate` returns 0 on the boundary), so a grid narrower than the
sampled range is a silent extra cut — the script warns when that happens.

### Checking what you built

```bash
python plot_weight.py ../weight_func.root w_Q2_Ep
```

### Iterating

`--prev` multiplies the new surface into a previous cumulative one and
`--archive <dir>` keeps a versioned copy (`w_xsec_iter<N>.root`), the same
way `build_weight_func.py` does — useful when you layer a cross-section
weight and a data-driven correction. All surfaces must share identical bin
edges.

### A 3-D cross section, binned (`make_pseudo_xsec.py`)

When the cross section is 3-D in (Q², W, M) rather than 2-D, the first
thing to pin down is the binning. `make_pseudo_xsec.py` builds a pseudo
cross section — **arbitrary values**, right shape — on such a grid and
draws it so the binning can be judged before the real numbers exist:

```bash
cd reweight
python make_pseudo_xsec.py                    # ../pseudo_xsec.pdf + .npz
python make_pseudo_xsec.py --q2-edges "1,2,3,4.5,7" \
    --w-range "2.85,4.65" --nw 9 --m-range "1.85,3.05" --nm 24
```

One **page per Q² bin**, one **panel per W bin**, **M along x**. The
Breit-Wigner defaults (2.0 GeV, 0.4 wide) match the `mass_9999: BW 2. 0.4`
placeholder in `input.txt`.

The point of looking at it is the kinematic edge. For
`reaction: 2212, 9999: 9999, 2212, -2212` the ppbar system satisfies

    2 m_p  <=  M  <=  W - m_p

so the open M range **grows panel by panel down a page** (shaded grey
beyond the edge). Cells past it can never be populated, whatever weight
they carry — with the defaults only 74.5% of the 4x9x24 grid is open at
all. A W or M binning chosen without that in mind spends bins on empty
space and smears the edge across the ones that are left.

The `.npz` written next to the PDF holds `counts` (`[nq, nw, nm]`) plus
`edges` and `varnames`, the same layout the weight builders read.

### 3-D weighting: &sigma;(Q&sup2;, W, M) (`build_xsec_weight3d.py`)

A 3-D cross section cannot be applied where the 2-D one is. `weight_func`
acts at electron-sampling time, where only `Q2` and `E'` exist (`W` is
fixed by them). `M` — the invariant mass of the intermediate `X` from the
first vertex, i.e. `M_ppbar` — does not exist until the intermediate mass
has been sampled and the chain decayed. So the 3-D weight is a **third
accept-reject stage** in the main loop, next to `mom_weight`.

```
xsec_weight: xsec_weight.root w_Q2_W_M
```

#### The proposal is not flat, so it must be measured

`build_xsec_weight.py` can skip the denominator because the generator
draws `Q2` and `E'` uniformly. Nothing like that holds in 3-D: `W` is a
derived quantity whose density is whatever the uniform `(Q2, E')` square
induces, and `M` comes from the Breit-Wigner in `mass_9999`, resampled
above threshold and filtered by whether the decay closes. So `--gen` is
required.

It reads the generator's **truth ntuple**, not the LUND file — the LUND
holds two interchangeable protons and picking the one that came from `X`
is ambiguous, while the ntuple records `M` from the truth 4-vector. Turn
it on with `truth_ntuple: gen_truth.root` in the input card.

#### Workflow

```bash
# 1. unweighted run, truth_ntuple: gen_truth_unweighted.root
root -l -b -q 'runEventGenerator.cpp+("out.lund","input.txt")'

# 2. build the TH3D
cd reweight
python build_xsec_weight3d.py --scan \
    --xsec ../pseudo_xsec.npz --gen gen_truth_unweighted.root --out /dev/null
python build_xsec_weight3d.py \
    --xsec ../pseudo_xsec.npz \
    --gen  gen_truth_unweighted.root \
    --min-gen 2000 \
    --out  ../xsec_weight.root

# 3. add `xsec_weight: xsec_weight.root w_Q2_W_M`, rerun (keep truth_ntuple:
#    on, pointing somewhere new)

# 4. closure
python plot_xsec_closure.py \
    --gen    gen_truth_weighted.root \
    --xsec   ../pseudo_xsec.npz \
    --weight ../xsec_weight.root \
    --out    ../xsec_closure.pdf
```

#### Per-bin lookup, not interpolation

`xsec_weight_mode` defaults to `bin`: the event takes the weight of the
bin it lands in. That is the correct pairing for a **binned** cross
section — the weight is a per-bin ratio `d/g`, so applying it per bin
makes the accepted density proportional to `d` bin by bin, exactly.

`interp` uses `TH3::Interpolate`, which blends neighbouring bins into each
event's accept probability. On a coarse grid that pulls the result away
from the cross section it was built from — measured on the 4x9x24 grid
here, the closure went from **rms pull 1.3 to 28.6**. Use `interp` only if
the underlying cross section really is smooth and the binning is fine
enough that the two agree.

(`TH3::Interpolate` has a second trap: it returns 0 outside the hull of
the *bin centers*, so with 4 Q&sup2; bins over [1,7] it silently discards
`Q2 < 1.5` and `Q2 > 5.75`. The generator clamps into the hull before
calling it.)

#### Reading the closure PDF

Same layout as `make_pseudo_xsec.py` so the two can be flipped through
side by side: page per Q&sup2; bin, panel per W bin, `M` on x, with a
`gen/xs` ratio under each panel and its rms pull printed.

Two things to look for:

- **Hatched cells** — the weight is zero there, so the cross section can
  never be delivered. Pass `--weight` so those are excluded from the
  normalization: they hold ~10% of the pseudo cross section here, and
  folding them in biases every other cell upward by exactly that amount,
  which reads as a 10% closure failure everywhere.
- **A panel whose ratio sits off 1** — that `(Q2, W)` cell got the wrong
  share of events. `--norm panel` renormalizes each panel to its own
  count, which separates a shape problem from a normalization one.

#### Using your own cross section

`import_xsec.py` converts what you have into the npz the builder reads:

```bash
# a ROOT TH3 with axes (Q2, W, M)
python import_xsec.py --th3 xsec.root:h_xsec --units density --out my_xsec.npz

# a CSV with explicit bin edges -- the reliable form for non-uniform binning
#   Q2_lo,Q2_hi,W_lo,W_hi,M_lo,M_hi,sigma
python import_xsec.py --csv-edges xsec.csv --units density --out my_xsec.npz

# a CSV of bin CENTRES: Q2,W,M,sigma  (outer edges are inferred, so it warns)
python import_xsec.py --csv xsec.csv --units integral --out my_xsec.npz
```

Or write it yourself — the format is five lines of numpy:

```python
np.savez("my_xsec.npz",
         counts=sigma,                                    # [nQ2, nW, nM]
         edges=np.array([q2_edges, w_edges, m_edges], dtype=object),
         varnames=np.array(["Q2", "W", "M"]),
         units="density")
```

From there the workflow is identical — `--xsec my_xsec.npz`. Verified end
to end on an external TH3 with non-uniform Q&sup2; bins: closure came out
at rms pull 1.05.

##### `units` is not decoration

The weight is a ratio of **per-bin** numbers, so the numerator must be the
number of events the cross section predicts in that bin — its integral
over the bin, not its average height.

- `units="integral"` — each value is already the cross section integrated
  over its bin. Used as is.
- `units="density"` — each value is `dsigma/dQ2 dW dM`. Multiplied by the
  bin volume.

With uniform binning the volume is a constant and cancels. With
non-uniform binning it does not: the Q&sup2; edges `1,2,3,4.5,7` span a
factor **2.5**, so a density fed in as an integral under-populates the
widest Q&sup2; bin by 2.5x — and the closure plot would not flag it,
because it would be comparing against the same mis-scaled target.

Nothing in the numbers tells you which you have, so `--units` is required
on the importer, and both the builder and the closure plotter read it back
from the npz. `make_pseudo_xsec.py` records `density`. If the binning is
non-uniform and the file says nothing, you get a warning naming the
factor you are risking.

##### Sanity checks before you spend CPU

The grid must live where the generator can actually produce events:

1. **`M >= 2 m_p`** (1.8765) and **`M <= W - m_p`** — the second moves with
   W, so the open M range differs panel to panel.
2. **`W >= W_min`** from the card.
3. **The theta ceiling.** `theta_range: 5 35` puts a floor on `E'` and so a
   ceiling on `W` that tightens as Q&sup2; rises (4.30 GeV at
   Q&sup2;=1, 3.08 at Q&sup2;=7 for a 10.2 GeV beam). Cross section binned
   above it is undeliverable at any statistics.
4. **The M proposal has to cover it.** `mass_9999`'s Breit-Wigner is the
   proposal in M; if your cross section has support where that BW does not,
   the weight goes huge there and the keep fraction collapses.

`build_xsec_weight3d.py` reports the first three as "N bins carry cross
section but no generated events (X% of the total)". If X is large, fix the
card or the grid rather than paying for it in rejections.

#### Exactness

With per-bin lookup the accepted density is proportional to `d` bin by bin
**by construction**, so the weighted events match the cross section
exactly, up to Poisson noise, once you have enough of them. Measured on
this grid at 600k weighted events against a 20M-event denominator:

```
[closure] over 354 populated cells: rms pull = 1.02, median |gen/xs - 1| = 0.022
```

2.2% median deviation against 2.4% expected from statistics alone — i.e.
exact. Only two things break that, and both are yours to control.

**A clip on the ratio breaks it permanently.** `--wclip-pct` / `--wmax`
cap cells you *still deliver*, so they come out low no matter how many
events you generate. The five cells clipped at the 99th percentile in an
earlier run were 15–28% under-produced, at 3–5σ. **Clipping is off by
default.** It buys speed by lying.

**Denominator noise sets the accuracy floor.** `w = d/g`, so a cell whose
`g` was measured from `N` events carries a `1/sqrt(N)` error into the
weight, and the closure inherits it. `--min-gen` bounds this — an accuracy
floor, not a cosmetic cut. `--target-accuracy` (default 0.05) does the
arithmetic: 5% accuracy → `--min-gen 400`.

Unlike clipping, `--min-gen` is honest — dropped cells go to `w = 0` and
show up hatched in the closure plot — and it is usually **faster**, because
the cells it removes are exactly the ones setting the `max = 1` scale.

#### `--scan`: choosing the floor

```bash
python build_xsec_weight3d.py --xsec ../pseudo_xsec.npz \
    --gen gen_truth.root --scan --out /dev/null
```

```
   --min-gen   cells  % of sigma  efficiency  worst cell wt err
           0     443       92.1%      0.0023              22.4%
         200     408       89.0%      0.0042               7.0%
         500     391       86.0%      0.0073               4.4%
        1000     374       83.5%      0.0091               3.2%
        2000     354       78.8%      0.1186               2.2%  <- cliff
        5000     317       76.9%      0.1200               1.4%
```

Note the cliff: **one** badly-sampled cell was holding the whole surface
hostage. Stepping past it costs 4.7% of the cross section and buys a 13×
speedup, and every delivered cell is still exact. That cliff is why to run
`--scan` rather than guess — and why not to reach for a clip, which keeps
that cell and under-produces it instead.

#### The proposal has to overlap the target

The card's `mass_9999: BW 2. 0.4` proposes `M` in a narrow peak at 2.0,
while the pseudo cross section has support out to 3.05. The weight has to
be huge in that tail, and the keep fraction collapses — each rejection
costs a whole decay chain. Widening the placeholder width is far cheaper
than paying for it in rejections, and unlike a clip it costs no accuracy.

#### Cross section outside the acceptance

Some of the grid is not reachable at all. With `theta_range: 5 35` and a
10.2 GeV beam, `Q2 = 4 E E' sin^2(theta/2)` puts a floor on `E'` and hence
a ceiling on `W`:

| Q&sup2; [GeV&sup2;] | 1.0 | 2.0 | 3.0 | 4.5 | 7.0 |
|---|---|---|---|---|---|
| W ceiling [GeV] | 4.30 | 4.12 | 3.94 | 3.64 | 3.08 |

4M unweighted events reproduce those ceilings to three decimals. So the
whole `Q2 > 4.5, W > 3.74` corner of the pseudo grid is closed by the
theta cut — it shows up as fully hatched panels on the last page of the
closure PDF. Cross section placed there is simply not deliverable: widen
`theta_range`, or do not bin into that corner.

### How this relates to `build_weight_func.py`

Same output, same generator plumbing, different numerator:

| | numerator | needs |
|---|---|---|
| `build_weight_func.py` | measured data density | real data + a generator/reco run |
| `build_xsec_weight.py` | a cross section you supply | nothing but the input card |

Use the cross-section builder to start from a physics model; use the
data-driven builder to correct toward measured data.

## Carried ratio weight: d&sigma;/dt(s, t) (`build_dsdt_table.py`)

A different kind of weight from everything above: it is **carried**, not
used to accept or reject. For `e p → e' p p p̄` the generator attaches
to every event

```
w_ratio = dσ/dt(s_pbarp, t) / dσ/dt(s_pp, t)
```

with one parametrization of the elastic dσ/dt evaluated at the p̄p and
the pp sub-energies of the same event (`s_pbarp = (p_pbar + p_recoil)²`,
`s_pp = (p_fromX + p_recoil)²`, `p_fromX = p_X − p_pbar`) at the event's
first-vertex `t = (p_target − p_recoil)²` — or, with a `_den` key, a
p̄p model in the numerator and a separate pp model in the denominator.
The weight is written as the
truth-ntuple branch `w_ratio`, together with `t`, `s_pbarp`, `s_pp`, and
optionally to a sidecar file — one `%.6f` per line, parallel to the LUND,
the same format Example 4 produces — via `ratio_weight_sidecar:`.

Nothing is normalized: any overall constant in dσ/dt cancels in the
ratio, so units do not matter and there is no max-1 rescaling.

### Handing over dσ/dt

Two ways, and they are checked against each other below:

```
ratio_weight_formula: exp((4.0 + 0.5*log(s))*t) * pow(s,-2)   # TFormula in s, t
```
```
ratio_weight: dsdt_table.root log_dsdt_s_t                    # TH2D from this script
ratio_weight_mode: log
```

Either can be paired with a separate denominator model,
`ratio_weight_formula_den:` or `ratio_weight_den: <file> [<hist>]`
(a `--log` table in the denominator needs the numerator to be `--log`
too: `ratio_weight_mode` applies to both). Without one, the same model is
evaluated at `s_pp`.

The formula is evaluated by ROOT's `TFormula` per event; write it with
`log` (natural) and `pow` so the identical string also works as
`--formula` here. The table is the general path — fit the Ambats/White
data in Python, or hand over a function or a CSV of points:

```bash
python build_dsdt_table.py --formula "exp((4.0 + 0.5*log(s))*t) * pow(s,-2)" \
    --s-range 3.4,12.4 --ns 90 --t-range=-14.2,0 --nt 200 --log \
    --gen gen_truth.root --out ../dsdt_table.root
python build_dsdt_table.py --xsec-py model.py:dsdt ...        # f(s, t) -> array
python build_dsdt_table.py --table dsdt.csv --cols s,t,dsdt ...   # scattered points
```

`--t-range` needs the `=` form: a value starting with `-` is otherwise
read as a flag.

### Bin centers, not bin integrals

The accept-reject builders above produce a per-bin ratio, so their
numerator has to be the cross section **integrated** over the bin. This
table is different: the generator reads it with `TH2::Interpolate`,
bilinearly between bin **centers**, as a continuous function. So the
function is evaluated *at* the centers, there is no `--supersample`, and
the question is only whether a bilinear patch follows dσ/dt between
neighbouring centers.

For an exponential in `t` it does not, unless the grid is fine. Hence
`--log`: the table then holds `ln dσ/dt`, the generator interpolates that
and exponentiates (`ratio_weight_mode: log`), and an exponential becomes
a plane. Measured with the formula above on 50 000 events:

| grid (s × t) | linear: median / max `|w_table/w_exact − 1|` | log: median / max |
|---|---|---|
| 90 × 200 | 3.5e-4 / 3.0e-3 | 6.7e-5 / 5e-4 |
| 90 × 20  | 1.9e-2 / 2.8e-1, mean w biased −7 % | 6.7e-5 / 9e-2 (edge clamp) |

Use `--log` unless dσ/dt crosses zero.

### `--gen`: coverage and error before you run

Give `--gen` the truth ntuple of any previous run (the `t`, `s_pbarp`,
`s_pp` branches are always written, weight or no weight). The script
prints the percentiles of the three variables, the fraction of events
whose **both** `s` values and `t` fall inside the grid — the generator
gives `w_ratio = 0` to the rest, and counts them under
"outside dsigma/dt table" — and, with a model on the command line, the
exact interpolation error the table will incur on those events. That
number is what the generator will reproduce: in the test above the
generator's own `w_ratio` and the script's prediction agreed to every
printed digit. If the ntuple already carries a `w_ratio ≠ 1` (a run with
`ratio_weight:`), it is compared to the model too, which is the
table-vs-formula closure without a second run.

### Seeing what it does

```bash
python plot_ratio_weight.py gen_truth.root --out ../ratio_weight.pdf \
    --formula "exp((4.0 + 0.5*log(s))*t) * pow(s,-2)"     # optional overlay
```

Three panels in `t`: the generated `dN/dt` with and without the weight
(shapes, unit area), their ratio — the mean `w_ratio` in each `t` bin,
both raw and with the integral fixed — and the per-event spread of
`log10 w_ratio` at each `t`. The ratio panel is the reweighting factor
the p̄p / pp hypothesis applies as a function of `t`. With `--formula`
(and `--formula-den`) the **input** ratio is evaluated from the same
truth branches and averaged in the same bins, so input and extracted sit
on the same axes. Smallest closure:

```
ratio_weight_formula: 2
ratio_weight_formula_den: 1
```

gives `w_ratio = 2` for every event and `python plot_ratio_weight.py
truth.root --formula 2 --formula-den 1` shows extracted on input at 2 in
every `t` bin.

### Recomputing offline

`w_ratio` is a deterministic function of the three truth branches, so it
can always be rebuilt in numpy (`f(s_pbarp, t) / f(s_pp, t)`) — the
formula path reproduces that to machine precision, the table path to the
interpolation error above. The sidecar is the same numbers, in LUND
order, for use downstream of GEMC.

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
