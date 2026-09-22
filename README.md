# ROOT Electroproduction Event Generator (runEventGenerator2)

This repository contains a ROOT/C++ macro that generates electroproduction events of the form

\[
e + p \rightarrow e' + W,\quad W \rightarrow \text{final state (with optional cascaded decays)}
\]

The generator is intended for **toy Monte Carlo**, acceptance studies, and background modeling. It is not meant to be a precision cross-section generator.

---

## Features

- Uniform sampling of scattered-electron kinematics
- Construction of the hadronic system \(W\)
- Two-body decay of \(W\) with optional exponential \(t\)-slope weighting
- Recursive decay of intermediate particles
- Support for custom “placeholder” PDG codes with user-defined masses
- Optional LUND file output
- Optional ROOT diagnostic plots

---

## Repository contents

- **runEventGenerator.cpp**  
  ROOT macro containing the generator logic and the `runEventGenerator()` entry point.

- **EventWeighter.h**  
  Header-only weighting class the macro `#include`s. It owns every
  accept-reject weight surface (`weight_func`, `mom_weight`,
  `xsec_weight`): parsing their input-card keys, loading the ROOT
  histograms, and deciding whether to keep an event. The generator only
  hands over the sampled kinematics, so the weighting can be changed
  without touching the generation code. See [Weighting](#weighting).

- **input.txt**  
  Input card that controls the generator configuration.

- **reweight/**  
  Standalone weighting tools. They run separately from the generator and
  hand it a weight surface through the `weight_func:` line of the input
  card — `build_xsec_weight.py` builds one from a cross section,
  `build_weight_func.py` from measured data. See [Weighting](#weighting).

---

## Requirements

- ROOT (with `TLorentzVector`, `TH1D`, `TCanvas`, `TRandom3`, `TGenPhaseSpace`)
- A C++ compiler compatible with your ROOT build (ACLiC / Cling)

---

## Quick start

Place `runEventGenerator.cpp`, `EventWeighter.h` and `input.txt` in the
same directory.

### Interactive mode

```bash
root -l runEventGenerator.cpp
```

### Compiled (ACLiC)

```bash
root -l -b -q 'runEventGenerator.cpp+("events.lund","input.txt")'
```

The `+` compiles the macro (and `EventWeighter.h`, which ACLiC tracks as a
dependency) into `runEventGenerator_cpp.so` and reuses it until either
file changes.

## Input File
The generator runs entirely through an input text file. Where each non-comment line follows the format

```
key: value(s)
```
Any lines beginning with # are ignored. 

### Required Parameters

| Key         | Description                                   | Type       |
|-------------|-----------------------------------------------|------------|
| num_events  | Number of generated events                    | int        |
| beam_energy | Electron beam energy (GeV)                    | float      |
| target_pid  | Particle ID number of target                  | int        |
| W_min       | Minimum hadronic final state invariant mass (GeV) | float   |
| Q2_range    | Range of Q^{2} (GeV^{2})                      | float float |
| theta_range | Range of scattered electron theta             | float float |
| E_range     | Range of scattered electron energy            | float float |
| t_slope     | Weighting parameter for t-slope               | float      |
| write_lund  | Write output LUND file                         | int (0 / 1) |
| gen_plots   | Show plots                                    | int (0 / 1) |
| print_debug | Print debug information                        | int (0 / 1) |
| reaction    | Wanted reaction string                         | string     |

### Optional Parameters

| Key         | Description                                   | Type       |
|-------------|-----------------------------------------------|------------|
| weight_func | Weight surface `w(Q^2, E')` applied at electron-sampling time: `<root file> [<hist name>]` (name defaults to `w_Q2_Ep`) | string |
| mom_weight  | Weight surface `w(p_lead, p_sub)` applied as a second accept-reject after the event is built | string |
| xsec_weight | 3-D weight surface `w(Q^2, W, M_X)` applied as a third accept-reject after the decay chain: `<root file> [<hist name>]` (name defaults to `w_Q2_W_M`) | string |
| xsec_weight_mode | How to read that TH3D: `bin` (default, per-bin lookup) or `interp` (trilinear) | string |
| ratio_weight, ratio_weight_formula | p̄p rescattering model `σ_p̄p(s, t)`: a TH2D table `<root file> [<hist>]` (name defaults to `dsdt_s_t`) or a `TFormula` in `s`, `t`. Turns on the rescattering stage (see [Rescattering weights](#rescattering-weights-pp-and-pp)) | string |
| ratio_weight_den, ratio_weight_formula_den | pp rescattering model `σ_pp(s, t)`, table or formula; without one the p̄p model is used for both | string |
| ratio_weight_gen | Generated `(s, t)` density `D_gen` from `build_dsdt_table.py --from-gen` (name defaults to `dgen_s_t`), per-bin lookup; weights are model / `D_gen`. Without it the weights are the model values | string |
| ratio_weight_mode | Whether the model tables hold `dσ/dt` (`linear`, default) or `ln dσ/dt` (`log`, built with `--log`; recommended) | string |
| ratio_weight_apply | `accept` (default): accept-reject on `w_pbarp · w_pp`, one unweighted sample out. `carry`: keep every event and record both weights instead | string |
| ratio_weight_max | Accept-reject ceiling on `w_pbarp · w_pp` — any number at or above its maximum, a plain constant included. Omit to scan the tables and the `D_gen` grid at load time (a formula model with neither has no grid, so there it is required) | float |
| ratio_weight_sidecar | Write the two weights per event, `w_pbarp w_pp`, one line per LUND event | string |
| truth_ntuple | Write a per-accepted-event truth TTree of `(Q2, W, M, Ep, theta_e, w_pbarp, w_pp, w_ratio, t, s_pbarp, s_pp)` to this ROOT file | string |

All of these are built by a **separate** script and simply loaded here;
the weighting keys are parsed and applied by `EventWeighter.h`, not by
the generator — see [Weighting](#weighting).

## Reaction
If you want to decay multiple particles just have to separate using :
Lets say you want the reaction ep->XZ->XWY
The reaction string should be X, Z: Z, W, Y
The first two are always assumed to be some final state particle (X) and an intermediate particle (Z). The generator can handle as many intermediate particles as wanted.

## How the generator works

1. **Read inputs:** Parses `input.txt`, ignoring `#` comments, and loads kinematic ranges and options.  
2. **Sample kinematics:** Uniformly samples scattered‑electron kinematics within the provided ranges (`Q2_range`, `theta_range`, `E_range`).  
3. **Build hadronic system:** Constructs the hadronic system \(W\) from the beam/target and scattered electron, enforcing \(W_{\min}\).  
4. **Decay \(W\):** Performs two‑body decay of \(W\); optional exponential \(t\)-slope weighting can reweight events.  
5. **Cascade decays:** Recursively decays intermediate particles per the `reaction` string, supporting placeholder PDG codes.  
6. **Output/diagnostics:** Optionally writes LUND output and fills ROOT histograms/plots.

---

## Weighting

Weights are **not** built by the generator, and they are not applied by
the generator's own code either. Everything weighting-related lives in
`EventWeighter.h`, a header-only class the macro includes:

- `WeightConfig` — the input-card side. `ReadInput` holds one, and the
  card parser delegates any key it does not recognise to
  `WeightConfig::parseKey`, which owns `weight_func`, `mom_weight`,
  `xsec_weight`, `xsec_weight_mode` and the `ratio_weight*` keys.
- `EventKinematics` — what a built event looks like to the weighter:
  `Q2`, `Ep`, `W`, `M_X`, the truth first-vertex 4-vectors
  (`q`, `p_target`, `p_recoil`, `p_X`) and a pointer to the final-state
  particle list.
- `EventWeighter` — loads the ROOT histograms once and exposes the
  accept-reject stages the generator calls:

  ```cpp
  EventWeighter weighter(input.weights);
  weighter.load();
  ...
  weighter.acceptElectron(Q2, Ep, rnd);        // inside the electron sampler
  weighter.acceptEvent(kin, rnd);              // after the full decay chain
  auto r = weighter.acceptRescattering(kin, rnd);  // r.keep, r.w.w_pbarp, r.w.w_pp
  weighter.printSummary(cout);                 // rejection counts per surface
  ```

  It also keeps the per-stage rejection counters, so the generator's
  main loop is a single `if (!weighter.acceptEvent(kin, gen.rnd)) continue;`.

To add a new weight surface: give `WeightConfig` a file/name pair and a
`parseKey` branch, load it in `EventWeighter::load()`, and evaluate it in
`acceptElectron()` (if it depends only on the scattered electron),
`acceptEvent()` (if it needs the decayed event) or `eventWeights()` (if
it belongs with the rescattering weights the event carries).
`runEventGenerator.cpp` does not change.

A separate script writes a `TH2D` of accept probabilities to a ROOT file;
the weighter loads it once and evaluates it with `TH2::Interpolate` —
bilinear interpolation between bin centers, i.e. a continuous
`w(Q^2, E')` — keeping each sampled electron with probability `w`. The
handoff is one line in `input.txt`:

```
weight_func: weight_func.root w_Q2_Ep
```

Two builders write that file, and they are interchangeable from the
generator's point of view:

- **From a cross section** — `reweight/build_xsec_weight.py`. Needs
  nothing but the input card. The generator samples `Q^2` and `E'`
  uniformly, so its proposal density is flat and the accept probability is
  just the cross section rescaled to a maximum of 1.

  ```bash
  cd reweight
  python build_xsec_weight.py \
      --input-card ../input.txt \
      --formula "Gamma * exp(-2.0 * (W - 2.85))" \
      --out ../weight_func.root
  ```

  The cross section can also come from your own Python function
  (`--xsec-py file.py:func`) or a table of measured points
  (`--table xsec.csv`), and `--diff` converts one quoted in
  `dsigma/dOmega dE'`, `dsigma/dx dQ^2` or `dsigma/dW dQ^2` into the
  `dQ^2 dE'` the generator needs.

- **From data** — `reweight/build_weight_func.py`. Builds the same surface
  as a data/MC ratio, for correcting a run toward measured distributions.

### 3-D cross sections

A cross section that is 3-D in `(Q^2, W, M)` — where `M` is the invariant
mass of the intermediate `X` from the first vertex — cannot be applied at
electron-sampling time, because `M` does not exist until the intermediate
mass has been sampled and the chain decayed. It is a **third**
accept-reject stage in the main loop, configured with `xsec_weight:` and
built by `reweight/build_xsec_weight3d.py`:

```bash
# 1. unweighted run with `truth_ntuple: gen_truth.root` in the card
# 2. build the TH3D
cd reweight
python build_xsec_weight3d.py --xsec ../pseudo_xsec.npz \
    --gen gen_truth.root --out ../xsec_weight.root
# 3. add `xsec_weight: xsec_weight.root w_Q2_W_M` and rerun
```

Unlike the 2-D case the proposal density is **not** flat here, so it has to
be measured — hence `truth_ntuple:`, which records `M` from the truth
4-vector (the LUND file's two protons are interchangeable, so `M` is
ambiguous downstream). `reweight/plot_xsec_closure.py` then overlays the
generated distribution on the cross section with a ratio panel per bin.
With per-bin lookup the weighted events match the cross section **exactly**
(up to Poisson noise) — measured at rms pull 1.02 and 2.2% median
deviation over all delivered cells. Ratio clipping would break that
permanently, so it is off by default; use `--min-gen` (or
`--target-accuracy`) instead, and run `--scan` to see what it costs.

See [reweight/README_reweight.md](reweight/README_reweight.md) for the
full workflow and the traps (per-bin vs interpolated lookup, denominator
statistics, proposal overlap, and cross section placed outside the
acceptance).

Inspect a surface before running with
`python reweight/plot_weight.py weight_func.root w_Q2_Ep`. Full details in
[reweight/README_reweight.md](reweight/README_reweight.md).

### Rescattering weights (p̄p and pp)

Rescattering in `e p → e' p_recoil p p̄` — a controlled test for
extracting the ratio of p̄p to pp rescattering (tying the small-|t|
Ambats et al. and large-|t| White et al. elastic ratios together).
Define, per event and from truth 4-vectors,

- `t = (p_target − p_recoil)²` — the target–recoil momentum transfer,
  shared by both subsystems;
- `s_rp̄ = (p_recoil + p_p̄)²` — the p̄p rescattering system;
- `s_rp = (p_recoil + p_produced)²` — the pp rescattering system, with
  `p_produced = p_X − p_p̄` (exact by conservation, so the two protons
  never have to be told apart).

The final state holds **both** subsystems, so every event carries two
weights, each a model `dσ/dt(s, t)` at its own sub-energy divided by the
generator's own `(s, t)` density:

```
w_pbarp = σ_p̄p(s_rp̄, t) / D_gen(s_rp̄, t)      p̄p rescattering
w_pp    = σ_pp (s_rp,  t) / D_gen(s_rp,  t)      pp rescattering
```

`D_gen` is one histogram — the truth ntuple of an unweighted run binned
in `(s, t)` — evaluated at two different points.

The generator then **accept-rejects on their product**, keeping the
event with probability `(w_pbarp · w_pp) / w_max`. What comes out is one
unweighted `e p → e' p_recoil p p̄` sample, one LUND file, the full
reaction exactly as in an unweighted run, distributed as

```
D_gen × w_pbarp × w_pp
```

— the generated distribution multiplied by both weights, each factor
reshaping its own subsystem. Nothing downstream has to apply a weight.

`w_max` comes from `ratio_weight_max:`, any number at or above the
product's maximum (a plain constant is fine; too high only costs
efficiency). Omit it and it is scanned off the model tables and the
`D_gen` grid when they load — a formula model with neither has no grid
to scan, so there the key is required and the stage refuses to run
without it. Events whose product exceeds the ceiling are clamped to
accept and counted, so a ceiling set too low shows up in the run summary
rather than silently flattening the sample:

```
  rescattering weights (accept-reject on w_pbarp * w_pp): 4107048 events weighted, ...
    - ceiling w_max on w_pbarp * w_pp: 2  (ratio_weight_max)
    - kept:                            40000
    - rejected:                        4067048
```

Closure (`σ_p̄p = e^{2t}(1 + 0.1 s)`, `σ_pp = e^{t}`, 40k accepted
against a 300k carried reference): the accepted sample matches the
reference reweighted by `w_pbarp · w_pp` with χ²/ndf = 0.75, 1.02, 1.04,
0.56, 0.66 over 20 bins of `t`, `s_rp̄`, `s_rp`, `Q²` and `W`.

Card:

```
ratio_weight_formula:      exp(2.0*t)*(1.0+0.1*s)   # sigma_pbarp(s, t)
ratio_weight_formula_den:  exp(1.0*t)               # sigma_pp(s, t)
ratio_weight_max:          2.0                      # ceiling on the product
ratio_weight_gen:          dgen.root dgen_s_t       # optional D_gen
```

or tables built in Python (`ratio_weight:` / `ratio_weight_den:`, from
`reweight/build_dsdt_table.py`, which takes a formula, a Python
`f(s, t)` or a CSV of points; prefer `--log` with
`ratio_weight_mode: log`). With tables the ceiling can be left out:

```
  accept-reject ceiling scanned over 14400 (s, t) grid points:
    max w_pbarp = 3.20175, max w_pp = 0.959189 -> w_max = 3.22464
```

A point outside a table's range has no model value, so the event is
rejected and counted under `outside dsigma/dt table`. Keep the grid
wider than the `(s, t)` the run populates.

#### Carried weights (`ratio_weight_apply: carry`)

`ratio_weight_apply: carry` turns the selection off: every event is
kept, nothing is normalized, and the weights ride along as truth-ntuple
branches `w_pbarp`, `w_pp`, their ratio `w_ratio` and their product
`w_event` — plus, with `ratio_weight_sidecar:`, two columns per LUND
event. This is the mode for measuring a ratio out of a *single* sample,
and for the reference histogram a closure check compares the accepted
sample against.

**Extraction** (`reweight/extract_dsdt_ratio.py`), exactly as with data:
histogram `t` for events by their `s_rp̄` bin weighted with `w_pbarp`,
histogram `t` for events by their `s_rp` bin weighted with `w_pp`, and
divide:

```
R_extracted(s, t) = dN/dt | s_rp̄ ∈ bin  (w_pbarp)
                    ------------------------------  =  σ_p̄p(s, t) / σ_pp(s, t)
                    dN/dt | s_rp  ∈ bin  (w_pp)
```

The input ratio is overlaid at the bin centres. Measured: a flat 2
comes back as 1.96–2.04 in every s bin (200k events, rms pull 0.9); with
`σ_p̄p = 4.48 e^{8t}` and `σ_pp = e^{5t}` (p̄p steeper, crossing at
|t| = 0.5) the extracted ratio follows `4.48 e^{3t}` in every s bin.

Two things `D_gen` does and does not do. The extracted *ratio* is right
with or without it, because the generator's shape is common to both
sides. Each weighted spectrum *on its own* follows its σ only with it:
fitting the t slope of the `w_pbarp`-weighted sample gives 7.1–7.9
without `D_gen` (generator shape leaking in) and 8.00–8.02 with it. The
same holds factor by factor in accept-reject mode.

Workflow:

```bash
# 1. unweighted run with `truth_ntuple: gen_truth.root`
# 2. D_gen on a grid covering the reported s and t ranges
cd reweight
python build_dsdt_table.py --from-gen gen_truth.root \
    --s-range 3.4,12.4 --ns 45 --t-range=-14.2,0 --nt 142 --out ../dgen.root
# 3. run with the two models (+ ratio_weight_gen), then extract
python extract_dsdt_ratio.py gen_truth_weighted.root \
    --formula "4.4817*exp(8.0*t)" --formula-den "exp(5.0*t)" \
    --t-range=-2.5,0 --out ../dsdt_ratio.pdf
```

Because `s_rp̄`, `s_rp` and `t` are in the truth ntuple, both weights can
always be recomputed offline in either mode; `plot_ratio_weight.py`
shows the per-event ratio `w_pbarp / w_pp` against `t`.

## Example usage

- **Toy Monte Carlo:** Generate a few events to test the generator.
- **Acceptance studies:** Generate events over a range of kinematic variables.
- **Background modeling:** Generate events to model background processes.

---

## Example input.txt

```
# basic run
num_events: 10000
beam_energy: 10.6
target_pid: 2212
W_min: 1.6
Q2_range: 1.0 6.0
theta_range: 5.0 35.0
E_range: 1.0 9.0
t_slope: 3.0
write_lund: 1
gen_plots: 1
print_debug: 0
reaction: 2212, 1000: 1000, 211, -211
```

## Reaction examples

- **Simple two‑body:**  
  `reaction: 2212, 211`  
  Produces \(W \to p \pi^+\).

- **One intermediate:**  
  `reaction: 2212, 1000: 1000, 211, -211`  
  Here `1000` is a placeholder intermediate that decays to \(\pi^+\pi^-\).

- **Two intermediates (correct format):**  
  `reaction: target, int1: int1, daughter1, int2: int2, daughter2, daughter3`  
  Example:  
  `reaction: 2212, 1000: 1000, 211, 1001: 1001, -211, 22`  
  Here `1000` decays to \(\pi^+\) and `1001`, then `1001` decays to \(\pi^-\) and \(\gamma\).

> **Note:** Placeholder PDG codes must have masses defined in the macro.

## Output

- **LUND:** Written when `write_lund: 1` (see macro for output path/name).  
- **ROOT plots:** Produced when `gen_plots: 1`.  
- **Downstream simulation:** The LUND file is intended to be passed to **GEMC/GEANT4** for detector simulation.

## Limitations

- Uniform kinematic sampling by default; supply a cross section via
  `weight_func:` to sample a physics distribution (see [Weighting](#weighting)).  
- Simple phase‑space decays; no detector effects.  
- Placeholder PDG codes require user‑defined masses.  
- No flag added to set RNG seed.  
- Detector effects are not modeled in this generator; use GEMC/GEANT4 with the LUND output.

## Troubleshooting

- **No events:** Check `W_min` vs your kinematic ranges.  
- **Bad reaction string:** Ensure proper comma/colon formatting.  
- **ROOT errors:** Verify your ROOT build matches your compiler.  
- **ACLiC fails with `KernelKit requires -fdefine_target_os_macros` /
  `redefinition of 'kTRUE'`:** the macOS Command Line Tools SDK is newer
  than the clang bundled with your ROOT (seen with conda ROOT 6.28 after
  an SDK update). Point ROOT at an older SDK that is still installed:

  ```bash
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX13.1.sdk root -l -b -q 'runEventGenerator.cpp+("events.lund","input.txt")'
  ```

  (`ls /Library/Developer/CommandLineTools/SDKs/` lists the candidates.)

