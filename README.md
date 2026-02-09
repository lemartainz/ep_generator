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

- **input.txt**  
  Input card that controls the generator configuration.

---

## Requirements

- ROOT (with `TLorentzVector`, `TH1D`, `TCanvas`, `TRandom3`, `TGenPhaseSpace`)
- A C++ compiler compatible with your ROOT build (ACLiC / Cling)

---

## Quick start

Place `runEventGenerator.cpp` and `input.txt` in the same directory.

### Interactive mode

```bash
root -l runEventGenerator.cpp
```

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

- Uniform kinematic sampling (not a physics cross‑section).  
- Simple phase‑space decays; no detector effects.  
- Placeholder PDG codes require user‑defined masses.  
- No flag added to set RNG seed.  
- Detector effects are not modeled in this generator; use GEMC/GEANT4 with the LUND output.

## Troubleshooting

- **No events:** Check `W_min` vs your kinematic ranges.  
- **Bad reaction string:** Ensure proper comma/colon formatting.  
- **ROOT errors:** Verify your ROOT build matches your compiler.

