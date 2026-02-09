#include <limits>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TMath.h>
#include <TGenPhaseSpace.h>
#include <TLegend.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

using namespace std;

// ---------------------------------------------------------
// PDG masses (GeV)
// ---------------------------------------------------------
struct PDG {
    static constexpr double proton   = 0.9382720813;
    static constexpr double pion     = 0.13957039;
    static constexpr double pion0    = 0.1349768;
    static constexpr double electron = 0.0005109989461;
    static constexpr double Kp       = 0.493677;
    static constexpr double Xi       = 1.32171;
    static constexpr double photon   = 0.0;
    static constexpr double neutron  = 0.9395654133;
};

struct MassSpec {
    std::string func = "fixed";
    double val = 0.0;
    double width = 0.0;
    bool enabled = false; // false => fall back to getMass() or fixed override
};

static std::map<int, MassSpec> customMassSpec;

// Custom mass overrides (for fake PDGs like 9999, 9998, etc.)
static std::map<int,double> customMass;

// ---------------------------------------------------------
// Helper functions
// ---------------------------------------------------------
double getMass(int pdg) {
    auto it = customMass.find(pdg);
    if (it != customMass.end()) return it->second;

    switch (pdg) {
        case 2212:   // p
        case -2212:  // pbar
            return PDG::proton;
        case 211:    // pi+
        case -211:   // pi-
            return PDG::pion;
        case 111:    // pi0
            return PDG::pion0;
        case 2112:   // n
        case -2112:  // nbar
            return PDG::neutron;
        case 11:     // e-
        case -11:    // e+
            return PDG::electron;
        case 321:
        case -321:
            return PDG::Kp;
        case 3312:
        case -3312:
            return PDG::Xi;
        default:
            return 0.0;
    }
}

double sampleMassInput(std::string func,
                       float mass_val,
                       float mass_width,
                       TRandom3 &rnd)
{
    if (func == "fixed") {
        return mass_val;

    } else if (func == "gaussian" || func == "gauss") {
        return rnd.Gaus(mass_val, mass_width);

    } else if (func == "uniform") {
        return rnd.Uniform(mass_val - mass_width,
                           mass_val + mass_width);

    } else if (func == "breitwigner" || func == "bw"
            || func == "BW" || func == "BreitWigner") {
        return rnd.BreitWigner(mass_val, mass_width);

    } else {
        cerr << "WARNING: unrecognized mass sampling function '"
             << func << "'. Using fixed value.\n";
        return mass_val;
    }
}

double getMassSampled(int pdg, TRandom3 &rnd,
                      double m_min = 0.0,
                      int max_tries = 20000,
                      bool *ok = nullptr)
{
    // default: fixed mass from table/override
    auto it = customMassSpec.find(pdg);
    if (it == customMassSpec.end() || !it->second.enabled) {
        double m = getMass(pdg);
        bool pass = std::isfinite(m) && (m >= m_min) && (m > 0.0);
        if (ok) *ok = pass;
        return pass ? m : 0.0;
    }

    const auto &spec = it->second;

    for (int i = 0; i < max_tries; ++i) {
        double m = sampleMassInput(spec.func, (float)spec.val, (float)spec.width, rnd);
        if (!std::isfinite(m)) continue;
        if (m <= 0.0) continue;
        if (m < m_min) continue;     // threshold cut
        if (ok) *ok = true;
        return m;
    }

    if (ok) *ok = false;
    return 0.0; // caller should treat as failure and skip event
}

double parentDecayThreshold(int parent_pdg,
                            const std::map<int, std::vector<int>>& decay_map)
{
    auto it = decay_map.find(parent_pdg);
    if (it == decay_map.end()) return 0.0;

    double thr = 0.0;
    for (int d : it->second) thr += getMass(d);
    return thr;
}

bool sampleIntermediateAboveThreshold(int pdg,
                                      TRandom3& rnd,
                                      const std::map<int, std::vector<int>>& decay_map,
                                      double& m_out,
                                      int max_tries = 5000)
{
    // if no decay, no threshold constraint needed
    double thr = parentDecayThreshold(pdg, decay_map);

    for (int i = 0; i < max_tries; ++i) {
        double m = getMassSampled(pdg, rnd);
        if (!std::isfinite(m) || m <= 0.0) continue;
        if (m < thr) continue;
        m_out = m;
        return true;
    }
    return false;
}

struct Range {
    double min;
    double max;
    bool is_range; // true = sample uniformly, false = single value
    double sample(TRandom3 &rnd) const {
        return is_range ? rnd.Uniform(min, max) : min;
    }
};

struct ReadInput {
    int num_events        = 0;
    double beam_energy    = 0;
    int    target_pid     = 2212;
    Range Q2_range        {0,0,false};
    Range E_range         {0,0,false};
    Range theta_range     {0,0,false}; // stored in radians
    double W_min          = 0;
    double t_slope        = 0;
    std::string reaction;
    bool print_debug      = false;
    bool write_lund       = true;
    bool gen_plots        = false;
};

// Two-body decay result
struct DecayResult {
    TLorentzVector d1_lab;
    TLorentzVector d2_lab;
    double t_min;
    double t_max;
};

double twoBodyMomentum(double M, double m1, double m2) {
    if (M <= m1 + m2) return NAN;
    double term = (M*M - (m1+m2)*(m1+m2)) * (M*M - (m1-m2)*(m1-m2));
    return 0.5/M * sqrt(term);
}

double invariantSquare(const TLorentzVector &a, const TLorentzVector &b, bool use_sum=false) {
    TLorentzVector d = use_sum ? (a+b) : (a-b);
    return d.M2();
}

// ---------------------------------------------------------
// 4-vector sanity helpers (NEW)
// ---------------------------------------------------------
static inline bool finite4(const TLorentzVector& v) {
    return std::isfinite(v.Px()) && std::isfinite(v.Py()) &&
           std::isfinite(v.Pz()) && std::isfinite(v.E());
}

static inline bool nonzeroP(const TLorentzVector& v, double eps=1e-12) {
    const double p2 = v.Px()*v.Px() + v.Py()*v.Py() + v.Pz()*v.Pz();
    return p2 > eps;
}

// ---------------------------------------------------------
// Input file reading
// ---------------------------------------------------------
ReadInput readInputFile(const string &filename) {
    ReadInput input;
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "ERROR: cannot open " << filename << " for reading." << endl;
        return input;
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        string key;
        if (!(iss >> key)) continue;
        if (key[0] == '#') continue;

        // strip possible trailing colon
        if (!key.empty() && key.back() == ':')
            key.pop_back();

        if (key == "num_events") {
            iss >> input.num_events;
        } else if (key == "beam_energy") {
            iss >> input.beam_energy;
        } else if (key == "target_pid") {
            iss >> input.target_pid;
        } else if (key == "Q2_range") {
            double min,max; iss >> min >> max;
            input.Q2_range = {min, max, true};
        } else if (key == "E_range") {
            double min,max; iss >> min >> max;
            input.E_range = {min, max, true};
        } else if (key == "theta_range") {
            double min_deg, max_deg; iss >> min_deg >> max_deg;
            input.theta_range = {min_deg*TMath::DegToRad(),
                                 max_deg*TMath::DegToRad(),
                                 true};
        } else if (key == "W_min") {
            iss >> input.W_min;
        } else if (key == "t_slope") {
            iss >> input.t_slope;
        } else if (key == "reaction") {
            // rest of the line is the reaction string
            std::string rest;
            std::getline(iss, rest);
            // trim leading spaces
            size_t pos = rest.find_first_not_of(" \t");
            if (pos != string::npos) rest = rest.substr(pos);
            input.reaction = rest;
        } else if (key == "print_debug") {
            std::string val; iss >> val;
            input.print_debug = (val == "true" || val == "1");
        } else if (key == "write_lund") {
            std::string val; iss >> val;
            input.write_lund = (val == "true" || val == "1");
        } else if (key == "gen_plots") {
            std::string val; iss >> val;
            input.gen_plots = (val == "true" || val == "1");
        } else if (key.rfind("mass_",0) == 0) {
            // Supports:
            //   mass_<pdg>: <number>
            //   mass_<pdg>: fixed <val>
            //   mass_<pdg>: gauss <val> <width>
            //   mass_<pdg>: uniform <val> <halfwidth>
            //   mass_<pdg>: breitwigner <val> <width>
            //
            // Accepts commas too: "breitwigner 2.5, 0.4"

            string pdg_str = key.substr(5);
            int pdg = stoi(pdg_str);

            // Read the next token; it can be a number or a word
            std::string tok;
            if (!(iss >> tok)) {
                cerr << "WARNING: mass_ spec missing for PDG " << pdg << "\n";
                continue;
            }

            // Strip commas from token (helps with "2.5," style)
            auto stripCommas = [](std::string s){
                s.erase(std::remove(s.begin(), s.end(), ','), s.end());
                return s;
            };
            tok = stripCommas(tok);

            // Try numeric first (backward compatible)
            {
                char* endp = nullptr;
                double val = std::strtod(tok.c_str(), &endp);
                if (endp && *endp == '\0') {
                    // tok was a pure number => fixed mass override
                    customMass[pdg] = val;
                    customMassSpec.erase(pdg); // if a spec existed, remove it
                    continue;
                }
            }

            // Otherwise interpret as function name
            std::string func = tok;

            std::string s_val, s_wid;
            if (!(iss >> s_val)) {
                cerr << "WARNING: mass_ spec needs a value for PDG " << pdg << "\n";
                continue;
            }
            s_val = stripCommas(s_val);

            double val = std::stod(s_val);
            double wid = 0.0;

            // fixed may omit width; others require width
            if (func != "fixed") {
                if (!(iss >> s_wid)) {
                    cerr << "WARNING: mass_ spec needs a width for PDG " << pdg
                        << " (func=" << func << ")\n";
                    continue;
                }
                s_wid = stripCommas(s_wid);
                wid = std::stod(s_wid);
            }

            // Save sampling spec
            customMassSpec[pdg] = MassSpec{func, val, wid, true};
            customMass.erase(pdg); // ensure fixed override doesn't take precedence
        } else {
            cerr << "WARNING: unrecognized key '" << key << "' in " << filename << endl;
        }
    }

    fin.close();
    return input;
}

// ---------------------------------------------------------
// Reaction parsing
// ---------------------------------------------------------
std::map<int, std::vector<int>> parseDecayMap(const std::string& reaction) {
    std::map<int, std::vector<int>> decay_map;
    std::istringstream ss(reaction);
    std::string first_decay, rest;

    std::getline(ss, first_decay, ':'); // skip first block (W daughters)

    while (std::getline(ss, rest, ':')) {
        std::istringstream subtok(rest);
        std::vector<int> pdgs;
        int pdg;
        while (subtok >> pdg) {
            pdgs.push_back(pdg);
            if (subtok.peek() == ',') subtok.ignore();
        }
        if (pdgs.size() > 1) {
            int parent = pdgs[0];
            std::vector<int> daughters(pdgs.begin() + 1, pdgs.end());
            decay_map[parent] = daughters;
        }
    }
    return decay_map;
}

std::vector<int> getFirstDecayDaughters(const std::string& reaction) {
    std::istringstream ss(reaction);
    std::string first_decay;
    std::getline(ss, first_decay, ':');
    std::istringstream subtok(first_decay);
    std::vector<int> pdgs;
    int pdg;
    while (subtok >> pdg) {
        pdgs.push_back(pdg);
        if (subtok.peek() == ',') subtok.ignore();
    }
    return pdgs;
}

// Recursively expand a particle into stable final-state leaves
static void expandToFinalState(int pdg,
                               const std::map<int, std::vector<int>>& decay_map,
                               std::vector<int>& out)
{
    auto it = decay_map.find(pdg);
    if (it == decay_map.end()) {
        out.push_back(pdg); // stable leaf
        return;
    }
    for (int d : it->second) {
        expandToFinalState(d, decay_map, out);
    }
}

// Public: return the full final state implied by the reaction string
std::vector<int> getFinalStateFromReaction(const std::string& reaction)
{
    auto decay_map = parseDecayMap(reaction);
    auto first = getFirstDecayDaughters(reaction);

    std::vector<int> final_pdgs;
    final_pdgs.reserve(16);

    for (int p : first) {
        expandToFinalState(p, decay_map, final_pdgs);
    }
    return final_pdgs;
}

// Threshold computed from the final state (sum of masses)
double getFinalStateThreshold(const std::string& reaction)
{
    auto fs = getFinalStateFromReaction(reaction);
    double thr = 0.0;
    for (int pdg : fs) thr += getMass(pdg);
    return thr;
}

// ---------------------------------------------------------
// EventGenerator class
// ---------------------------------------------------------
class eventGenerator {
public:
    struct ElectroProduction {
        TLorentzVector p_beam;
        TLorentzVector p_target;
        vector<TLorentzVector> v_scattered;
        vector<TLorentzVector> v_virtual;
        vector<TLorentzVector> v_W;
    };

    int num_events;
    double beam_energy;
    double target_mass;
    TRandom3 rnd;

    eventGenerator(int n, double Ebeam, double Mtarget)
        : num_events(n), beam_energy(Ebeam), target_mass(Mtarget), rnd(0) {}

    ElectroProduction generateScatteredElectron(const Range &Q2_range,
                                                const Range &E_range,
                                                const Range &theta_range,
                                                double W_min) {
        ElectroProduction event;
        event.p_beam   = TLorentzVector(0,0,beam_energy, beam_energy);
        event.p_target = TLorentzVector(0,0,0,target_mass);

        while (event.v_scattered.size() < (size_t)num_events) {
            double Q2 = Q2_range.sample(rnd);
            double E_scattered = E_range.sample(rnd);

            double arg = 1.0 - Q2/(2.0*beam_energy*E_scattered);
            if (!isfinite(arg) || arg < -1.0 || arg > 1.0) continue;

            double theta = acos(arg);

            if (theta < theta_range.min || theta > theta_range.max)
                continue;

            double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());

            TLorentzVector p_scattered(E_scattered*sin(theta)*cos(phi),
                                       E_scattered*sin(theta)*sin(phi),
                                       E_scattered*cos(theta),
                                       E_scattered);

            TLorentzVector p_virtual = event.p_beam - p_scattered;
            TLorentzVector p_W       = event.p_beam + event.p_target - p_scattered;

            if (p_W.M() >= W_min) {
                event.v_scattered.push_back(p_scattered);
                event.v_virtual.push_back(p_virtual);
                event.v_W.push_back(p_W);
            }
        }

        return event;
    }

    // t-weighted 2-body decay in parent rest frame, boosted to lab
    DecayResult twoBodyDecayWeighted(const TLorentzVector &parent_lab,
                                     double m1, double m2,
                                     double B,
                                     const TLorentzVector &p_virtual_lab,
                                     int nCandidates = 10000) {
        double M = parent_lab.M();
        TLorentzVector nanv(0,0,0,0);
        if (M <= m1+m2) return {nanv,nanv,NAN,NAN};

        double p_mag = twoBodyMomentum(M, m1, m2);
        if (!isfinite(p_mag)) return {nanv,nanv,NAN,NAN};

        // t at cosTheta = ±1
        TLorentzVector d1_plus;  // cosθ = +1
        TLorentzVector d1_minus; // cosθ = -1
        d1_plus.SetPxPyPzE(0,0, p_mag, sqrt(p_mag*p_mag + m1*m1));
        d1_minus.SetPxPyPzE(0,0,-p_mag, sqrt(p_mag*p_mag + m1*m1));

        TVector3 boostToLab = parent_lab.BoostVector();
        d1_plus.Boost(boostToLab);
        d1_minus.Boost(boostToLab);

        double t_plus  = invariantSquare(p_virtual_lab,d1_plus,false);
        double t_minus = invariantSquare(p_virtual_lab,d1_minus,false);

        double t_min = std::min(t_plus,t_minus);
        double t_max = std::max(t_plus,t_minus);

        if (!isfinite(t_min) || !isfinite(t_max) || t_max <= t_min)
            return {nanv,nanv,NAN,NAN};

        double cosT;
        if (B > 0) {
            vector<double> t_candidates(nCandidates), weights(nCandidates);
            double weight_sum = 0.0;
            for (int i=0; i<nCandidates; ++i) {
                t_candidates[i] = rnd.Uniform(t_min, t_max);
                weights[i]      = exp(-B * fabs(t_candidates[i] - t_min));
                weight_sum     += weights[i];
            }
            for (auto &w : weights) w /= weight_sum;

            double r = rnd.Uniform();
            double cumsum = 0.0;
            int pick_idx = 0;
            for (; pick_idx<nCandidates; ++pick_idx) {
                cumsum += weights[pick_idx];
                if (r <= cumsum) break;
            }
            double t_sampled = t_candidates[std::min(pick_idx, nCandidates-1)];

            // Map t_sampled linearly back to cosθ
            double frac = (t_sampled - t_min) / (t_max - t_min); // 0→t_min,1→t_max
            cosT = 1.0 - 2.0 * frac; // t_min -> cosT=1, t_max -> cosT=-1
            cosT = std::clamp(cosT, -1.0, 1.0);
        } else {
            cosT = rnd.Uniform(-1.0,1.0);
        }

        double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());
        double sinT = sqrt(std::max(0.0,1.0 - cosT*cosT));

        TLorentzVector d1_rest;
        d1_rest.SetPxPyPzE(p_mag*sinT*cos(phi),
                           p_mag*sinT*sin(phi),
                           p_mag*cosT,
                           sqrt(p_mag*p_mag + m1*m1));
        TLorentzVector d2_rest(-d1_rest.Vect(), sqrt(p_mag*p_mag + m2*m2));

        // boost to lab
        d1_rest.Boost(boostToLab);
        d2_rest.Boost(boostToLab);

        return {d1_rest, d2_rest, t_min, t_max};
    }

    // 3-body phase space decay using TGenPhaseSpace
    std::vector<TLorentzVector> threeBodyDecay(const TLorentzVector &parent_lab,
                                               double m1, double m2, double m3) {
        std::vector<TLorentzVector> daughters(3);
        double masses[3] = {m1, m2, m3};

        TGenPhaseSpace gen;
        TLorentzVector parent_copy = parent_lab;
        if (!gen.SetDecay(parent_copy, 3, masses)) {
            return {}; // empty, kinematically impossible
        }

        gen.Generate(); // uniform in phase space
        for (int i=0; i<3; ++i) {
            daughters[i] = *gen.GetDecay(i);
        }
        return daughters;
    }
};

// ---------------------------------------------------------
// Recursive decay (CHANGED: returns bool and validates)
// ---------------------------------------------------------
bool performDecay(const TLorentzVector& parent_lab, int parent_pdg,
                  const std::map<int, std::vector<int>>& decay_map,
                  eventGenerator& gen,
                  std::vector<std::pair<int, TLorentzVector>>& final_particles,
                  const TLorentzVector& p_virtual_lab,
                  double t_slope_for_this_vertex)
{
    // Reject invalid/placeholder parents (prevents (0,0,0,0) from being saved)
    if (!finite4(parent_lab)) return false;
    if (parent_lab.E() <= 0.0) return false;
    if (parent_lab.M2() < -1e-6) return false;

    auto it = decay_map.find(parent_pdg);
    if (it == decay_map.end()) {
        // Stable particle: require nonzero momentum (except photons)
        if (parent_pdg != 22 && !nonzeroP(parent_lab)) return false;
        final_particles.emplace_back(parent_pdg, parent_lab);
        return true;
    }

    const std::vector<int>& daughters = it->second;

    if (daughters.size() == 2) {
        double m1 = getMass(daughters[0]);
        double m2 = getMass(daughters[1]);
        if (m1 <= 0.0 || m2 <= 0.0) return false;

        auto decay = gen.twoBodyDecayWeighted(parent_lab, m1, m2,
                                              t_slope_for_this_vertex,
                                              p_virtual_lab);

        // twoBodyDecayWeighted signals failure via NaNs / zero vector
        if (!finite4(decay.d1_lab) || !finite4(decay.d2_lab)) return false;
        if (decay.d1_lab.E() <= 0.0 || decay.d2_lab.E() <= 0.0) return false;

        if (!performDecay(decay.d1_lab, daughters[0], decay_map, gen,
                          final_particles, p_virtual_lab, 0.0)) return false;

        if (!performDecay(decay.d2_lab, daughters[1], decay_map, gen,
                          final_particles, p_virtual_lab, 0.0)) return false;

        return true;

    } else if (daughters.size() == 3) {
        double m1 = getMass(daughters[0]);
        double m2 = getMass(daughters[1]);
        double m3 = getMass(daughters[2]);
        if (m1 <= 0.0 || m2 <= 0.0 || m3 <= 0.0) return false;

        auto outs = gen.threeBodyDecay(parent_lab, m1, m2, m3);
        if (outs.size() != 3) return false; // kinematically invalid => fail event

        for (int i=0; i<3; ++i) {
            if (!finite4(outs[i]) || outs[i].E() <= 0.0) return false;
        }

        for (int i=0; i<3; ++i) {
            if (!performDecay(outs[i], daughters[i], decay_map, gen,
                              final_particles, p_virtual_lab, 0.0)) return false;
        }
        return true;

    } else {
        cerr << "WARNING: N-body decays with N=" << daughters.size()
             << " not implemented (parent=" << parent_pdg << ")\n";
        // Treat as failure rather than writing garbage
        return false;
    }
}

// ---------------------------------------------------------
// Main event-generation driver
// ---------------------------------------------------------
void runEventGenerator() {
    auto input = readInputFile("input.txt");

    if (input.num_events <= 0) {
        cerr << "ERROR: num_events <= 0 in input file.\n";
        return;
    }
    if (input.reaction.empty()) {
        cerr << "ERROR: reaction not specified in input file.\n";
        return;
    }

    double target_mass = getMass(input.target_pid);
    cout << "Initializing event generator with Ebeam=" << input.beam_energy
         << " GeV, target PID=" << input.target_pid
         << " (mass=" << target_mass << " GeV)\n";

    eventGenerator gen(input.num_events, input.beam_energy, target_mass);

    // Histos (optional)
    TH1D *h_e_theta = new TH1D("h_e_theta",
                               "Scattered electron #theta; #theta [deg]; Counts",
                               100, 0.0, 180.0);
    TH1D *h_W       = new TH1D("h_W",
                               "Invariant Mass W; W [GeV]; Counts",
                               100, 0.0, 6.0);

    TH1D *h_X = new TH1D("h_X",
                     "M(X); M_{X} [GeV]; Counts",
                     100, 1.5, 3.5);

    TH1D *h_int = new TH1D("h_int",
                     "M(p pbar); Mass [GeV]; Counts",
                     100, 1.5, 3.5);

    TLorentzVector p_target(0,0,0,target_mass);
    TLorentzVector p_beam(0,0,input.beam_energy,input.beam_energy);

    auto electronEvents = gen.generateScatteredElectron(input.Q2_range,
                                                        input.E_range,
                                                        input.theta_range,
                                                        input.W_min);

    cout << "Generated " << electronEvents.v_scattered.size()
         << " events passing W_min = " << input.W_min << endl;

    // Parse reaction
    auto parents   = getFirstDecayDaughters(input.reaction);
    auto decay_map = parseDecayMap(input.reaction);

    cout << "First W decay: W -> ";
    for (size_t i=0; i<parents.size(); ++i) {
        cout << parents[i];
        if (i+1 < parents.size()) cout << " + ";
    }
    cout << endl;

    for (const auto& entry : decay_map) {
        cout << "Decay: " << entry.first << " -> ";
        for (size_t i=0; i<entry.second.size(); ++i) {
            cout << entry.second[i];
            if (i+1 < entry.second.size()) cout << " + ";
        }
        cout << endl;
    }

    double threshold = getFinalStateThreshold(input.reaction);
    cout << "Final-state threshold: " << threshold << " GeV\n";
    if (input.W_min < threshold) {
        cout << "WARNING: W_min < final-state threshold. No events will be generated.\n";
        return;
    }

    std::vector<std::vector<std::pair<int, TLorentzVector>>> all_final_particles;

    // Loop over events
    for (size_t i=0; i<electronEvents.v_W.size(); ++i) {
        std::vector<std::pair<int, TLorentzVector>> final_particles;

        auto first_decay = parents;
        if (first_decay.size() != 2) {
            cerr << "ERROR: Only two-body first decays W->d1+d2 are currently supported.\n";
            return;
        }

        int pdg1 = first_decay[0];
        int pdg2 = first_decay[1];

        double m1 = getMass(pdg1);
        double m2 = 0.0;
        if (!sampleIntermediateAboveThreshold(pdg2, gen.rnd, decay_map, m2)) {
            // could not find a kinematically allowed intermediate mass
            continue; // skip this event
        }

        // W -> pdg1 + pdg2
        // We want t-slope weighting on the gamma* - "meson" leg.
        // Here we assume pdg2 is the "X" or meson-like object.
        auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i],
                                               m2, m1,
                                               input.t_slope,
                                               electronEvents.v_virtual[i]);

        TLorentzVector p1_lab = decay1.d2_lab; // corresponds to m1
        TLorentzVector p2_lab = decay1.d1_lab; // corresponds to m2

        // Recursively decay both daughters (no further t-slope)
        bool ok = true;
        ok &= performDecay(p1_lab, pdg1, decay_map, gen, final_particles,
                           electronEvents.v_virtual[i], 0.0);
        ok &= performDecay(p2_lab, pdg2, decay_map, gen, final_particles,
                           electronEvents.v_virtual[i], 0.0);

        if (!ok) continue; // reject unphysical/failed decays

        // Add scattered electron as final state (sanity-check it)
        if (!finite4(electronEvents.v_scattered[i])) continue;
        if (electronEvents.v_scattered[i].E() <= 0.0) continue;
        if (!nonzeroP(electronEvents.v_scattered[i])) continue;
        final_particles.emplace_back(11, electronEvents.v_scattered[i]);

        // Final pass: no zero-momentum tracks (except photons if any)
        for (const auto& pr : final_particles) {
            const int pid = pr.first;
            const auto& v = pr.second;
            if (!finite4(v) || v.E() <= 0.0) { ok = false; break; }
            if (pid != 22 && !nonzeroP(v))   { ok = false; break; }
        }
        if (!ok) continue;

        all_final_particles.push_back(final_particles);

        // Truth X 4-vector from the first vertex (you already have it):
        TLorentzVector X_lab = p2_lab;                  // X from W -> p + X
        double mX_true = X_lab.M();
        h_X->Fill(mX_true);                             // use this, not m2

        // Collect final-state protons and antiprotons from THIS event
        std::vector<TLorentzVector> protons;
        std::vector<TLorentzVector> pbars;

        for (const auto& pr : final_particles) {
            if (pr.first ==  2212)  protons.push_back(pr.second);
            if (pr.first == -2212)  pbars.push_back(pr.second);
        }

        // Reconstruct: choose (p + pbar) combo closest to truth mX
        if (!protons.empty() && !pbars.empty()) {
            double best_d = 1e99;
            double best_m = NAN;

            for (const auto& p : protons) {
                for (const auto& pb : pbars) {
                    double m = (p + pb).M();
                    double d = std::abs(m - mX_true);
                    if (d < best_d) { best_d = d; best_m = m; }
                }
            }

            if (std::isfinite(best_m)) {
                h_int->Fill(best_m); // reconstructed M(X) from daughters
            }
        }

        if (input.gen_plots) {
            h_e_theta->Fill(electronEvents.v_scattered[i].Theta() * TMath::RadToDeg());
            h_W->Fill(electronEvents.v_W[i].M());
        }

        if (input.print_debug) {
            cout << "[DEBUG] Event " << i << ":\n";
            cout << "  W: (" << electronEvents.v_W[i].Px() << ", "
                              << electronEvents.v_W[i].Py() << ", "
                              << electronEvents.v_W[i].Pz() << ", "
                              << electronEvents.v_W[i].E()  << ")\n";
            for (size_t j=0; j<final_particles.size(); ++j) {
                cout << "  final[" << j << "]: PDG=" << final_particles[j].first
                     << " P=(" << final_particles[j].second.Px() << ", "
                     << final_particles[j].second.Py() << ", "
                     << final_particles[j].second.Pz() << ", "
                     << final_particles[j].second.E()  << ")\n";
            }
        }
    }

    cout << "Finished processing decays." << endl;

    // -----------------------------------------------------
    // Write LUND file
    // -----------------------------------------------------
    if (input.write_lund) {
        cout << "Creating LUND file..." << endl;
        ofstream fout("events.lund");
        if (!fout.is_open()) {
            cerr << "ERROR: cannot open events_signal.lund for writing." << endl;
        } else {
            int nEvents = (int)all_final_particles.size();
            for (int i=0; i<nEvents; ++i) {
                // Belt + suspenders: skip any event that still contains junk
                bool event_ok = true;
                for (const auto& pr : all_final_particles[i]) {
                    const int pid = pr.first;
                    const auto& v = pr.second;
                    if (!finite4(v) || v.E() <= 0.0) { event_ok = false; break; }
                    if (pid != 22 && !nonzeroP(v))   { event_ok = false; break; }
                }
                if (!event_ok) continue;

                int num_particles = (int)all_final_particles[i].size();
                fout << "\t" << num_particles
                     << " 1 1 0 0 11 " << input.beam_energy
                     << " 2212 " << PDG::proton << " 0\n";

                double vz_rand = gen.rnd.Uniform(-5.0, 0.0);

                for (size_t j=0; j<all_final_particles[i].size(); ++j) {
                    int pid = all_final_particles[i][j].first;
                    TLorentzVector &v = all_final_particles[i][j].second;
                    fout << j+1 << " "
                         << 0 << " " << 1 << " " << pid << " 0 0 "
                         << std::fixed << std::setprecision(6)
                         << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                         << getMass(pid) << " " << vz_rand << " " << 0.0 << "\n";
                }
                fout << "\n";
            }
            fout.close();
            cout << "Written events.lund" << endl;
        }
    } else {
        cout << "Skipping LUND file creation." << endl;
    }

    // -----------------------------------------------------
    // Simple plots, if requested
    // -----------------------------------------------------
    if (input.gen_plots) {
        for (const auto& e : electronEvents.v_scattered) {
            h_e_theta->Fill(e.Theta() * TMath::RadToDeg());
        }

        cout << "Generating plots..." << endl;
        TCanvas *c1 = new TCanvas("c1", "Scattered Electron Theta", 800, 600);
        h_e_theta->Draw();

        TCanvas *c2 = new TCanvas("c2", "Invariant Mass W", 800, 600);
        h_W->Draw();

        TCanvas *c3 = new TCanvas("c3", "All Plots", 800, 600);
        h_X->SetLineColor(kRed);
        h_X->Draw();
        h_int->Draw("SAMEE");
    }

    cout << "Event generation complete." << endl;
}