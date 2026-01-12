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
            // mass_<pdg>: value
            string pdg_str = key.substr(5); // after "mass_"
            int pdg = stoi(pdg_str);
            double mval;
            iss >> mval;
            customMass[pdg] = mval;
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

    // first_decay: initial W → ...
    std::getline(ss, first_decay, ':'); // we don't put this into map; only used to get first daughters

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

// Get the first W daughters from the reaction string
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

double getReactionThreshold(const std::string& reaction) {
    auto first_decay = getFirstDecayDaughters(reaction);
    double threshold = 0.0;
    for (int pdg : first_decay) threshold += getMass(pdg);
    return threshold;
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
// Recursive decay
// ---------------------------------------------------------
void performDecay(const TLorentzVector& parent_lab, int parent_pdg,
                  const std::map<int, std::vector<int>>& decay_map,
                  eventGenerator& gen,
                  std::vector<std::pair<int, TLorentzVector>>& final_particles,
                  const TLorentzVector& p_virtual_lab,
                  double t_slope_for_this_vertex)
{
    auto it = decay_map.find(parent_pdg);
    if (it == decay_map.end()) {
        // Stable particle
        final_particles.emplace_back(parent_pdg, parent_lab);
        return;
    }

    const std::vector<int>& daughters = it->second;
    if (daughters.size() == 2) {
        double m1 = getMass(daughters[0]);
        double m2 = getMass(daughters[1]);
        auto decay = gen.twoBodyDecayWeighted(parent_lab, m1, m2,
                                              t_slope_for_this_vertex,
                                              p_virtual_lab);
        performDecay(decay.d1_lab, daughters[0], decay_map, gen,
                     final_particles, p_virtual_lab, 0.0);
        performDecay(decay.d2_lab, daughters[1], decay_map, gen,
                     final_particles, p_virtual_lab, 0.0);
    } else if (daughters.size() == 3) {
        double m1 = getMass(daughters[0]);
        double m2 = getMass(daughters[1]);
        double m3 = getMass(daughters[2]);
        auto outs = gen.threeBodyDecay(parent_lab, m1, m2, m3);
        if (outs.size() != 3) return; // kinematically invalid, skip

        for (int i=0; i<3; ++i) {
            performDecay(outs[i], daughters[i], decay_map, gen,
                         final_particles, p_virtual_lab, 0.0);
        }
    } else {
        // For >3-body: could be extended with TGenPhaseSpace and N masses
        cerr << "WARNING: N-body decays with N=" << daughters.size()
             << " not implemented (parent=" << parent_pdg << ")\n";
        final_particles.emplace_back(parent_pdg, parent_lab);
    }
}

// ---------------------------------------------------------
// Main event-generation driver
// ---------------------------------------------------------
void runEventGenerator2() {
    // cout << "Reading input file: " << inputFile << endl;
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

    double threshold = getReactionThreshold(input.reaction);
    cout << "Reaction threshold (from first W daughters): " << threshold << " GeV" << endl;
    if (input.W_min < threshold) {
        cout << "WARNING: W_min < reaction threshold. No events will be generated.\n";
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
        double m2 = getMass(pdg2);

        // W -> pdg1 + pdg2
        // We want t-slope weighting on the gamma* - "meson" leg, so
        // we can choose which one gets the weighting. Here we assume
        // pdg2 is the "X" or meson-like object.
        auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i],
                                               m2, m1,
                                               input.t_slope,
                                               electronEvents.v_virtual[i]);
        TLorentzVector p1_lab = decay1.d2_lab; // corresponds to m1
        TLorentzVector p2_lab = decay1.d1_lab; // corresponds to m2

        // Recursively decay both daughters (no further t-slope)
        performDecay(p1_lab, pdg1, decay_map, gen, final_particles,
                     electronEvents.v_virtual[i], 0.0);
        performDecay(p2_lab, pdg2, decay_map, gen, final_particles,
                     electronEvents.v_virtual[i], 0.0);

        // Add scattered electron as final state
        final_particles.emplace_back(11, electronEvents.v_scattered[i]);
        all_final_particles.push_back(final_particles);

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
        ofstream fout("events_back_p_pipi.lund");
        if (!fout.is_open()) {
            cerr << "ERROR: cannot open events_signal.lund for writing." << endl;
        } else {
            int nEvents = (int)all_final_particles.size();
            for (int i=0; i<nEvents; ++i) {
                int num_particles = all_final_particles[i].size();
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
        for (size_t i=0; i<electronEvents.v_W.size(); ++i) {
            h_W->Fill(electronEvents.v_W[i].M());
        }

        cout << "Generating plots..." << endl;
        TCanvas *c1 = new TCanvas("c1", "Scattered Electron Theta", 800, 600);
        h_e_theta->Draw();

        TCanvas *c2 = new TCanvas("c2", "Invariant Mass W", 800, 600);
        h_W->Draw();
    }

    cout << "Event generation complete." << endl;
}
