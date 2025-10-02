#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TMath.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
// #include <map>
// #include <sstream>

using namespace std;

// PDG masses
struct PDG {
    static constexpr double proton   = 0.9382720813;
    static constexpr double pion     = 0.13957039;
    static constexpr double electron = 0.0005109989461;
    static constexpr double Kp       = 0.493677;
    static constexpr double Xi       = 1.322;
    static constexpr double photon   = 0.0;
    static constexpr double neutron  = 0.9395654133;
    static constexpr double lambda  = 1.115683;
};

// Two-body decay result
struct DecayResult {
    TLorentzVector d1_lab;
    TLorentzVector d2_lab;
    double t_min;
    double t_max;
};

struct Range {
        double min;
        double max;
        bool is_range; // true = sample uniformly, false = single value
        double sample(TRandom3 &rnd) const {
            return is_range ? rnd.Uniform(min, max) : min;
        }
};

struct ReadInput {
    int num_events;
    double beam_energy;
    double target_pid;
    Range Q2_range;
    Range E_range;
    Range theta_range;
    double W_min;
    double t_slope;
    std::string reaction;
    bool print_debug = false;
    bool write_lund = true;
    bool gen_plots = false;
    std::string mass_X = "2.5"; // default mass of X in GeV
};


// Helper functions
double twoBodyMomentum(double M, double m1, double m2) {
    if(M <= m1+m2) return NAN;
    double term = (M*M - (m1+m2)*(m1+m2))*(M*M - (m1-m2)*(m1-m2));
    return 0.5/M * sqrt(term);
}

double invariantSquare(const TLorentzVector &a, const TLorentzVector &b, bool use_sum=false) {
    TLorentzVector d = use_sum ? (a+b) : (a-b);
    return d.M2();
}

ReadInput readInputFile(const string &filename) {
        ReadInput input;
        ifstream fin(filename);
        if (!fin.is_open()) {
            cerr << "ERROR: cannot open " << filename << " for reading." << endl;
            return input;
        }
        string line;
        while (getline(fin, line)) {
            istringstream iss(line);
            string key;
            if (!(iss >> key)) { continue; } // skip empty lines
            if (key[0] == '#') { continue; } // skip comments
            if (key == "num_events:") {
                iss >> input.num_events;
            } else if (key == "beam_energy:") {
                iss >> input.beam_energy;
            } else if (key == "target_pid:") {
                iss >> input.target_pid;
            } else if (key == "Q2_range:") {
                double min, max;
                iss >> min >> max;
                input.Q2_range = {min, max, true};
            } else if (key == "E_range:") {
                double min, max;
                iss >> min >> max;
                input.E_range = {min, max, true};
            } else if (key == "theta_range:") {
                double min, max;
                iss >> min >> max;
                input.theta_range = {min*TMath::DegToRad(), max*TMath::DegToRad(), true};
            } else if (key == "W_min:") {
                iss >> input.W_min;
            } else if (key == "t_slope:") {
                iss >> input.t_slope;
            } else if (key == "reaction:") {
                std::getline(iss, input.reaction);
            } else if (key == "print_debug:") {
                std::string val;
                iss >> val;
                input.print_debug = (val == "true" || val == "1");
            } else if (key == "write_lund:") {
                std::string val;
                iss >> val;
                input.write_lund = (val == "true" || val == "1");
            } else if (key == "gen_plots:") {
                std::string val;
                iss >> val;
                input.gen_plots = (val == "true" || val == "1");
            } else if (key == "mass_X:") {
                iss >> input.mass_X;

            } else {
                cerr << "WARNING: unrecognized key '" << key << "' in " << filename << endl;
            }
        }
        fin.close();
        return input;
    }

std::map<int, std::vector<int>> parseDecayMap(const std::string& reaction) {
    std::map<int, std::vector<int>> decay_map;
    std::istringstream ss(reaction);
    std::string first_decay, rest;
    std::getline(ss, first_decay, ':'); // skip first decay
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

// Recursively generate the final state from a decay map
void generateDecay(int parent_pdg, const std::map<int, std::vector<int>>& decay_map, std::vector<int>& final_state) {
    auto it = decay_map.find(parent_pdg);
    if (it == decay_map.end()) {
        final_state.push_back(parent_pdg);
    } else {
        for (int daughter : it->second) {
            generateDecay(daughter, decay_map, final_state);
        }
    }
}

// EventGenerator class
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

        while(event.v_scattered.size() < (size_t)num_events) {
            double Q2 = Q2_range.sample(rnd);
            double E_scattered = E_range.sample(rnd);
            double arg = 1.0 - Q2/(2.0*beam_energy*E_scattered);
            if(!isfinite(arg) || arg<-1.0 || arg>1.0) continue;

            double theta = acos(arg);

            if (theta < theta_range.min || theta > theta_range.max) { // theta_range already in radians
                continue;
            }
            double phi   = rnd.Uniform(-TMath::Pi(), TMath::Pi());

            TLorentzVector p_scattered(E_scattered*sin(theta)*cos(phi),
                                       E_scattered*sin(theta)*sin(phi),
                                       E_scattered*cos(theta),
                                       E_scattered);

            TLorentzVector p_virtual = event.p_beam - p_scattered;
            TLorentzVector p_W = event.p_beam + event.p_target - p_scattered;

            if(p_W.M() >= W_min) {
                event.v_scattered.push_back(p_scattered);
                event.v_virtual.push_back(p_virtual);
                event.v_W.push_back(p_W);
            }
        }

        return event;
    }

    DecayResult twoBodyDecayWeighted(const TLorentzVector &parent_lab,
                                     double m1, double m2,
                                     double B,
                                     const TLorentzVector &p_virtual_lab,
                                     int nCandidates = 10000) {
        double M = parent_lab.M();
        TLorentzVector nanv(0,0,0,0);
        if(M <= m1+m2) return {nanv,nanv,NAN,NAN};

        double p_mag = twoBodyMomentum(M,m1,m2);
        if(!isfinite(p_mag)) return {nanv,nanv,NAN,NAN};

        // t_min / t_max at cosTheta = ±1
        TLorentzVector d1_plus; d1_plus.SetPxPyPzE(0,0,p_mag,sqrt(p_mag*p_mag + m1*m1));
        TLorentzVector d1_minus; d1_minus.SetPxPyPzE(0,0,-p_mag,sqrt(p_mag*p_mag + m1*m1));
        TVector3 boostToLab = parent_lab.BoostVector();
        d1_plus.Boost(boostToLab);
        d1_minus.Boost(boostToLab);

        double t_plus  = invariantSquare(p_virtual_lab,d1_plus,false);
        double t_minus = invariantSquare(p_virtual_lab,d1_minus,false);
        double t_min = std::max(t_plus,t_minus);
        double t_max = std::min(t_plus,t_minus);

        if(!isfinite(t_min) || !isfinite(t_max) || t_max>=t_min) return {nanv,nanv,NAN,NAN};

        // Sample cos(theta) using exponential weighting in t
        double cosT;
        if(B>0) {
            vector<double> t_candidates(nCandidates), weights(nCandidates);
            double weight_sum=0;
            for(int i=0;i<nCandidates;i++) {
                t_candidates[i] = rnd.Uniform(t_min,t_max);
                weights[i] = exp(-B*fabs(t_candidates[i]-t_min));
                weight_sum += weights[i];
            }
            for(auto &w : weights) w/=weight_sum;
            double r = rnd.Uniform();
            double cumsum=0; int pick_idx=0;
            for(; pick_idx<nCandidates; ++pick_idx){
                cumsum += weights[pick_idx];
                if(r<=cumsum) break;
            }
            double t_sampled = t_candidates[min(pick_idx,nCandidates-1)];
            cosT = std::clamp(1 - 2.0*(t_sampled-t_min)/(t_max-t_min), -1.0, 1.0);
        } else {
            cosT = rnd.Uniform(-1.0,1.0);
        }

        double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());
        double sinT = sqrt(max(0.0,1.0-cosT*cosT));

        TLorentzVector d1_rest;
        d1_rest.SetPxPyPzE(p_mag*sinT*cos(phi),
                            p_mag*sinT*sin(phi),
                            p_mag*cosT,
                            sqrt(p_mag*p_mag + m1*m1));
        TLorentzVector d2_rest(-d1_rest.Vect(), sqrt(p_mag*p_mag + m2*m2));
        d1_rest.Boost(boostToLab);
        d2_rest.Boost(boostToLab);

        return {d1_rest,d2_rest,t_min,t_max};
    }

};

// Lookup mass by PDG code (expand as needed)
double getMass(int pdg) {
    if (pdg == 2212) return PDG::proton;
    if (pdg == -2212) return PDG::proton;
    if (pdg == 11) return PDG::electron;
    if (pdg == 211) return PDG::pion;
    if (pdg == -211) return PDG::pion;
    if (pdg == 321) return PDG::Kp;
    if (pdg == -321) return PDG::Kp;
    if (pdg == 3312) return PDG::Xi;
    if (pdg == -3312) return PDG::Xi;
    if (pdg == 2112) return PDG::neutron;
    if (pdg == 9999) return 2.5; // Example for intermediate state
    // Add more as needed
    return 0.0;
}

// Recursively decay a particle
void performDecay(const TLorentzVector& parent_lab, int parent_pdg,
                  const std::map<int, std::vector<int>>& decay_map,
                  eventGenerator& gen,
                  std::vector<std::pair<int, TLorentzVector>>& final_particles,
                  const TLorentzVector& p_virtual_lab,
                  double t_slope) {
    auto it = decay_map.find(parent_pdg);
    if (it == decay_map.end()) {
        // Stable particle, add to final state
        final_particles.emplace_back(parent_pdg, parent_lab);
    } else if (it->second.size() == 2) {
        // Two-body decay
        double m1 = getMass(it->second[0]);
        double m2 = getMass(it->second[1]);
        auto decay = gen.twoBodyDecayWeighted(parent_lab, m1, m2, t_slope, p_virtual_lab);
        performDecay(decay.d1_lab, it->second[0], decay_map, gen, final_particles, p_virtual_lab, t_slope);
        performDecay(decay.d2_lab, it->second[1], decay_map, gen, final_particles, p_virtual_lab, t_slope);
    }
    // For more than two daughters, generalize as needed
}

// Get the first decay daughters from the reaction string
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
    auto decay_map = parseDecayMap(reaction);
    auto first_decay = getFirstDecayDaughters(reaction);
    double threshold = 0.0;
    for (int pdg : first_decay) {
        threshold += getMass(pdg);
    }
    return threshold;
}

void runEventGenerator() {
    cout << "Reading input file..." << endl;
    auto input = readInputFile("input.txt");

    cout << "Initializing event generator..." << endl;

    eventGenerator gen(input.num_events, input.beam_energy, getMass(input.target_pid));

    TH1D *h_e_theta    = new TH1D("h_e_theta", "Scattered electron #theta; #theta [rad]; Counts", 100, 0.0, 180.0);

    TLorentzVector p_target = TLorentzVector(0, 0, 0, getMass(input.target_pid)); // Target at rest
    TLorentzVector p_beam = TLorentzVector(0, 0, input.beam_energy, input.beam_energy);
    auto electronEvents = gen.generateScatteredElectron(input.Q2_range, input.E_range, input.theta_range, input.W_min);

    cout << "Generated " << electronEvents.v_scattered.size() << " events passing W_min = " << input.W_min << endl;

    TH1D *h_t_gamma_X = new TH1D("h_t_gamma_X", "t: #gamma^{*} - X; t [GeV^{2}]; Counts", 100, 0, 5);
    TH1D *h_t_p_p1      = new TH1D("h_t_p_p1",       "t: p_{t} - p; t [GeV^{2}]; Counts", 100, 0, 5);
    TH1D *h_m_X        = new TH1D("h_m_X",         "Invariant Mass of X; M_{X} [GeV]; Counts", 100, 0, 6.0);
    TH1D *h_m_X_calc   = new TH1D("h_m_X_calc",    "Calculated Invariant Mass of X from daughters; M_{X} [GeV]; Counts", 100, 0, 6.0);

    cout << "Processing decays..." << endl;

    // Parse the decay map from the reaction string in your input file
    auto parents = getFirstDecayDaughters(input.reaction);

    auto decay_map = parseDecayMap(input.reaction);
    cout << "Decay: W -> ";
    for (size_t i = 0; i < parents.size(); ++i) {
            cout << parents[i];
            if (i < parents.size() - 1)
                cout << " + ";
            }
            cout << endl;
            
    for (const auto& entry : decay_map) {
        cout << "Decay: " << entry.first << " -> ";
        for (size_t i = 0; i < entry.second.size(); ++i) {
            cout << entry.second[i];
            if (i < entry.second.size() - 1)
                cout << " + ";
            }
            cout << endl;
    }
    double threshold = getReactionThreshold(input.reaction);
    cout << "Reaction threshold: " << threshold << " GeV" << endl;
    if (input.W_min < threshold) {
        cout << "WARNING: W_min < reaction threshold. No events will be generated." << endl;
        return;
    }
    std::vector<std::vector<std::pair<int, TLorentzVector>>> all_final_particles;
    std::vector<double> all_t_max;
    std::vector<double> all_t_min;
    // // For each event:
    for(size_t i=0; i<electronEvents.v_W.size(); i++) {
        std::vector<std::pair<int, TLorentzVector>> final_particles;

        // First decay: W -> pdg1 + pdg2
        auto first_decay = getFirstDecayDaughters(input.reaction);
        if (first_decay.size() != 2) {
            std::cerr << "ERROR: Only two-body first decays are supported." << std::endl;
            return;
        }
        int pdg1 = first_decay[0];
        int pdg2 = first_decay[1];
        // cout << "Event " << i+1 << " first decay: " << pdg1 << " + " << pdg2 << endl;
        double m1 = getMass(pdg1);
        double m2 = getMass(pdg2);

        // TF1 *mass_func = new TF1("mass_X_func", input.mass_X.c_str(), 0, (p_beam + p_target).M());
        // double mass_X = mass_func->GetRandom();

        // First decay: W -> pdg1 + pdg2
        // Obtain decay products in lab frame
        // for t slope weighting, make sure the particle on the meson vertex is the one we pass in as m1
        // (e.g. if reaction is 2212, 9999: 9999, 321, 3312, then m1 = m_K, m2 = m_Xi)
        // and t is calculated as t = (p_virtual - p_m2)^2 if m2 is the particle on the meson vertex
        auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i], m2, m1, input.t_slope, electronEvents.v_virtual[i]);
        TLorentzVector p1_lab = decay1.d2_lab;
        TLorentzVector p2_lab = decay1.d1_lab;
        double t_max = decay1.t_max;
        double t_min = decay1.t_min;


        // Recursively decay both daughters
        performDecay(p1_lab, pdg1, decay_map, gen, final_particles, electronEvents.v_virtual[i], 0);
        performDecay(p2_lab, pdg2, decay_map, gen, final_particles, electronEvents.v_virtual[i], 0);
        final_particles.emplace_back(11, electronEvents.v_scattered[i]); // Add scattered electron
        all_final_particles.push_back(final_particles);
        all_t_max.push_back(t_max);
        all_t_min.push_back(t_min);
        if (input.print_debug) {
            // Print out modular method debug output in the same structure as the old method
            std::cout << "[MODULAR] Event " << i << ":\n";
            std::cout << "  W:    (" << electronEvents.v_W[i].Px() << ", " << electronEvents.v_W[i].Py() << ", " << electronEvents.v_W[i].Pz() << ", " << electronEvents.v_W[i].E() << ")\n";
            std::cout << "  p1:   (" << p1_lab.Px() << ", " << p1_lab.Py() << ", " << p1_lab.Pz() << ", " << p1_lab.E() << ") [PDG: " << pdg1 << "]\n";
            std::cout << "  p1_test: (" << final_particles[0].second.Px() << ", " << final_particles[0].second.Py() << ", " << final_particles[0].second.Pz() << ", " << final_particles[0].second.E() << ") [PDG: " << final_particles[0].first << "]\n";
            std::cout << "  X:   (" << p2_lab.Px() << ", " << p2_lab.Py() << ", " << p2_lab.Pz() << ", " << p2_lab.E() << ") [PDG: " << pdg2 << "]\n";

            // If the decay map contains 9999, print its daughters
            auto it = decay_map.find(9999);
            if (it != decay_map.end() && it->second.size() == 2) {
                // Find the TLorentzVectors for the daughters of 9999
                TLorentzVector daughter1_lab, daughter2_lab;
                daughter1_lab = final_particles[1].second; // Assuming first daughter is first in final_particles
                daughter2_lab = final_particles[2].second; // Assuming second daughter is second in final_particles
                std::cout << "  p2:   (" << daughter1_lab.Px() << ", " << daughter1_lab.Py() << ", " << daughter1_lab.Pz() << ", " << daughter1_lab.E() << ") [PDG: " << it->second[0] << "]\n";
                std::cout << "  pbar: (" << daughter2_lab.Px() << ", " << daughter2_lab.Py() << ", " << daughter2_lab.Pz() << ", " << daughter2_lab.E() << ") [PDG: " << it->second[1] << "]\n";
            }

            std::cout << "  e-:   (" << electronEvents.v_scattered[i].Px() << ", " << electronEvents.v_scattered[i].Py() << ", " << electronEvents.v_scattered[i].Pz() << ", " << electronEvents.v_scattered[i].E() << ")\n";
        }

    }

    cout << "Finished processing decays." << endl;
    // Write to LUND file as necessary...
    if (input.write_lund) {
        cout << "Creating LUND file..." << endl;
        ofstream fout2("events.lund");
        if (!fout2.is_open()) {
            cerr << "ERROR: cannot open events.lund for writing." << endl;
        } else {
            int nEvents = (int)electronEvents.v_W.size();
            for (int i=0;i<nEvents;++i) {
                // Assuming final_particles is available here
                int num_particles = all_final_particles[i].size(); // +1 for scattered electron
                fout2 << "\t" << num_particles << " 1 1 0 0 11 " << input.beam_energy << " 2212 " << PDG::proton << " 0\n";
                double vz_rand = gen.rnd.Uniform(-5.0, 0.0);
                
                // Write final state particles
                for (size_t j = 0; j < all_final_particles[i].size(); ++j) {
                    int pid = all_final_particles[i][j].first;
                    TLorentzVector &v = all_final_particles[i][j].second;
                    fout2 << j+1 << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                    fout2 << std::fixed << std::setprecision(6)
                         << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                         << getMass(pid) << " " << vz_rand << " " << 0.0 << "\n";
                }
                
                fout2 << "\n";
            }
            fout2.close();
            cout << "Written events.lund" << endl;
        }
    } else {
        cout << "Skipping LUND file creation." << endl;
    }
    // Generate plots if requested
    if (input.gen_plots) {
        for (const auto& event : electronEvents.v_scattered) {
            double theta = event.Theta();
            h_e_theta->Fill(theta*TMath::RadToDeg());
        }
        for (size_t i = 0; i < electronEvents.v_W.size(); ++i) {
            TLorentzVector p_W = electronEvents.v_W[i];
            TLorentzVector p_scattered = electronEvents.v_scattered[i];
            TLorentzVector p_virtual = electronEvents.v_virtual[i];
            TLorentzVector p_proton = p_target; // Target proton at rest
            TLorentzVector p_electron = all_final_particles[i].back().second; // Last particle is scattered electron
            TLorentzVector p_p1 = all_final_particles[i][0].second; // First particle from W decay
            TLorentzVector p_p2 = all_final_particles[i][1].second; // First particle from X decay
            TLorentzVector p_pbar = all_final_particles[i][2].second; // Second particle from X decay
            TLorentzVector p_X = p_p2 + p_pbar; // Assuming X decays to p2, pbar
            double t_gamma_X = invariantSquare(p_virtual, p_X, false);
            double t_p_p1 = invariantSquare(p_target, p_p1, false);
            double u_gamma_p1 = invariantSquare(p_virtual, p_p1, false);
            double u_p_X = invariantSquare(p_target, p_X, false);
            double m_X = p_X.M();
            double m_X_calc = (p_p2 + p_pbar).M();
            double t_max = all_t_max[i];
            double t_min = all_t_min[i];

            h_m_X->Fill(m_X);
            h_m_X_calc->Fill(m_X_calc);
            h_t_p_p1->Fill(-(t_p_p1 - t_min));
            h_t_gamma_X->Fill(-(t_gamma_X - t_min));

            if (input.print_debug) {
                cout << "Event " << i+1 << ": t_gamma_X = " << t_gamma_X << ", t_p_p1 = " << t_p_p1 << ", m_X = " << m_X << ", m_X_calc = " << m_X_calc << endl;
                cout << "  t_min = " << t_min << ", t_max = " << t_max << endl;
                cout << "  u_gamma_p1 = " << u_gamma_p1 << ", u_p_X = " << u_p_X << endl;
            }

        }

        cout << "Generating plots..." << endl;
        TCanvas *c1 = new TCanvas("c1", "Scattered Electron Theta", 800, 600);
        h_e_theta->Draw();
        // c1->SaveAs("scattered_electron_theta.png");
        // delete c1;

        TCanvas *c2 = new TCanvas("c2", "Momentum Transfer", 800, 600);
        TF1 *fit_t = new TF1("fit_t", "[0]*exp(-[1]*x)", 0., 6.0);
        fit_t->SetParameters(1.0, 1.0);
        h_t_gamma_X->SetLineColor(kRed);
        // h_t_gamma_k1->SetLineStyle(2);
        h_t_gamma_X->SetFillStyle(3001);
        h_t_gamma_X->SetFillColor(kRed);
        h_t_gamma_X->Draw();
        h_t_gamma_X->Fit(fit_t, "R");
        fit_t->SetLineColor(kBlue);
        fit_t->Draw("SAME");
        TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
        leg->AddEntry(h_t_gamma_X, "t(#gamma^{*} - X)", "lep");
        leg->AddEntry(h_t_p_p1, "t(p_{t} - p)", "lep");

        leg->AddEntry(fit_t, TString::Format("Fit: B=%.3f±%.3f",
                                    fit_t->GetParameter(1), fit_t->GetParError(1)), "l");
        // leg->AddEntry(fit_t, oss.str().c_str(), "l");
        leg->Draw();
        h_t_gamma_X->SetStats(0);
        h_t_p_p1->SetStats(0);
        c2->Update();

        // c2->cd(2);

        h_t_p_p1->Draw("SAMEE");
        // c2->SaveAs("momentum_transfer.png");
        // delete c2;

        TCanvas *c3 = new TCanvas("c3", "Invariant Mass of X", 800, 600);
        // c3->Divide(2,1);
        // c3->cd(1);
        // h_m_X->Draw();
        // c3->cd(2);
        h_m_X_calc->SetLineColor(kRed);
        h_m_X_calc->SetLineStyle(2);
        h_m_X_calc->SetFillStyle(3001);        
        h_m_X_calc->Draw();
        h_m_X->Draw("SAMEE");

    }
    cout << "Event generation complete." << endl;
    
}

