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
    double target_mass;
    Range Q2_range;
    Range E_range;
    Range theta_range;
    double W_min;
    double t_slope;
    std::string reaction;
    bool print_debug = false;
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
            } else if (key == "target_mass:") {
                iss >> input.target_mass;
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
        double t_min = std::min(t_plus,t_minus);
        double t_max = std::max(t_plus,t_minus);

        if(!isfinite(t_min) || !isfinite(t_max) || t_min>=t_max) return {nanv,nanv,NAN,NAN};

        // Sample cos(theta)
        double cosT = 0.0;
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
            cosT = std::clamp(1.0 - 2.0*(t_sampled-t_min)/(t_max-t_min), -1.0, 1.0);
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


void runEventGenerator() {
    cout << "Reading input file..." << endl;
    auto input = readInputFile("input.txt");

    cout << "Initializing event generator..." << endl;

    eventGenerator gen(input.num_events, input.beam_energy, input.target_mass);

    TH1D *h_e_theta    = new TH1D("h_e_theta", "Scattered electron #theta; #theta [rad]; Counts", 100, 0.0, 180.0);

    TLorentzVector p_target = TLorentzVector(0, 0, 0, input.target_mass);
    TLorentzVector p_beam = TLorentzVector(0, 0, input.beam_energy, input.beam_energy);
    auto electronEvents = gen.generateScatteredElectron(input.Q2_range, input.E_range, input.theta_range, input.W_min);

    cout << "Generated " << electronEvents.v_scattered.size() << " events passing W_min = " << input.W_min << endl;

    // for (const auto& event : electronEvents.v_scattered) {
    //     double theta = event.Theta();
    //     h_e_theta->Fill(theta*TMath::RadToDeg());
    //     cout << "Scattered electron theta (deg): " << theta * TMath::RadToDeg() << endl;
    // }

    // Example of a decay
    vector<TLorentzVector> p1_lab_list; p1_lab_list.reserve(input.num_events);
    vector<TLorentzVector> X_lab_list;  X_lab_list.reserve(input.num_events);
    vector<TLorentzVector> p2_lab_list; p2_lab_list.reserve(input.num_events);
    vector<TLorentzVector> pbar_lab_list; pbar_lab_list.reserve(input.num_events);

    TH1D *h_t_gamma_k1 = new TH1D("h_t_gamma_k1", "t: virtual gamma - K1; t [GeV^{2}]; Counts", 100, 0, 6.0);
    TH1D *h_t_p_X      = new TH1D("h_t_p_X",       "t: proton - X; t [GeV^{2}]; Counts", 100, 0, 6.0);

    cout << "Processing decays..." << endl;
    // double mp = PDG::proton, mpbar = PDG::proton;
    // for(size_t i=0;i<electronEvents.v_W.size();i++){
    //     double mX = 2.5;
    //     if (mX <= (mp + mpbar)) continue;

    //     auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i], mp, mX, input.t_slope, electronEvents.v_virtual[i]);
    //     TLorentzVector p_p1_lab = decay1.d1_lab;
    //     TLorentzVector p_X_lab  = decay1.d2_lab;

    //     if (p_X_lab.E() == 0 && p_p1_lab.P() == 0) continue;
    //     double t_min = decay1.t_min;
    //     double t_max = decay1.t_max;


    //     auto decay2 = gen.twoBodyDecayWeighted(p_X_lab, mp, mpbar, 0, electronEvents.v_virtual[i]);
    //     TLorentzVector p_p2_lab = decay2.d1_lab;
    //     TLorentzVector p_pbar_lab = decay2.d2_lab;

    //     if (p_p2_lab.E() == 0 && p_pbar_lab.E() == 0) continue;

    //     p1_lab_list.push_back(p_p1_lab);
    //     X_lab_list.push_back(p_X_lab);
    //     p2_lab_list.push_back(p_p2_lab);
    //     pbar_lab_list.push_back(p_pbar_lab);

    //     double t1 = invariantSquare(electronEvents.v_virtual[i], p_p1_lab, false);
    //     double t2 = invariantSquare(p_target, p_X_lab, false);

    //     h_t_gamma_k1->Fill(-(t1-t_max));
    //     h_t_p_X->Fill(-(t2-t_max));

    // }
    // cout << "Finished processing decays." << endl;

    // cout << "Creating LUND file..." << endl;
    // // Write to LUND file
    // ofstream fout("events.lund");
    // if (!fout.is_open()) {
    //     cerr << "ERROR: cannot open events.lund for writing." << endl;
    // } else {
    //     int nEvents = (int)p1_lab_list.size();
    //     for (int i=0;i<nEvents;++i) {
    //         int num_particles = 4;
    //         fout << "\t" << num_particles << " 1 1 0 0 11 " << input.beam_energy << " 2212 " << PDG::proton << " 0\n";
    //         double vz_rand = gen.rnd.Uniform(-5.0, 0.0);
    //         // 1) p1
    //         {
    //             int j = 1;
    //             int pid = 2212; // proton
    //             TLorentzVector &v = p1_lab_list[i];
    //             fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
    //             fout << std::fixed << std::setprecision(6)
    //                  << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
    //                  << mp << " " << vz_rand << " " << 0.0 << "\n";
    //         }
    //         // 2) pbar
    //         {
    //             int j = 2;
    //             int pid = -2212; // antiproton
    //             TLorentzVector &v = pbar_lab_list[i];
    //             fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
    //             fout << std::fixed << std::setprecision(6)
    //                  << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
    //                  << mpbar << " " << vz_rand << " " << 0.0 << "\n";
    //         }
    //         // 3) p2
    //         {
    //             int j = 3;
    //             int pid = 2212; // proton
    //             TLorentzVector &v = p2_lab_list[i];
    //             fout << j << " " << 0 << " " << 1 << " "    << pid << " 0 0 ";
    //             fout << std::fixed << std::setprecision(6)
    //                  << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
    //                  << mp << " " << vz_rand << " " << 0.0 << "\n";
    //         }
    //         // 4) scattered electron
    //         {
    //             int j = 4;
    //             int pid = 11;
    //             TLorentzVector &v = electronEvents.v_scattered[i];
    //             fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
    //             fout << std::fixed << std::setprecision(6)
    //                  << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
    //                  << PDG::electron << " " << vz_rand << " " << 0.0 << "\n";
    //         }
    //         fout << "\n";
    //     }
    //     fout.close();
    //     cout << "Written events.lund" << endl;
    // }

    // double fit_min = 0.;
    // double fit_max = 6.0;
    // TF1 *f_exp1 = new TF1("f_exp1", "[0]*exp(-[1]*x)", fit_min, fit_max);
    // f_exp1->SetParameters(1000, 2.0);
    // // h_t1prime->Fit(f_exp1, "RQ", "", fit_min, fit_max);

    // TF1 *f_exp_t1raw = new TF1("f_exp_t1raw", "[0]*exp(-[1]*x)", 0.0, 3.0);
    // f_exp_t1raw->SetParameters(f_exp1->GetParameter(0), f_exp1->GetParameter(1));

    // TF1 *f_exp_gamma_k1 = new TF1("f_exp_gamma_k1", "[0]*exp(-[1]*x)", fit_min, fit_max);
    // f_exp_gamma_k1->SetParameters(100, 2.0);
    // h_t_gamma_k1->Fit(f_exp_gamma_k1, "RQ", "", fit_min, fit_max);

    // TF1 *f_exp_p_X = new TF1("f_exp_p_X", "[0]*exp(-[1]*x)", fit_min, fit_max);
    // f_exp_p_X->SetParameters(100, 2.0);
    // h_t_p_X->Fit(f_exp_p_X, "RQ ", "", fit_min, fit_max);

    // TCanvas *c1 = new TCanvas("c1","t distributions",1200,500);
    // c1->Divide(2,1);
    // c1->cd(1);
    // h_t_gamma_k1->Draw();
    // f_exp_gamma_k1->SetLineColor(kRed);
    // f_exp_gamma_k1->Draw("same");
    // c1->cd(2);
    // h_t_p_X->Draw();
    // f_exp_p_X->SetLineColor(kRed);
    // f_exp_p_X->Draw("same");
    // c1->Update();

    // TCanvas *c2 = new TCanvas("c2","t distributions log",1200,500);
    // h_e_theta->Draw();
    // c2->Update();

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
    // cout << decay_map.size() << " decay entries parsed from reaction string." << endl;
    for (const auto& entry : decay_map) {
        cout << "Decay: " << entry.first << " -> ";
        for (size_t i = 0; i < entry.second.size(); ++i) {
            cout << entry.second[i];
            if (i < entry.second.size() - 1)
                cout << " + ";
            }
            cout << endl;
    }
    std::vector<std::vector<std::pair<int, TLorentzVector>>> all_final_particles;
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

        // First decay: W -> pdg1 + pdg2
        auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i], m1, m2, input.t_slope, electronEvents.v_virtual[i]);
        TLorentzVector p1_lab = decay1.d1_lab;
        TLorentzVector p2_lab = decay1.d2_lab;

        // Recursively decay both daughters
        performDecay(p1_lab, pdg1, decay_map, gen, final_particles, electronEvents.v_virtual[i], 0);
        performDecay(p2_lab, pdg2, decay_map, gen, final_particles, electronEvents.v_virtual[i], 0);
        final_particles.emplace_back(11, electronEvents.v_scattered[i]); // Add scattered electron
        all_final_particles.push_back(final_particles);
        if (input.print_debug) {
            // Print out Initial and Intermediate states for verification
            std::cout << "Sum Intermediate State Daughters: (Px, Py, Pz, E): ("
                    << p1_lab.Px() + p2_lab.Px() << ", "
                    << p1_lab.Py() + p2_lab.Py() << ", "
                    << p1_lab.Pz() + p2_lab.Pz() << ", "
                    << p1_lab.E()  + p2_lab.E()  << ")\n";
            
            std::cout << "Sum Initial State: (Px, Py, Pz, E): ("
                    << electronEvents.v_W[i].Px() << ", " << electronEvents.v_W[i].Py() << ", "
                    << electronEvents.v_W[i].Pz() << ", " << electronEvents.v_W[i].E() << ")\n";

            TLorentzVector daughter1_lab = final_particles[1].second; // Assuming first final particle is daughter1
            TLorentzVector daughter2_lab = final_particles[2].second; // Assuming second final particle is daughter2
            TLorentzVector sum_daughters = daughter1_lab + daughter2_lab;

            // Print out the Intermediate state and sum of daughters
            std::cout << "Parent 9999: (Px, Py, Pz, E): ("
                    << p2_lab.Px() << ", " << p2_lab.Py() << ", " << p2_lab.Pz() << ", " << p2_lab.E() << ")\n";
            std::cout << "Sum daughters: (Px, Py, Pz, E): ("
                    << sum_daughters.Px() << ", " << sum_daughters.Py() << ", "
                    << sum_daughters.Pz() << ", " << sum_daughters.E() << ")\n";
        }
        // Now final_particles contains all final state particles for this event
        if (input.print_debug) {
            std::cout << "Event " << i+1 << " final state particles:" << std::endl;
            for (const auto& p : final_particles) {
                std::cout << "  PDG: " << p.first << ", (Px, Py, Pz, E): (" 
                          << p.second.Px() << ", " << p.second.Py() << ", " 
                          << p.second.Pz() << ", " << p.second.E() << ")" << std::endl;
            }
        }

    }

    cout << "Finished processing decays." << endl;
    // Write to LUND file as necessary...
    // if (input.write_lund) {
        cout << "Creating LUND file..." << endl;
        ofstream fout("events.lund");
        if (!fout.is_open()) {
            cerr << "ERROR: cannot open events.lund for writing." << endl;
        } else {
            int nEvents = (int)electronEvents.v_W.size();
            for (int i=0;i<nEvents;++i) {
                // Assuming final_particles is available here
                int num_particles = all_final_particles[i].size() + 1; // +1 for scattered electron
                fout << "\t" << num_particles << " 1 1 0 0 11 " << input.beam_energy << " 2212 " << PDG::proton << " 0\n";
                double vz_rand = gen.rnd.Uniform(-5.0, 0.0);
                
                // Write final state particles
                for (size_t j = 0; j < all_final_particles[i].size(); ++j) {
                    int pid = all_final_particles[i][j].first;
                    TLorentzVector &v = all_final_particles[i][j].second;
                    fout << j+1 << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                    fout << std::fixed << std::setprecision(6)
                         << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                         << getMass(pid) << " " << vz_rand << " " << 0.0 << "\n";
                }
                
                fout << "\n";
            }
            fout.close();
            cout << "Written events.lund" << endl;
        }
    // } else {
        // cout << "Skipping LUND file creation." << endl;
    // }

    cout << "Event generation complete." << endl;
    
}

