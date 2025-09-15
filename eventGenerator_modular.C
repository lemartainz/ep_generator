// Modular event generator for ep -> final state
// User provides: final state, Q2 range, energy range
// This is a refactored version of eventGenerator.C

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TMath.h>
#include <TVector3.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>
#include <sstream>

using namespace std;

// PDG masses (GeV)
struct PDG {
    static constexpr double proton = 0.9382720813;
    static constexpr double pion  = 0.13957039;
    static constexpr double electron = 0.0005109989461;
    static constexpr double photon = 0.0;
    static constexpr double KL = 0.497611;
    static constexpr double KS = 0.497611;
    static constexpr double Kp = 0.493677;
    static constexpr double neutron = 0.9395654133;
    static constexpr double Xi = 1.322;
};

// Struct for user parameters
struct GeneratorParams {
    std::string final_state; // e.g. "K+ K+ Xi-"
    double Q2_min = 0.0, Q2_max = 0.0;
    double E_min = 0.0, E_max = 0.0;
    int num_events = 0;
};

// Read parameters from a text file
bool readParamsFromFile(const std::string& filename, GeneratorParams& params) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Could not open parameter file: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key)) continue;
        if (key[0] == '#') continue; // skip comments
        if (key == "final_state") {
            std::getline(iss, params.final_state);
            params.final_state.erase(0, params.final_state.find_first_not_of(" \t"));
        } else if (key == "Q2_min") {
            iss >> params.Q2_min;
        } else if (key == "Q2_max") {
            iss >> params.Q2_max;
        } else if (key == "E_min") {
            iss >> params.E_min;
        } else if (key == "E_max") {
            iss >> params.E_max;
        } else if (key == "num_events") {
            iss >> params.num_events;
        }
    }
    fin.close();
    return true;
}

// ...existing code for twoBodyMomentum, invariantSquare, twoBodyDecayWeighted...
// ...modularized event generation and output functions will go here...

double twoBodyMomentum(double M, double m1, double m2) {
    /*
    Calculate the momentum magnitude of a two body decay from a parent particle

    Parameters:
    =========================
    M : Mass of the parent particle
    m1 : Mass of the first daughter particle
    m2 : Mass of the second daughter particle

    Returns:
    =========================
    Momentum magnitude of the two body decay
    */
   if (M <= m1 + m2) return NAN;
    double term = (M*M - (m1 + m2)*(m1 + m2)) * (M*M - (m1 - m2)*(m1 - m2));
    if (term < 0) return NAN;
    return 0.5 / M * sqrt(term);
}

double invariantSquare(const TLorentzVector &a, const TLorentzVector &b, bool use_sum) {
    /* 
    Calculates the invariant mass squared from two four-vectors
    If use_sum is true, returns (a+b)^2, else (a-b)^2 (default)

    Parameters:
    =========================
    a : First four-vector
    b : Second four-vector
    use_sum : If true, use sum (a+b), else subtraction (a-b)

    Returns:
    =========================
    Invariant mass squared
    */
    TLorentzVector d = use_sum ? (a + b) : (a - b);
    return d.M2();
}
TH1D *h_tprime1 = new TH1D("h_tprime1", "t' distribution; t' (GeV^{2}); Events", 100, 0, 6); 
TH1D *h_tprime2 = new TH1D("h_tprime2", "t' distribution; t' (GeV^{2}); Events", 100, 0, 6);



struct DecayResult {
    TLorentzVector d1_lab;
    TLorentzVector d2_lab;
    double t_min;
    double t_max;
};

DecayResult twoBodyDecayWeighted(const TLorentzVector &parent_lab,
                                 double m1, double m2,
                                 double B,
                                 const TLorentzVector &p_virtual_lab,
                                 TRandom3 &rnd,
                                 int nCandidates = 10000) {

    /* 
    Generate two-body decay products with weighting

    Parameters:
    =========================
    parent_lab : Four-momentum of the parent particle in the lab frame
    m1 : Mass of the first daughter particle
    m2 : Mass of the second daughter particle
    B : Exponential decay constant
    p_virtual_lab : Four-momentum of the virtual particle in the lab frame
    rnd : Random number generator
    nCandidates : Number of candidates for the decay

    Returns:
    =========================
    Pair of four-vectors for the two decay products
    */
    double M = parent_lab.M();
    if (M <= m1 + m2) {
        // If the parent mass is less than the sum of daughter masses, return zero vectors
        TLorentzVector nanv(0,0,0,0);
        return {nanv, nanv, NAN, NAN};
    }

    double p_mag = twoBodyMomentum(M, m1, m2);
    if (!isfinite(p_mag)) {
        // If the momentum calculation is invalid, return zero vectors
        TLorentzVector nanv(0,0,0,0);
        return {nanv, nanv, NAN, NAN};
    }

    // Calculate t at cosTheta = ±1
    TLorentzVector d1_rest_plus = [&]() {
        double cosTheta = +1.0;
        double sinTheta = sqrt(max(0.0, 1.0 - cosTheta*cosTheta));
        TLorentzVector v;
        v.SetPxPyPzE(p_mag * sinTheta, 0.0, p_mag * cosTheta, sqrt(p_mag*p_mag + m1*m1));
        return v;
    }();

    TLorentzVector d1_rest_minus = [&]() {
        double cosTheta = -1.0;
        double sinTheta = sqrt(max(0.0, 1.0 - cosTheta*cosTheta));
        TLorentzVector v;
        v.SetPxPyPzE(p_mag * sinTheta, 0.0, p_mag * cosTheta, sqrt(p_mag*p_mag + m1*m1));
        return v;
    }();

    // Boost to Lab Frame
    TVector3 boostToLab = parent_lab.BoostVector();
    d1_rest_plus.Boost(boostToLab);
    d1_rest_minus.Boost(boostToLab);

    // t_min and t_max for costheta = ±1
    double t_plus  = invariantSquare(p_virtual_lab, d1_rest_plus, false);
    double t_minus = invariantSquare(p_virtual_lab, d1_rest_minus, false);

    // t_min -> t_max : t_max -> t_min
    double t_min = std::min(t_plus, t_minus);
    double t_max = std::max(t_plus, t_minus);

    // Skip unphysical events
    if (!isfinite(t_min) || !isfinite(t_max) || t_min >= t_max) {
        // Check whether the reaction is kinematically feasible
        cerr << "Invalid kinematic configuration" << endl;
        TLorentzVector nanv(0,0,0,0);
        return {nanv, nanv, NAN, NAN};
    }

    double cosT = 0.0;

    if (B > 0) {
        // Exponential weighting
        vector<double> t_candidates(nCandidates);
        vector<double> weights(nCandidates);
        double weight_sum = 0.0;

        for (int i=0; i<nCandidates; ++i) {
            // Sample t uniformly
            t_candidates[i] = rnd.Uniform(t_min, t_max);
            // double t_prime = t_candidates[i] - t_min; // <-- t'
            // h_tprime->Fill(t_prime); // Fill histogram for each sampled candidate
            weights[i] = exp(-B * fabs(t_candidates[i] - t_min));
            weight_sum += weights[i];
        }
        for (int i=0; i<nCandidates; ++i) {
            // Normalize weights
            weights[i] /= weight_sum;
        }

        // Weighted selection
        double r = rnd.Uniform(0.0, 1.0);
        double cumsum = 0.0;
        int pick_idx = 0;
        for (; pick_idx < nCandidates; ++pick_idx) {
            cumsum += weights[pick_idx];
            if (r <= cumsum) break;
        }
        if (pick_idx >= nCandidates) pick_idx = nCandidates - 1;

        double t_sampled = t_candidates[pick_idx];
        double t_prime_sampled = t_sampled - t_min;
        h_tprime1->Fill(t_prime_sampled); 
        cosT = 1.0 - 2.0 * (t_sampled - t_min) / (t_max - t_min);
        cosT = std::clamp(cosT, -1.0, 1.0);

    } else {
        // Uniform cos(theta)
        cosT = rnd.Uniform(-1.0, 1.0);
    }

    // Final decay kinematics
    double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());
    double sinT = sqrt(max(0.0, 1.0 - cosT*cosT));

    TLorentzVector d1_rest;
    d1_rest.SetPxPyPzE(p_mag * sinT * cos(phi),
                       p_mag * sinT * sin(phi),
                       p_mag * cosT,
                       sqrt(p_mag*p_mag + m1*m1));

    TLorentzVector d2_rest(-d1_rest.Vect(), sqrt(p_mag*p_mag + m2*m2));

    d1_rest.Boost(boostToLab);
    d2_rest.Boost(boostToLab);

    return {d1_rest, d2_rest, t_min, t_max};
}


// Example input.txt format:
// beam_energy 6.5
// num_events 10000
// final_state K+ K+ Xi-

// Parse a space-separated final state string into a vector
std::vector<std::string> parseFinalState(const std::string& fs) {
    std::istringstream iss(fs);
    std::vector<std::string> out;
    std::string token;
    while (iss >> token) out.push_back(token);
    return out;
}

int main(int argc, char** argv) {
    GeneratorParams params;
    std::string paramfile = "input.txt";
    if (argc > 1) paramfile = argv[1];
    if (!readParamsFromFile(paramfile, params)) {
        std::cerr << "Failed to read parameters. Exiting." << std::endl;
        return 1;
    }
    std::vector<std::string> final_state_particles = parseFinalState(params.final_state);
    // Example: print parsed reaction
    std::cout << "Beam energy: " << params.beam_energy << " GeV\n";
    std::cout << "Number of events: " << params.num_events << "\n";
    std::cout << "Final state:";
    for (const auto& p : final_state_particles) std::cout << " " << p;
    std::cout << std::endl;
    // Call modular event generator (to be implemented)
    // generate_events(params, final_state_particles);
    return 0;
}

// Main macro function
void eventGenerator_modular() {

    
    const double beam_energy = 10.2; // GeV
    const double target_mass = PDG::proton;
    const int num_events_target = 100000;
    const double W_cut = 4;
    const double B_k1 = 1;
    const double B_X  = 0;
    const double mK = PDG::Kp;
    const double mXi = PDG::Xi;
    TRandom3 rnd(0);

    cout << "EventGenerator: Starting generation of " << num_events_target << " events!" << endl;

    TLorentzVector p_beam(0,0,beam_energy, beam_energy);
    TLorentzVector p_target(0,0,0, target_mass);

    vector<TLorentzVector> v_scattered; v_scattered.reserve(num_events_target);
    vector<TLorentzVector> v_virtual;   v_virtual.reserve(num_events_target);
    vector<TLorentzVector> v_W;         v_W.reserve(num_events_target);

    int batch = max(4 * num_events_target, 10000);
    while ((int)v_scattered.size() < num_events_target) {
        for (int i=0;i<batch && (int)v_scattered.size() < num_events_target; ++i) {
            double Q2 = rnd.Uniform(0., 6); // fixed Q2
            double E_scattered = rnd.Uniform(0.0, 6.5);
            double arg = 1.0 - Q2 / (2.0 * beam_energy * E_scattered);
            if (!isfinite(arg) || arg < -1.0 || arg > 1.0) continue;
            double theta = acos(arg);
            double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());

            double px = E_scattered * sin(theta) * cos(phi);
            double py = E_scattered * sin(theta) * sin(phi);
            double pz = E_scattered * cos(theta);
            TLorentzVector p_scattered(px, py, pz, E_scattered);

            TLorentzVector p_virtual = p_beam - p_scattered;
            TLorentzVector p_Wvec = p_beam + p_target - p_scattered;

            double W = p_Wvec.M();
            if (W >= W_cut) {
                v_scattered.push_back(p_scattered);
                v_virtual.push_back(p_virtual);
                v_W.push_back(p_Wvec);
            }
        }
    }

    if ((int)v_scattered.size() > num_events_target) {
        v_scattered.resize(num_events_target);
        v_virtual.resize(num_events_target);
        v_W.resize(num_events_target);
    }

    cout << "Collected " << v_scattered.size() << " events passing W >= " << W_cut << endl;

    vector<TLorentzVector> K1_lab_list; K1_lab_list.reserve(num_events_target);
    vector<TLorentzVector> X_lab_list;  X_lab_list.reserve(num_events_target);
    vector<TLorentzVector> K2_lab_list; K2_lab_list.reserve(num_events_target);
    vector<TLorentzVector> Xi_lab_list; Xi_lab_list.reserve(num_events_target);

    TH1D *h_t_gamma_k1 = new TH1D("h_t_gamma_k1", "t: virtual gamma - K1; t [GeV^{2}]; Counts", 100, 0, 6.0);
    TH1D *h_t_p_X      = new TH1D("h_t_p_X",       "t: proton - X; t [GeV^{2}]; Counts", 100, 0, 6.0);

    for (int ie=0; ie < (int)v_scattered.size(); ++ie) {
        const TLorentzVector &p_sc = v_scattered[ie];
        const TLorentzVector &p_v  = v_virtual[ie];
        const TLorentzVector &pW   = v_W[ie];

        double mX = 2.5;
        if (mX <= (mK + mXi)) continue;

        auto k1_x_pair = twoBodyDecayWeighted(pW, mK, mX, B_k1, p_v, rnd, 2000);
        TLorentzVector p_k1_lab = k1_x_pair.d1_lab;
        TLorentzVector p_X_lab  = k1_x_pair.d2_lab;
        double t_min = k1_x_pair.t_min;
        double t_max = k1_x_pair.t_max;
        // cout << "t_min: " << t_min << ", t_max: " << t_max << endl;

        if (p_k1_lab.E() == 0 && p_X_lab.E() == 0) continue;

        auto k2_xi_pair = twoBodyDecayWeighted(p_X_lab, mK, mXi, B_X, p_v, rnd, 2000);
        TLorentzVector p_k2_lab = k2_xi_pair.d1_lab;
        TLorentzVector p_xi_lab = k2_xi_pair.d2_lab;

        if (p_k2_lab.E() == 0 && p_xi_lab.E() == 0) continue;

        K1_lab_list.push_back(p_k1_lab);
        X_lab_list.push_back(p_X_lab);
        K2_lab_list.push_back(p_k2_lab);
        Xi_lab_list.push_back(p_xi_lab);

        double t1 = invariantSquare(p_v, p_k1_lab, false);
        double t2 = invariantSquare(p_target, p_X_lab, false);

        h_t_gamma_k1->Fill(-(t1-t_max));
        h_t_p_X->Fill(-(t2-t_max));
    }

    cout << "After decays: produced " << K1_lab_list.size() << " full decay chains." << endl;

    double t1_min_val = 1e99;
    for (int b=1; b<=h_t_gamma_k1->GetNbinsX(); ++b) {
        double c = h_t_gamma_k1->GetBinContent(b);
        if (c > 0) {
            double center = h_t_gamma_k1->GetBinCenter(b);
            if (center < t1_min_val) t1_min_val = center;
        }
    }
    if (!isfinite(t1_min_val)) t1_min_val = 0.0;

    // TH1D *h_t1prime = new TH1D("h_t1prime", "t'_{#gamma^*,K^+}; t' [GeV^{2}]; Counts", 100, 0, 3.0);
    // for (int b=1; b<=h_t_gamma_k1->GetNbinsX(); ++b) {
    //     double c = h_t_gamma_k1->GetBinContent(b);
    //     double center = h_t_gamma_k1->GetBinCenter(b);
    //     double tprime = center - t1_min_val;
    //     if (tprime >= 0 && c > 0) {
    //         h_t1prime->Fill(tprime, c);
    //     }
    // }

    double fit_min = 0.;
    double fit_max = 6.0;
    TF1 *f_exp1 = new TF1("f_exp1", "[0]*exp(-[1]*x)", fit_min, fit_max);
    f_exp1->SetParameters(1000, 2.0);
    // h_t1prime->Fit(f_exp1, "RQ", "", fit_min, fit_max);

    TF1 *f_exp_t1raw = new TF1("f_exp_t1raw", "[0]*exp(-[1]*x)", 0.0, 3.0);
    f_exp_t1raw->SetParameters(f_exp1->GetParameter(0), f_exp1->GetParameter(1));

    TF1 *f_exp_gamma_k1 = new TF1("f_exp_gamma_k1", "[0]*exp(-[1]*x)", fit_min, fit_max);
    f_exp_gamma_k1->SetParameters(100, 2.0);
    h_t_gamma_k1->Fit(f_exp_gamma_k1, "RQ", "", fit_min, fit_max);

    TF1 *f_exp_p_X = new TF1("f_exp_p_X", "[0]*exp(-[1]*x)", fit_min, fit_max);
    f_exp_p_X->SetParameters(100, 2.0);
    h_t_p_X->Fit(f_exp_p_X, "RQ ", "", fit_min, fit_max);

    TCanvas *c1 = new TCanvas("c1","t distributions",1200,500);
    c1->Divide(2,1);
    c1->cd(1);
    h_t_gamma_k1->Draw();
    f_exp_gamma_k1->SetLineColor(kRed);
    f_exp_gamma_k1->Draw("same");
    c1->cd(2);
    h_t_p_X->Draw();
    f_exp_p_X->SetLineColor(kRed);
    f_exp_p_X->Draw("same");
    c1->Update();


    TCanvas *c2 = new TCanvas("c2", "t'", 800, 600);
    h_tprime1->Draw();
    c2->Update();


    // TCanvas *c2 = new TCanvas("c2","t' gamma-K1",700,500);
    // h_t1prime->Draw();
    // f_exp_t1raw->SetLineColor(kRed);
    // f_exp_t1raw->Draw("same");
    // c2->Update();

    // Write LUND file
    string outname = "test_root_B1.lund";
    ofstream fout(outname);
    if (!fout.is_open()) {
        cerr << "ERROR: cannot open " << outname << " for writing." << endl;
    } else {
        int nEvents = (int)K1_lab_list.size();
        for (int i=0;i<nEvents;++i) {
            int num_particles = 4;
            fout << "\t" << num_particles << " 1 1 0 0 11 " << beam_energy << " 2212 " << PDG::proton << " 0\n";
            double vz_rand = rnd.Uniform(-10.0, 2.0);

            // 1) K2
            {
                int j = 1;
                int pid = 321; // K+
                TLorentzVector &v = K2_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mK << " " << 0.0 << " " << 0.0 << "\n";
            }
            // 2) Xi (pid 3312)
            {
                int j = 2;
                int pid = 3312;
                TLorentzVector &v = Xi_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mXi << " " << 0.0 << " " << 0.0 << "\n";
            }
            // 3) K1
            {
                int j = 3;
                int pid = 321;
                TLorentzVector &v = K1_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mK << " " << 0.0 << " " << 0.0 << "\n";
            }
            // 4) scattered electron
            {
                int j = 4;
                int pid = 11;
                TLorentzVector &v = v_scattered[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << PDG::electron << " " << 0.0 << " " << 0.0 << "\n";
            }
            fout << "\n";
        }
        fout.close();
        cout << "Written " << outname << endl;
    }

    // ROOT file output
    TFile *froot = new TFile("test_root_v1.root","RECREATE");
    h_t_gamma_k1->Write();
    h_t_p_X->Write();
    //h_t1prime->Write();
    f_exp1->Write("f_exp_t1prime");
    f_exp_t1raw->Write("f_exp_t1raw");
    f_exp_gamma_k1->Write("f_exp_gamma_k1");
    f_exp_p_X->Write("f_exp_p_X");
    froot->Close();
    cout << "Wrote ROOT histograms and fits to test_root_v1.root" << endl;
}
