
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
#include <iomanip>   // setprecision, fixed
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

// helper: two-body momentum magnitude in parent rest (GeV)
double twoBodyMomentum(double M, double m1, double m2) {
    if (M <= m1 + m2) return NAN;
    double term = (M*M - (m1 + m2)*(m1 + m2)) * (M*M - (m1 - m2)*(m1 - m2));
    if (term < 0) return NAN;
    return 0.5 / M * sqrt(term);
}

// Compute Lorentz invariant (p1 - p2)^2 (GeV^2)
double invariantSquare(const TLorentzVector &a, const TLorentzVector &b) {
    TLorentzVector d = a - b;
    return d.M2();
}



TH1D *h_tprime = new TH1D("h_tprime", "t' distribution; t' (GeV^{2}); Events", 100, -1, 2.0); 
// The last parameter (2.0) is just an example max range; adjust depending on your kinematics.

// pair<TLorentzVector,TLorentzVector> twoBodyDecayWeighted(const TLorentzVector &parent_lab,
//                                                          double m1, double m2,
//                                                          double B,
//                                                          const TLorentzVector &p_virtual_lab,
//                                                          TRandom3 &rnd,
//                                                          int nCandidates = 10000) {
//     double M = parent_lab.M();
//     if (M <= m1 + m2) {
//         TLorentzVector nanv(0,0,0,0);
//         return make_pair(nanv, nanv);
//     }

//     double p_mag = twoBodyMomentum(M, m1, m2);
//     if (!isfinite(p_mag)) {
//         TLorentzVector nanv(0,0,0,0);
//         return make_pair(nanv, nanv);
//     }

//     // Calculate t at cosTheta = ±1
//     TLorentzVector d1_rest_plus = [&]() {
//         double cosTheta = +1.0;
//         double sinTheta = sqrt(max(0.0, 1.0 - cosTheta*cosTheta));
//         TLorentzVector v;
//         v.SetPxPyPzE(p_mag * sinTheta, 0.0, p_mag * cosTheta, sqrt(p_mag*p_mag + m1*m1));
//         return v;
//     }();
//     TLorentzVector d1_rest_minus = [&]() {
//         double cosTheta = -1.0;
//         double sinTheta = sqrt(max(0.0, 1.0 - cosTheta*cosTheta));
//         TLorentzVector v;
//         v.SetPxPyPzE(p_mag * sinTheta, 0.0, p_mag * cosTheta, sqrt(p_mag*p_mag + m1*m1));
//         return v;
//     }();

//     TVector3 boostToLab = parent_lab.BoostVector();
//     d1_rest_plus.Boost(boostToLab);
//     d1_rest_minus.Boost(boostToLab);

//     double t_plus  = invariantSquare(p_virtual_lab, d1_rest_plus);
//     double t_minus = invariantSquare(p_virtual_lab, d1_rest_minus);

//     double t_min = std::min(t_plus, t_minus);
//     double t_max = std::max(t_plus, t_minus);

//     // Skip unphysical events
//     if (!isfinite(t_min) || !isfinite(t_max) || t_min >= t_max) {
//         TLorentzVector nanv(0,0,0,0);
//         return make_pair(nanv, nanv);
//     }

//     double cosT = 0.0;

//     if (B > 0) {
//         vector<double> t_candidates(nCandidates);
//         vector<double> weights(nCandidates);
//         double weight_sum = 0.0;

//         for (int i=0; i<nCandidates; ++i) {
//             t_candidates[i] = rnd.Uniform(t_min, t_max);
//             // double t_prime = t_candidates[i] - t_min; // <-- t'
//             // h_tprime->Fill(t_prime); // Fill histogram for each sampled candidate
//             weights[i] = exp(-B * fabs(t_candidates[i] - t_min));
//             weight_sum += weights[i];
//         }
//         for (int i=0; i<nCandidates; ++i) {
//             weights[i] /= weight_sum;
//         }

//         // Weighted selection
//         double r = rnd.Uniform(0.0, 1.0);
//         double cumsum = 0.0;
//         int pick_idx = 0;
//         for (; pick_idx < nCandidates; ++pick_idx) {
//             cumsum += weights[pick_idx];
//             if (r <= cumsum) break;
//         }
//         if (pick_idx >= nCandidates) pick_idx = nCandidates - 1;

//         double t_sampled = t_candidates[pick_idx];
//         double t_prime_sampled = t_sampled - t_min;
//         h_tprime->Fill(t_prime_sampled); 
//         cosT = 1.0 - 2.0 * (t_sampled - t_min) / (t_max - t_min);
//         cosT = std::clamp(cosT, -1.0, 1.0);

//     } else {
//         // Uniform cos(theta)
//         cosT = rnd.Uniform(-1.0, 1.0);
//     }

//     // Final decay kinematics
//     double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());
//     double sinT = sqrt(max(0.0, 1.0 - cosT*cosT));

//     TLorentzVector d1_rest;
//     d1_rest.SetPxPyPzE(p_mag * sinT * cos(phi),
//                        p_mag * sinT * sin(phi),
//                        p_mag * cosT,
//                        sqrt(p_mag*p_mag + m1*m1));

//     TLorentzVector d2_rest(-d1_rest.Vect(), sqrt(p_mag*p_mag + m2*m2));

//     d1_rest.Boost(boostToLab);
//     d2_rest.Boost(boostToLab);

//     return make_pair(d1_rest, d2_rest);
// }

pair<TLorentzVector,TLorentzVector> twoBodyDecayWeighted(
    const TLorentzVector &parent_lab,
    double m1, double m2,
    double B,
    const TLorentzVector &p_virtual_lab,
    TRandom3 &rnd,
    int nCandidates = 10000) 
{
    double M = parent_lab.M();
    if (M <= m1 + m2) {
        TLorentzVector nanv(0,0,0,0);
        return make_pair(nanv, nanv);
    }

    double p_mag = twoBodyMomentum(M, m1, m2);
    if (!std::isfinite(p_mag)) {
        TLorentzVector nanv(0,0,0,0);
        return make_pair(nanv, nanv);
    }

    // ==== PDG t_min / t_max calculation in parent rest frame ====
    TVector3 boost_to_parent = -parent_lab.BoostVector(); // LAB -> parent rest
    TLorentzVector q_star = p_virtual_lab;
    q_star.Boost(boost_to_parent);

    const double pK = p_mag;                              // |p_K^*|
    const double EK = std::sqrt(m1*m1 + pK*pK);           //  E_K^*
    const double pq = q_star.P();                         // |q^*|
    const double Eq = q_star.E();                         //  E_q^*
    const double mq2 = q_star.M2();                       // q^2

    // t(cosθ*) = mq2 + m1^2 - 2( Eq*EK - pq*pK*cosθ* )
    const double t_min = mq2 + m1*m1 - 2.0*(Eq*EK + pq*pK); // cosθ* = +1
    const double t_max = t_min - 4.0*pq*pK;                 // cosθ* = -1

    if (!std::isfinite(t_min) || !std::isfinite(t_max) || !(t_max < t_min) || pq <= 0.0) {
        TLorentzVector nanv(0,0,0,0);
        return make_pair(nanv, nanv);
    }

    // ==== Sample t with exponential weighting or uniform ====
    double t_sampled = 0.0;
    if (B > 0) {
        // Importance sampling: exponential slope in (t_max, t_min)
        const double Xmax = (t_min - t_max);
        double X;
        do {
            double u = std::max(1e-16, rnd.Uniform());
            X = -std::log(u)/B;
        } while (X > Xmax);
        t_sampled = t_min - X;
        h_tprime->Fill(-(t_sampled - t_min)); //here t_max < t_min and X is always positve so t_sampled tmin-X is always less than t_min
    } else {
        // Uniform in cosθ* means uniform t since t ~ linear(cosθ*)
        t_sampled = rnd.Uniform(t_max, t_min);
    }

    // ==== Exact PDG mapping t -> cosθ* ====
    double cosT = 1.0 - (t_min - t_sampled)/(2.0*pq*pK);
    cosT = std::clamp(cosT, -1.0, 1.0);
    const double phi = rnd.Uniform(-TMath::Pi(), TMath::Pi());
    const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT*cosT));

    // ==== Build daughter momenta in parent rest frame ====
    TVector3 ez = q_star.Vect();
    if (ez.Mag2() > 0) ez = ez.Unit(); else ez = TVector3(0,0,1);
    TVector3 tmp = (std::fabs(ez.Z()) < 0.9) ? TVector3(0,0,1) : TVector3(1,0,0);
    TVector3 ex = (tmp.Cross(ez)).Unit();
    TVector3 ey = (ez.Cross(ex)).Unit();

    TVector3 pK_dir = (sinT*std::cos(phi))*ex + (sinT*std::sin(phi))*ey + (cosT)*ez;
    TLorentzVector d1_rest(pK*pK_dir.X(), pK*pK_dir.Y(), pK*pK_dir.Z(), EK);

    TLorentzVector d2_rest(-d1_rest.Vect(), std::sqrt(p_mag*p_mag + m2*m2));

    // ==== Boost daughters back to LAB frame ====
    TVector3 boostToLab = parent_lab.BoostVector();
    d1_rest.Boost(boostToLab);
    d2_rest.Boost(boostToLab);

    return make_pair(d1_rest, d2_rest);
}


// Main macro function
void eventGenerator() {
    const double beam_energy = 6.5; // GeV
    const double target_mass = PDG::proton;
    const int num_events_target = 100000;
    const double W_cut = 3.1;
    const double B_k1 = 4.0;
    const double B_X  = 4.0;
    const double mK = PDG::Kp;
    const double mXi = PDG::Xi;
    TRandom3 rnd(0);

    cout << "mc_generator: Starting generation of " << num_events_target << " events (attempting)..." << endl;

    TLorentzVector p_beam(0,0,beam_energy, beam_energy);
    TLorentzVector p_target(0,0,0, target_mass);

    vector<TLorentzVector> v_scattered; v_scattered.reserve(num_events_target);
    vector<TLorentzVector> v_virtual;   v_virtual.reserve(num_events_target);
    vector<TLorentzVector> v_W;         v_W.reserve(num_events_target);

    int batch = max(4 * num_events_target, 10000);
    while ((int)v_scattered.size() < num_events_target) {
        for (int i=0;i<batch && (int)v_scattered.size() < num_events_target; ++i) {
            double Q2 = rnd.Uniform(0.2, 0.2); // fixed Q2
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

    TH1D *h_t_gamma_k1 = new TH1D("h_t_gamma_k1", "t: virtual gamma - K1; t [GeV^{2}]; Counts", 100, 0, 3.0);
    TH1D *h_t_p_X       = new TH1D("h_t_p_X",       "t: proton - X; t [GeV^{2}]; Counts", 100, 0, 3.0);

    for (int ie=0; ie < (int)v_scattered.size(); ++ie) {
        const TLorentzVector &p_sc = v_scattered[ie];
        const TLorentzVector &p_v  = v_virtual[ie];
        const TLorentzVector &pW   = v_W[ie];

        double mX = 2.5;
        if (mX <= (mK + mXi)) continue;

        auto k1_x_pair = twoBodyDecayWeighted(pW, mK, mX, B_k1, p_v, rnd, 2000);
        TLorentzVector p_k1_lab = k1_x_pair.first;
        TLorentzVector p_X_lab  = k1_x_pair.second;

        if (p_k1_lab.E() == 0 && p_X_lab.E() == 0) continue;

        auto k2_xi_pair = twoBodyDecayWeighted(p_X_lab, mK, mXi, B_X, p_v, rnd, 2000);
        TLorentzVector p_k2_lab = k2_xi_pair.first;
        TLorentzVector p_xi_lab = k2_xi_pair.second;

        if (p_k2_lab.E() == 0 && p_xi_lab.E() == 0) continue;

        K1_lab_list.push_back(p_k1_lab);
        X_lab_list.push_back(p_X_lab);
        K2_lab_list.push_back(p_k2_lab);
        Xi_lab_list.push_back(p_xi_lab);

        double t1 = invariantSquare(p_v, p_k1_lab);
        double t2 = invariantSquare(p_target, p_X_lab);

        h_t_gamma_k1->Fill(-t1);
        h_t_p_X->Fill(-t2);
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

    double fit_min = 0.5;
    double fit_max = 3.0;
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
    h_t_p_X->Fit(f_exp_p_X, "RQ", "", fit_min, fit_max);

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
    h_tprime->Draw();
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
