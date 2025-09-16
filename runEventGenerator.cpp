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

using namespace std;

// PDG masses
struct PDG {
    static constexpr double proton   = 0.9382720813;
    static constexpr double pion     = 0.13957039;
    static constexpr double electron = 0.0005109989461;
    static constexpr double Kp       = 0.493677;
    static constexpr double Xi       = 1.322;
};

// Two-body decay result
struct DecayResult {
    TLorentzVector d1_lab;
    TLorentzVector d2_lab;
    double t_min;
    double t_max;
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

// EventGenerator class
class eventGenerator_test {
public:
    struct Range {
        double min;
        double max;
        bool is_range; // true = sample uniformly, false = single value
        double sample(TRandom3 &rnd) const {
            return is_range ? rnd.Uniform(min, max) : min;
        }
    };

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

    eventGenerator_test(int n, double Ebeam, double Mtarget)
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

            if (theta < theta_range.min || theta > theta_range.max) {
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

void runEventGenerator() {
    const int num_events = 10000;
    const double beam_energy = 10.2;
    const double B = 2.0;

    eventGenerator_test gen(num_events, beam_energy, PDG::proton);

    eventGenerator_test::Range Q2_range {0.0, 6.0, true};   // range
    eventGenerator_test::Range E_range  {0.5, 6.0, true};   // range
    eventGenerator_test::Range theta_range {5.0*TMath::DegToRad(), 35.0*TMath::DegToRad(), true}; // range in radians
    double W_min = 4.0;

    TH1D *h_e_theta    = new TH1D("h_e_theta", "Scattered electron #theta; #theta [rad]; Counts", 100, 0.0, 180);

    TLorentzVector p_target = TLorentzVector(0, 0, 0, PDG::proton);
    TLorentzVector p_beam = TLorentzVector(0, 0, beam_energy, beam_energy);
    auto electronEvents = gen.generateScatteredElectron(Q2_range, E_range, theta_range, W_min);

    cout << "Generated " << electronEvents.v_scattered.size() << " events passing W_min" << endl;

    for (const auto& event : electronEvents.v_scattered) {
        double theta = event.Theta();
        h_e_theta->Fill(theta*TMath::RadToDeg());
    }

    // Example of a decay
    vector<TLorentzVector> p1_lab_list; p1_lab_list.reserve(num_events);
    vector<TLorentzVector> X_lab_list;  X_lab_list.reserve(num_events);
    vector<TLorentzVector> p2_lab_list; p2_lab_list.reserve(num_events);
    vector<TLorentzVector> pbar_lab_list; pbar_lab_list.reserve(num_events);

    TH1D *h_t_gamma_k1 = new TH1D("h_t_gamma_k1", "t: virtual gamma - K1; t [GeV^{2}]; Counts", 100, 0, 6.0);
    TH1D *h_t_p_X      = new TH1D("h_t_p_X",       "t: proton - X; t [GeV^{2}]; Counts", 100, 0, 6.0);

    cout << "Processing decays..." << endl;
    double mp = PDG::proton, mpbar = PDG::proton;
    for(size_t i=0;i<electronEvents.v_W.size();i++){
        double mX = 2.5;
        if (mX <= (mp + mpbar)) continue;

        auto decay1 = gen.twoBodyDecayWeighted(electronEvents.v_W[i], mp, mX, B, electronEvents.v_virtual[i]);
        TLorentzVector p_p1_lab = decay1.d1_lab;
        TLorentzVector p_X_lab  = decay1.d2_lab;

        if (p_X_lab.E() == 0 && p_p1_lab.P() == 0) continue;
        double t_min = decay1.t_min;
        double t_max = decay1.t_max;


        auto decay2 = gen.twoBodyDecayWeighted(p_X_lab, mp, mpbar, 0, electronEvents.v_virtual[i]);
        TLorentzVector p_p2_lab = decay2.d1_lab;
        TLorentzVector p_pbar_lab = decay2.d2_lab;

        if (p_p2_lab.E() == 0 && p_pbar_lab.E() == 0) continue;

        p1_lab_list.push_back(p_p1_lab);
        X_lab_list.push_back(p_X_lab);
        p2_lab_list.push_back(p_p2_lab);
        pbar_lab_list.push_back(p_pbar_lab);

        double t1 = invariantSquare(electronEvents.v_virtual[i], p_p1_lab, false);
        double t2 = invariantSquare(p_target, p_X_lab, false);

        h_t_gamma_k1->Fill(-(t1-t_max));
        h_t_p_X->Fill(-(t2-t_max));

    }
    cout << "Finished processing decays." << endl;

    cout << "Creating LUND file..." << endl;
    // Write to LUND file
    ofstream fout("events.lund");
    if (!fout.is_open()) {
        cerr << "ERROR: cannot open events.lund for writing." << endl;
    } else {
        int nEvents = (int)p1_lab_list.size();
        for (int i=0;i<nEvents;++i) {
            int num_particles = 4;
            fout << "\t" << num_particles << " 1 1 0 0 11 " << beam_energy << " 2212 " << PDG::proton << " 0\n";
            double vz_rand = gen.rnd.Uniform(-5.0, 0.0);
            // 1) p1
            {
                int j = 1;
                int pid = 2212; // proton
                TLorentzVector &v = p1_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mp << " " << vz_rand << " " << 0.0 << "\n";
            }
            // 2) pbar
            {
                int j = 2;
                int pid = -2212; // antiproton
                TLorentzVector &v = pbar_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mpbar << " " << vz_rand << " " << 0.0 << "\n";
            }
            // 3) p2
            {
                int j = 3;
                int pid = 2212; // proton
                TLorentzVector &v = p2_lab_list[i];
                fout << j << " " << 0 << " " << 1 << " "    << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << mp << " " << vz_rand << " " << 0.0 << "\n";
            }
            // 4) scattered electron
            {
                int j = 4;
                int pid = 11;
                TLorentzVector &v = electronEvents.v_scattered[i];
                fout << j << " " << 0 << " " << 1 << " " << pid << " 0 0 ";
                fout << std::fixed << std::setprecision(6)
                     << v.Px() << " " << v.Py() << " " << v.Pz() << " " << v.E() << " "
                     << PDG::electron << " " << vz_rand << " " << 0.0 << "\n";
            }
            fout << "\n";
        }
        fout.close();
        cout << "Written events.lund" << endl;
    }

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

    TCanvas *c2 = new TCanvas("c2","t distributions log",1200,500);
    h_e_theta->Draw();
    c2->Update();
}

