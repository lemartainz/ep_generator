// ---------------------------------------------------------------------
// EventWeighter.h -- accept-reject weighting for runEventGenerator.cpp
//
// Every weight surface the generator can apply lives here, so the
// generator itself only has to ask two questions:
//
//   weighter.acceptElectron(Q2, Ep, rnd)   -- at electron-sampling time
//   weighter.acceptEvent(kin, rnd)         -- once the decay chain exists
//
// The surfaces themselves are built OUTSIDE the generator (see reweight/)
// and handed over as ROOT histograms through the input card:
//
//   weight_func:       <root file> [<hist>]   TH2D w(Q2, E')        default w_Q2_Ep
//   mom_weight:        <root file> [<hist>]   TH2D w(p_lead, p_sub) default w_pp
//   xsec_weight:       <root file> [<hist>]   TH3D w(Q2, W, M_X)    default w_Q2_W_M
//   xsec_weight_mode:  bin | interp           (default bin)
//
// Each is normalized so that max(w) = 1 and the event is kept with
// probability w. To add a new weight: give WeightConfig a file/name pair
// and a parseKey branch, load it in EventWeighter::load(), and evaluate it
// in acceptElectron() or acceptEvent() -- the generator does not change.
//
// Header-only so it can be #included straight into an ACLiC-compiled
// macro (root -l 'runEventGenerator.cpp+').
// ---------------------------------------------------------------------
#ifndef EVENT_WEIGHTER_H
#define EVENT_WEIGHTER_H

#include <TROOT.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TRandom3.h>
#include <TLorentzVector.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------
// Input-card side: which surfaces to load. Plain data, safe to copy, so
// it can sit inside the generator's ReadInput struct.
// ---------------------------------------------------------------------
struct WeightConfig {
    // Continuous weight function w(Q2, E') = data / gen (data density
    // divided by the generator's proposal density), built by
    // build_weight_func.py or build_xsec_weight.py. Evaluated with
    // TH2::Interpolate (bilinear -> continuous) on top of uniform Q2/E'
    // sampling.
    std::string weight_func_file;
    std::string weight_func_name = "w_Q2_Ep";
    // Continuous momentum weight w(p_lead, p_sub) = data / gen over the
    // leading / sub-leading proton (2212 only) momentum magnitudes, built
    // by build_weight_func.py --mode pmom. Applied AFTER the full event is
    // built, since the proton momenta only exist once the decay chain is
    // done.
    std::string mom_weight_file;
    std::string mom_weight_name = "w_pp";
    // 3-D cross-section weight w(Q2, W, M_X) built by
    // build_xsec_weight3d.py. M_X is the invariant mass of the intermediate
    // X from the FIRST vertex (for `reaction: 2212, 9999: 9999, 2212, -2212`
    // that is M_ppbar). Also applied AFTER the decay chain, because M_X does
    // not exist until the intermediate mass has been sampled.
    std::string xsec_weight_file;
    std::string xsec_weight_name = "w_Q2_W_M";
    // How to read the 3-D weight: "bin" (default) looks up the bin the
    // event falls in; "interp" trilinearly interpolates between bin
    // centers. "bin" is the correct pairing for a BINNED cross section:
    // the weight is a per-bin ratio d/g, so applying it per bin makes the
    // accepted density exactly proportional to d in every bin.
    // Interpolating blends neighbouring bins into each event's accept
    // probability, which on a coarse grid pulls the result away from the
    // cross section it was built from -- measurably so: on a 4x9x24 grid
    // it costs ~25% per bin. Use "interp" only when the underlying cross
    // section really is smooth and the binning is fine enough that the
    // two agree.
    bool xsec_weight_interp = false;

    // Consume one `key: value(s)` line of the input card (key already
    // stripped of its trailing colon). Returns true if the key belongs to
    // the weighting configuration, false so the caller can try its own
    // keys.
    bool parseKey(const std::string &key, std::istringstream &iss) {
        if (key == "weight_func") {
            readFileAndName(iss, weight_func_file, weight_func_name);
        } else if (key == "mom_weight") {
            readFileAndName(iss, mom_weight_file, mom_weight_name);
        } else if (key == "xsec_weight") {
            readFileAndName(iss, xsec_weight_file, xsec_weight_name);
        } else if (key == "xsec_weight_mode") {
            std::string val; iss >> val;
            xsec_weight_interp = (val == "interp");
            if (val != "interp" && val != "bin") {
                std::cerr << "WARNING: unrecognized xsec_weight_mode '" << val
                          << "'; using bin." << std::endl;
            }
        } else {
            return false;
        }
        return true;
    }

private:
    // `<key>: path/to/file.root  [hist_name]` -- the name keeps its
    // default when omitted.
    static void readFileAndName(std::istringstream &iss,
                                std::string &file, std::string &name) {
        std::string fname, hname;
        iss >> fname;
        if (iss >> hname) name = hname;
        file = fname;
    }
};

// ---------------------------------------------------------------------
// What a fully built event looks like to the weighter. The generator
// fills this once per event after the decay chain; the weighter reads
// whichever members its surfaces need.
// ---------------------------------------------------------------------
struct EventKinematics {
    double Q2  = 0.0;   // -q^2 of the virtual photon [GeV^2]
    double Ep  = 0.0;   // scattered-electron energy [GeV]
    double W   = 0.0;   // hadronic invariant mass [GeV]
    double M_X = 0.0;   // invariant mass of the intermediate X from the
                        // first vertex, from the TRUTH 4-vector -- the
                        // final state holds two protons and picking the
                        // one that came from X is ambiguous downstream,
                        // while here it is exact by construction.
    // Final-state (pdg, 4-vector) list, for surfaces that reshape
    // particle-level variables such as the proton momenta.
    const std::vector<std::pair<int, TLorentzVector>> *final_particles = nullptr;
};

// ---------------------------------------------------------------------
// The weighter. Owns the ROOT files behind the surfaces, so it is
// non-copyable; construct it once in the driver and pass it around by
// pointer/reference.
// ---------------------------------------------------------------------
class EventWeighter {
public:
    explicit EventWeighter(const WeightConfig &cfg) : cfg_(cfg) {}
    ~EventWeighter() { close(); }
    EventWeighter(const EventWeighter &) = delete;
    EventWeighter &operator=(const EventWeighter &) = delete;

    // Open every surface named in the config. A surface that fails to
    // load is reported and skipped (the run continues unweighted in that
    // variable), matching the generator's previous behaviour.
    void load() {
        // TFile::Open makes the new file the current directory; anything
        // the caller books afterwards would then be owned by it and die
        // with it. Restore gDirectory so loading is side-effect free.
        TDirectory *saved = gDirectory;

        if (!cfg_.weight_func_file.empty() &&
            q2ep_.open(cfg_.weight_func_file, cfg_.weight_func_name, "weight_func")) {
            std::cout << "Continuous weight function enabled: "
                      << cfg_.weight_func_file << ":" << cfg_.weight_func_name
                      << "  (bilinear Interpolate on Q2, E')" << std::endl;
        }
        if (!cfg_.mom_weight_file.empty() &&
            pp_.open(cfg_.mom_weight_file, cfg_.mom_weight_name, "mom_weight")) {
            std::cout << "Momentum weight function enabled: "
                      << cfg_.mom_weight_file << ":" << cfg_.mom_weight_name
                      << "  (bilinear Interpolate on p_lead, p_sub)" << std::endl;
        }
        if (!cfg_.xsec_weight_file.empty() &&
            xsec_.open(cfg_.xsec_weight_file, cfg_.xsec_weight_name, "xsec_weight")) {
            std::cout << "Cross-section weight enabled: "
                      << cfg_.xsec_weight_file << ":" << cfg_.xsec_weight_name
                      << "  (" << (cfg_.xsec_weight_interp ? "trilinear Interpolate"
                                                           : "per-bin lookup")
                      << " on Q2, W, M_X)" << std::endl;
        }

        if (saved) saved->cd(); else gROOT->cd();
    }

    void close() {
        q2ep_.close();
        pp_.close();
        xsec_.close();
    }

    bool hasElectronStage() const { return q2ep_.hist != nullptr; }
    bool hasEventStage()    const { return pp_.hist != nullptr || xsec_.hist != nullptr; }

    // -----------------------------------------------------------------
    // Stage 1: at electron-sampling time, before anything is decayed.
    // Keep the proposal (Q2, E') with probability w(Q2, E').
    // -----------------------------------------------------------------
    bool acceptElectron(double Q2, double Ep, TRandom3 &rnd) {
        if (!q2ep_.hist) return true;
        if (!keep(q2ep_.interpolate(Q2, Ep), rnd)) { ++n_reject_electron_; return false; }
        return true;
    }

    // -----------------------------------------------------------------
    // Stage 2: once the whole event exists. Each surface is an
    // independent accept-reject; an event has to survive all of them.
    // -----------------------------------------------------------------
    bool acceptEvent(const EventKinematics &kin, TRandom3 &rnd) {
        if (pp_.hist   && !acceptMomentum(kin, rnd)) { ++n_reject_mom_;  return false; }
        if (xsec_.hist && !acceptXsec(kin, rnd))     { ++n_reject_xsec_; return false; }
        return true;
    }

    // -----------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------
    long long nRejectElectron() const { return n_reject_electron_; }
    long long nRejectMom()      const { return n_reject_mom_; }
    long long nRejectXsec()     const { return n_reject_xsec_; }
    // Post-decay rejections only: the electron stage retries inside the
    // sampler and never costs the driver an attempt.
    long long nRejectEvent()    const { return n_reject_mom_ + n_reject_xsec_; }

    void printSummary(std::ostream &os = std::cout) const {
        os << "    - momentum weight:      " << n_reject_mom_  << std::endl;
        os << "    - cross-section weight: " << n_reject_xsec_ << std::endl;
        if (q2ep_.hist) {
            os << "  (Q2, E') weight rejected " << n_reject_electron_
               << " electron proposals before decay" << std::endl;
        }
    }

private:
    // -----------------------------------------------------------------
    // One histogram in one ROOT file, with the range checks that every
    // surface needs. TH::Interpolate returns 0 outside the hull of the
    // bin centers, so a point outside the axis RANGE is rejected outright
    // (the surface says nothing there) and the caller decides how to
    // treat the half-bin band between range and hull.
    // -----------------------------------------------------------------
    template <class H>
    struct Surface {
        TFile *file = nullptr;
        H     *hist = nullptr;

        bool open(const std::string &fname, const std::string &hname,
                  const char *label) {
            file = TFile::Open(fname.c_str(), "READ");
            if (!file || file->IsZombie()) {
                std::cerr << "ERROR: cannot open " << label << " file "
                          << fname << std::endl;
                close();
                return false;
            }
            hist = dynamic_cast<H *>(file->Get(hname.c_str()));
            if (!hist) {
                std::cerr << "ERROR: " << H::Class_Name() << " '" << hname
                          << "' not found in " << fname << std::endl;
                close();
                return false;
            }
            return true;
        }

        void close() {
            hist = nullptr;
            if (file) { file->Close(); delete file; file = nullptr; }
        }

        static bool inRange(const TAxis *ax, double x) {
            return x > ax->GetXmin() && x < ax->GetXmax();
        }
        // Clamp into the hull of the bin centers, where Interpolate is
        // defined.
        static double clampToCenters(const TAxis *ax, double x) {
            return std::min(std::max(x, ax->GetBinCenter(1)),
                            ax->GetBinCenter(ax->GetNbins()));
        }
    };

    // 2-D: bilinear interpolation between bin centers -> a smooth w(x, y).
    struct Surface2D : Surface<TH2D> {
        double interpolate(double x, double y) const {
            // Outside the interior Interpolate returns 0 -> reject.
            if (!inRange(hist->GetXaxis(), x) ||
                !inRange(hist->GetYaxis(), y)) return 0.0;
            return hist->Interpolate(x, y);
        }
    };

    // 3-D: per-bin lookup (see WeightConfig::xsec_weight_interp) or
    // trilinear interpolation.
    struct Surface3D : Surface<TH3D> {
        double evaluate(double x, double y, double z, bool interp) const {
            const TAxis *xa = hist->GetXaxis();
            const TAxis *ya = hist->GetYaxis();
            const TAxis *za = hist->GetZaxis();
            if (!inRange(xa, x) || !inRange(ya, y) || !inRange(za, z)) return 0.0;

            if (!interp) {
                // Piecewise-constant lookup: the event takes the weight of
                // the bin it lands in -- exact closure for a per-bin ratio
                // d/g, by construction.
                return hist->GetBinContent(xa->FindBin(x), ya->FindBin(y),
                                           za->FindBin(z));
            }
            // Inside the range but within half a bin of an edge, TH3::
            // Interpolate returns 0 -- it only interpolates inside the hull
            // of the BIN CENTERS. On a coarse axis that is a large slice of
            // the range (with 4 Q2 bins over [1,7] it silently discards
            // Q2 < 1.5 and Q2 > 5.75), so clamp into the hull.
            return hist->Interpolate(clampToCenters(xa, x),
                                     clampToCenters(ya, y),
                                     clampToCenters(za, z));
        }
    };

    // Accept-reject on a weight normalized to max 1.
    static bool keep(double w, TRandom3 &rnd) {
        if (!std::isfinite(w) || w <= 0.0) return false;
        return rnd.Uniform() <= w;
    }

    // w(p_lead, p_sub) over the two largest proton (pid == 2212,
    // antiproton excluded) momentum magnitudes. Events without two
    // protons are rejected.
    bool acceptMomentum(const EventKinematics &kin, TRandom3 &rnd) const {
        if (!kin.final_particles) return false;
        double p_lead = -1.0, p_sub = -1.0;
        for (const auto &pr : *kin.final_particles) {
            if (pr.first != 2212) continue;
            double p = pr.second.Vect().Mag();
            if (p > p_lead)      { p_sub = p_lead; p_lead = p; }
            else if (p > p_sub)  { p_sub = p; }
        }
        if (p_sub < 0.0) return false;
        return keep(pp_.interpolate(p_lead, p_sub), rnd);
    }

    // w(Q2, W, M_X). Outside the histogram range the cross section says
    // nothing, so the event is rejected: that is the domain the user
    // asked for.
    bool acceptXsec(const EventKinematics &kin, TRandom3 &rnd) const {
        return keep(xsec_.evaluate(kin.Q2, kin.W, kin.M_X,
                                   cfg_.xsec_weight_interp), rnd);
    }

    WeightConfig cfg_;
    Surface2D q2ep_;   // weight_func
    Surface2D pp_;     // mom_weight
    Surface3D xsec_;   // xsec_weight

    long long n_reject_electron_ = 0;
    long long n_reject_mom_      = 0;
    long long n_reject_xsec_     = 0;
};

#endif // EVENT_WEIGHTER_H
