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
//   xsec_weight:       <root file> [<hist>]   TH3D w(Q2, W, M_X)    default w_Q2_W_M
//   xsec_weight_mode:  interp | bin           (default interp)
//   pair_weight:       <root file> <hist> <pidA> <pidB>
//                                             TH3D w(Q2, W, M_AB), one per
//                                             species pair, repeatable
//   pair_weight_mode:  interp | bin           (default interp; must match
//                                             what build_pair_weight.py
//                                             was run with)
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
    // 3-D cross-section weight w(Q2, W, M_X) built by
    // build_xsec_weight3d.py. M_X is the invariant mass of the intermediate
    // X from the FIRST vertex (for `reaction: 2212, 9999: 9999, 2212, -2212`
    // that is M_ppbar). Also applied AFTER the decay chain, because M_X does
    // not exist until the intermediate mass has been sampled.
    std::string xsec_weight_file;
    std::string xsec_weight_name = "w_Q2_W_M";
    // How to read the 3-D weight: "interp" (default) trilinearly
    // interpolates between bin centers, giving a continuous w(Q2, W, M) --
    // the right choice for a cross section that is itself smooth, where a
    // per-bin step function would imprint the grid on the events. "bin"
    // looks up the bin the event falls in; it is the exact pairing for a
    // weight that is a per-bin ratio d/g against a BINNED target (the
    // accepted density is then proportional to d bin by bin), and is what
    // to use when the grid is coarse: interpolation blends neighbouring
    // bins into each event's accept probability, which on a 4x9x24 grid
    // moved the per-bin closure by ~25%.
    bool xsec_weight_interp = true;
    // Pair-mass weights w(Q2, W, M_AB), one TH3D per (pidA, pidB) species
    // pair, built jointly by build_pair_weight.py. Unlike xsec_weight, which
    // needs the TRUTH pairing (M_X), these are evaluated on EVERY (A, B)
    // pair in the final state and multiplied together: with two protons and
    // one antiproton, `2212 -2212` contributes two factors -- M(p1 pbar) and
    // M(p2 pbar) -- and `2212 2212` one. That is exactly how a POOLED data
    // histogram is filled, so the target can be the measured pooled M(p pbar)
    // and M(p p) distributions, both symmetric under p1 <-> p2, instead of a
    // "true" M_X that would have to be unfolded from them (and can go
    // negative). All pair surfaces go into ONE accept-reject.
    struct PairWeight {
        std::string file;
        std::string name;
        int pidA = 0;
        int pidB = 0;
    };
    std::vector<PairWeight> pair_weights;
    // How the pair surfaces are read. They are FITTED, with the lookup as
    // part of the model, so this must be the mode build_pair_weight.py was
    // run with (its --mode, default interp): a surface fitted per bin and
    // read interpolated -- or the reverse -- no longer reproduces its
    // target. interp gives a continuous w(Q2, W, M); bin imprints the grid
    // on the events as steps in Q2 and W.
    bool pair_weight_interp = true;

    // Consume one `key: value(s)` line of the input card (key already
    // stripped of its trailing colon). Returns true if the key belongs to
    // the weighting configuration, false so the caller can try its own
    // keys.
    bool parseKey(const std::string &key, std::istringstream &iss) {
        if (key == "weight_func") {
            readFileAndName(iss, weight_func_file, weight_func_name);
        } else if (key == "xsec_weight") {
            readFileAndName(iss, xsec_weight_file, xsec_weight_name);
        } else if (key == "xsec_weight_mode") {
            std::string val; iss >> val;
            xsec_weight_interp = (val != "bin");
            if (val != "interp" && val != "bin") {
                std::cerr << "WARNING: unrecognized xsec_weight_mode '" << val
                          << "'; using interp." << std::endl;
            }
        } else if (key == "pair_weight_mode") {
            std::string val; iss >> val;
            pair_weight_interp = (val != "bin");
            if (val != "interp" && val != "bin") {
                std::cerr << "WARNING: unrecognized pair_weight_mode '" << val
                          << "'; using interp." << std::endl;
            }
        } else if (key == "pair_weight") {
            // pair_weight: file.root hist_name pidA pidB  (all four required:
            // several surfaces can live in one file, so the name is not
            // defaulted)
            PairWeight pw;
            if (!(iss >> pw.file >> pw.name >> pw.pidA >> pw.pidB)) {
                std::cerr << "WARNING: pair_weight needs '<file> <hist> "
                          << "<pidA> <pidB>'; line ignored." << std::endl;
            } else {
                // A repeated line would apply the surface twice -- w^2 per
                // event, and a keep fraction that collapses quadratically.
                bool dup = false;
                for (const auto &q : pair_weights) {
                    if (q.pidA == pw.pidA && q.pidB == pw.pidB) dup = true;
                }
                if (dup) {
                    std::cerr << "WARNING: pair_weight for (" << pw.pidA << ", "
                              << pw.pidB << ") given more than once; keeping "
                              << "the first line only." << std::endl;
                } else {
                    pair_weights.push_back(pw);
                }
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
        if (!cfg_.xsec_weight_file.empty() &&
            xsec_.open(cfg_.xsec_weight_file, cfg_.xsec_weight_name, "xsec_weight")) {
            std::cout << "Cross-section weight enabled: "
                      << cfg_.xsec_weight_file << ":" << cfg_.xsec_weight_name
                      << "  (" << (cfg_.xsec_weight_interp ? "trilinear Interpolate"
                                                           : "per-bin lookup")
                      << " on Q2, W, M_X)" << std::endl;
        }
        // The pair surfaces are fitted JOINTLY (build_pair_weight.py rakes
        // them against each other on one sample), so applying a subset is
        // not "partially weighted", it is wrong. One failure drops them all.
        bool pairs_ok = true;
        for (const auto &pw : cfg_.pair_weights) {
            PairSurface ps;
            ps.pidA = pw.pidA;
            ps.pidB = pw.pidB;
            if (!ps.open(pw.file, pw.name, "pair_weight")) { pairs_ok = false; break; }
            std::cout << "Pair-mass weight enabled: " << pw.file << ":" << pw.name
                      << "  (" << (cfg_.pair_weight_interp ? "trilinear Interpolate"
                                                           : "per-bin lookup")
                      << " on Q2, W, M(" << pw.pidA << "," << pw.pidB
                      << "), product over all such pairs)" << std::endl;
            pairs_.push_back(std::move(ps));
        }
        if (!pairs_ok) {
            std::cerr << "ERROR: a pair_weight surface failed to load; the "
                      << "pair surfaces are fitted jointly, so ALL of them "
                      << "are disabled." << std::endl;
            for (auto &ps : pairs_) ps.close();
            pairs_.clear();
        }
        if (xsec_.hist && !pairs_.empty()) {
            std::cerr << "WARNING: xsec_weight and pair_weight are both active. "
                      << "Both reshape the (Q2, W) marginal, so the run will "
                      << "be corrected twice in Q2 and W." << std::endl;
        }

        if (saved) saved->cd(); else gROOT->cd();
    }

    void close() {
        q2ep_.close();
        xsec_.close();
        for (auto &ps : pairs_) ps.close();
        pairs_.clear();
    }

    bool hasElectronStage() const { return q2ep_.hist != nullptr; }
    bool hasEventStage()    const {
        return xsec_.hist != nullptr || !pairs_.empty();
    }

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
        if (xsec_.hist && !acceptXsec(kin, rnd))       { ++n_reject_xsec_; return false; }
        if (!pairs_.empty() && !acceptPairs(kin, rnd)) { ++n_reject_pair_; return false; }
        return true;
    }

    // -----------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------
    long long nRejectElectron() const { return n_reject_electron_; }
    long long nRejectXsec()     const { return n_reject_xsec_; }
    long long nRejectPair()     const { return n_reject_pair_; }
    // Post-decay rejections only: the electron stage retries inside the
    // sampler and never costs the driver an attempt.
    long long nRejectEvent()    const {
        return n_reject_xsec_ + n_reject_pair_;
    }

    void printSummary(std::ostream &os = std::cout) const {
        os << "    - cross-section weight: " << n_reject_xsec_ << std::endl;
        os << "    - pair-mass weight:     " << n_reject_pair_ << std::endl;
        if (n_pair_over_one_ > 0) {
            os << "  (pair-mass weight > 1 on " << n_pair_over_one_
               << " events -- outside the builder's sample; accepted outright)"
               << std::endl;
        }
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
        // defined. Strictly inside at the top: TH::Interpolate treats a
        // point AT the last center as outside the domain and returns 0,
        // so clamping onto it would silently reject the whole outer
        // half-bin.
        static double clampToCenters(const TAxis *ax, double x) {
            const double lo = ax->GetBinCenter(1);
            const double hi = std::nextafter(ax->GetBinCenter(ax->GetNbins()), lo);
            return std::min(std::max(x, lo), hi);
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

    // One pair_weight line: a 3-D surface plus the species pair it is
    // evaluated on. Unlike the other surfaces, a pair mass OUTSIDE the M
    // range is not a rejection: the builder stores a pass-through factor
    // in the M under/overflow bins for every (Q2, W) column (an entry the
    // surface knows nothing about must not veto the event, whose OTHER
    // pairing the data still counts). Q2 or W outside the grid is still
    // outside the analysis domain -> 0. Inside, either the bin's value or
    // trilinear interpolation between bin centers (clamped into the hull),
    // whichever the surface was fitted for.
    struct PairSurface : Surface3D {
        int pidA = 0;
        int pidB = 0;

        double lookup(double q2, double w, double m, bool interp) const {
            const TAxis *xa = hist->GetXaxis();
            const TAxis *ya = hist->GetYaxis();
            const TAxis *za = hist->GetZaxis();
            if (!inRange(xa, q2) || !inRange(ya, w)) return 0.0;
            if (!inRange(za, m) || !interp) {
                // FindBin returns 0 / nbins+1 outside the axis: the bins
                // the pass-through factor lives in.
                return hist->GetBinContent(xa->FindBin(q2), ya->FindBin(w),
                                           za->FindBin(m));
            }
            return hist->Interpolate(clampToCenters(xa, q2),
                                     clampToCenters(ya, w),
                                     clampToCenters(za, m));
        }
    };

    // Accept-reject on a weight normalized to max 1.
    static bool keep(double w, TRandom3 &rnd) {
        if (!std::isfinite(w) || w <= 0.0) return false;
        return rnd.Uniform() <= w;
    }

    // w(Q2, W, M_X). Outside the histogram range the cross section says
    // nothing, so the event is rejected: that is the domain the user
    // asked for.
    bool acceptXsec(const EventKinematics &kin, TRandom3 &rnd) const {
        return keep(xsec_.evaluate(kin.Q2, kin.W, kin.M_X,
                                   cfg_.xsec_weight_interp), rnd);
    }

    // Product over every pair_weight surface and, within each, over every
    // (pidA, pidB) pair the final state offers -- each unordered pair once
    // when A == B, each (A, B) combination once otherwise. The lookup mode
    // is whatever the surfaces were fitted with (pair_weight_mode); the
    // interpolation is part of the fitted model, not a smoothing applied
    // afterwards. Since every factor is <= 1,
    // the product is a valid accept probability as it stands. An event
    // with no such pair at all is rejected: the surface says nothing
    // about it.
    bool acceptPairs(const EventKinematics &kin, TRandom3 &rnd) const {
        if (!kin.final_particles) return false;
        const auto &fp = *kin.final_particles;
        double w = 1.0;
        for (const auto &ps : pairs_) {
            int n_pairs = 0;
            for (size_t i = 0; i < fp.size(); ++i) {
                if (fp[i].first != ps.pidA) continue;
                for (size_t j = (ps.pidA == ps.pidB ? i + 1 : 0); j < fp.size(); ++j) {
                    if (j == i || fp[j].first != ps.pidB) continue;
                    const double m = (fp[i].second + fp[j].second).M();
                    w *= ps.lookup(kin.Q2, kin.W, m, cfg_.pair_weight_interp);
                    if (!(w > 0.0)) return false;
                    ++n_pairs;
                }
            }
            if (n_pairs == 0) return false;
        }
        // The builder normalizes to the largest factor realized on ITS
        // sample; a product above 1 means this event sits where that
        // sample had nothing, and is accepted outright. Counted, so it
        // cannot pass unnoticed.
        if (w > 1.0) ++n_pair_over_one_;
        return keep(w, rnd);
    }

    WeightConfig cfg_;
    Surface2D q2ep_;   // weight_func
    Surface3D xsec_;   // xsec_weight
    std::vector<PairSurface> pairs_;   // pair_weight (one per line)

    long long n_reject_electron_ = 0;
    long long n_reject_xsec_     = 0;
    long long n_reject_pair_     = 0;
    mutable long long n_pair_over_one_ = 0;
};

#endif // EVENT_WEIGHTER_H
