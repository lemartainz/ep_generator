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
// probability w.
//
// A third kind of weight is CARRIED rather than accept-rejected: every
// event is kept and the weight travels with it (truth-ntuple branch
// w_ratio, and a sidecar file next to the LUND). It is not normalized.
//
//   weighter.eventWeight(kin)              -- once the decay chain exists
//
//   ratio_weight:          <root file> [<hist>]  TH2D dsigma/dt(s, t)  default dsdt_s_t
//   ratio_weight_mode:     linear | log          table holds dsigma/dt or ln(dsigma/dt)
//   ratio_weight_formula:  <expression in s, t>  TFormula instead of a table
//
//   w_ratio = dsigma/dt(s_pbarp, t) / dsigma/dt(s_pp, t)
//
// with ONE parametrization evaluated at two sub-energies, for the pbar-p
// vs p-p rescattering comparison in e p -> e' p p pbar (see ratioVars()).
//
// To add a new weight: give WeightConfig a file/name pair and a parseKey
// branch, load it in EventWeighter::load(), and evaluate it in
// acceptElectron(), acceptEvent() or eventWeight() -- the generator does
// not change.
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
#include <TFormula.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
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
    // Carried ratio weight w = dsigma/dt(s_pbarp, t) / dsigma/dt(s_pp, t).
    // The parametrization dsigma/dt(s, t) comes either as a TH2D table
    // (x = s, y = t [GeV^2]) built by reweight/build_dsdt_table.py, looked
    // up with bilinear interpolation between bin centers, or as a TFormula
    // string in the variables s and t (x, y also accepted). Any overall
    // constant cancels in the ratio, so nothing is normalized.
    std::string ratio_weight_file;
    std::string ratio_weight_name = "dsdt_s_t";
    // Table stored as ln(dsigma/dt) (built with --log): interpolate in log
    // space and exponentiate. Much more accurate for a steep exponential in
    // t than interpolating dsigma/dt itself.
    bool ratio_weight_log = false;
    std::string ratio_formula;

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
        } else if (key == "ratio_weight") {
            readFileAndName(iss, ratio_weight_file, ratio_weight_name);
        } else if (key == "ratio_weight_mode") {
            std::string val; iss >> val;
            ratio_weight_log = (val == "log");
            if (val != "log" && val != "linear") {
                std::cerr << "WARNING: unrecognized ratio_weight_mode '" << val
                          << "'; using linear." << std::endl;
            }
        } else if (key == "ratio_weight_formula") {
            // The expression contains spaces: take the rest of the line
            // (as readInputFile does for `reaction`), minus a trailing
            // `# comment` and surrounding whitespace.
            std::string rest;
            std::getline(iss, rest);
            size_t hash = rest.find('#');
            if (hash != std::string::npos) rest.erase(hash);
            size_t a = rest.find_first_not_of(" \t");
            size_t b = rest.find_last_not_of(" \t\r");
            ratio_formula = (a == std::string::npos) ? "" : rest.substr(a, b - a + 1);
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
    // Truth 4-vectors of the first vertex gamma* + target -> recoil + X,
    // for weights built from sub-system invariants (see ratioVars()).
    TLorentzVector q;         // virtual photon
    TLorentzVector p_target;  // target, at rest
    TLorentzVector p_recoil;  // first-vertex proton
    TLorentzVector p_X;       // first-vertex intermediate X
    bool have_vertex = false; // the generator filled the four above
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

        if (!cfg_.ratio_weight_file.empty() && !cfg_.ratio_formula.empty()) {
            std::cerr << "ERROR: give ratio_weight OR ratio_weight_formula, not "
                         "both; ratio weight disabled." << std::endl;
        } else if (!cfg_.ratio_weight_file.empty()) {
            if (ratio_.open(cfg_.ratio_weight_file, cfg_.ratio_weight_name, "ratio_weight")) {
                std::cout << "Ratio weight enabled (carried as w_ratio): "
                          << cfg_.ratio_weight_file << ":" << cfg_.ratio_weight_name
                          << "  (bilinear Interpolate on s, t; table holds "
                          << (cfg_.ratio_weight_log ? "ln dsigma/dt" : "dsigma/dt")
                          << ")" << std::endl;
            }
        } else if (!cfg_.ratio_formula.empty()) {
            loadFormula(cfg_.ratio_formula);
        }

        if (saved) saved->cd(); else gROOT->cd();
    }

    void close() {
        q2ep_.close();
        pp_.close();
        xsec_.close();
        ratio_.close();
        formula_.reset();
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
    // Carried weight: w_ratio = dsigma/dt(s_pbarp, t) / dsigma/dt(s_pp, t).
    // Not an accept-reject -- the caller keeps every event and records
    // the number. Returns 1.0 when the stage is off, 0.0 when the event
    // cannot be weighted (wrong topology, outside the table, invalid
    // dsigma/dt); each of those is counted for printSummary().
    // -----------------------------------------------------------------
    struct RatioVars {
        double s_pbarp = NAN;   // (p_pbar + p_recoil)^2
        double s_pp    = NAN;   // (p_fromX + p_recoil)^2
        double t       = NAN;   // (p_target - p_recoil)^2 == (q - p_X)^2
    };

    // The sub-system invariants for e p -> e' p_recoil X, X -> p pbar.
    // p_fromX = p_X - p_pbar is exact by four-momentum conservation, so
    // the two 2212s in the final state never have to be told apart.
    // Needs the truth vertex and exactly one antiproton; otherwise the
    // members are left NaN and false is returned.
    static bool ratioVars(const EventKinematics &kin, RatioVars &rv) {
        rv = RatioVars{};
        if (!kin.have_vertex || !kin.final_particles) return false;
        const TLorentzVector *p_pbar = nullptr;
        int n_pbar = 0;
        for (const auto &pr : *kin.final_particles) {
            if (pr.first == -2212) { p_pbar = &pr.second; ++n_pbar; }
        }
        if (n_pbar != 1) return false;
        const TLorentzVector p_fromX = kin.p_X - *p_pbar;
        rv.t       = (kin.p_target - kin.p_recoil).M2();
        rv.s_pbarp = (*p_pbar + kin.p_recoil).M2();
        rv.s_pp    = (p_fromX + kin.p_recoil).M2();
        return true;
    }

    bool hasRatioStage() const { return ratio_.hist != nullptr || formula_ != nullptr; }

    double eventWeight(const EventKinematics &kin, RatioVars *out = nullptr) const {
        RatioVars rv;
        const bool ok = ratioVars(kin, rv);
        if (out) *out = rv;
        if (!hasRatioStage()) return 1.0;
        ++n_ratio_eval_;
        if (!ok) { ++n_ratio_topology_; return 0.0; }

        double num, den;
        if (ratio_.hist) {
            // Outside the table's range dsigma/dt says nothing. Inside it,
            // clamp into the bin-center hull: the table is a smooth
            // function, not a per-bin ratio, so the half-bin band at the
            // edge must not be zeroed the way the accept-reject surfaces do.
            if (!ratio_.covers(rv.s_pbarp, rv.t) ||
                !ratio_.covers(rv.s_pp,    rv.t)) { ++n_ratio_out_; return 0.0; }
            num = ratio_.interpolateClamped(rv.s_pbarp, rv.t);
            den = ratio_.interpolateClamped(rv.s_pp,    rv.t);
            if (cfg_.ratio_weight_log) { num = std::exp(num); den = std::exp(den); }
        } else {
            num = evalFormula(rv.s_pbarp, rv.t);
            den = evalFormula(rv.s_pp,    rv.t);
        }
        if (!std::isfinite(num) || !std::isfinite(den) || den <= 0.0 || num < 0.0) {
            ++n_ratio_bad_; return 0.0;
        }
        const double w = num / den;
        sum_w_ratio_ += w;
        return w;
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
        if (hasRatioStage()) {
            const long long n_ok = n_ratio_eval_ - n_ratio_out_
                                 - n_ratio_bad_ - n_ratio_topology_;
            os << "  ratio weight w_ratio (carried, not accept-reject): "
               << n_ratio_eval_ << " events, mean w = "
               << (n_ok > 0 ? sum_w_ratio_ / n_ok : 0.0) << std::endl;
            os << "    - outside dsigma/dt table (w=0):   " << n_ratio_out_ << std::endl;
            os << "    - zero/invalid dsigma/dt (w=0):    " << n_ratio_bad_ << std::endl;
            os << "    - no unique antiproton (w=0):      " << n_ratio_topology_ << std::endl;
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
        bool covers(double x, double y) const {
            return inRange(hist->GetXaxis(), x) && inRange(hist->GetYaxis(), y);
        }
        // For a tabulated smooth function: clamp into the hull of the bin
        // centers (where Interpolate is defined) instead of returning 0.
        double interpolateClamped(double x, double y) const {
            return hist->Interpolate(clampToCenters(hist->GetXaxis(), x),
                                     clampToCenters(hist->GetYaxis(), y));
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

    // dsigma/dt(s, t) as a TFormula. `t` is one of TFormula's four
    // built-in variables (x, y, z, t -> slots 0-3), so after AddVariable("s")
    // the layout is [x, y, z, t, s]: Eval(s, t) would fill slots 0 and 1
    // and silently evaluate garbage. Evaluate through EvalPar() with a
    // buffer filled by variable index instead. Slots 0 and 1 are filled
    // too, so `x` and `y` work as aliases for s and t.
    static constexpr int kFormulaMaxDim = 8;

    void loadFormula(const std::string &expr) {
        auto f = std::make_unique<TFormula>("ratio_dsdt", "", /*addToGlobList=*/false);
        f->AddVariable("s");
        f->AddVariable("t");
        if (f->Compile(expr.c_str()) != 0 || !f->IsValid()) {
            std::cerr << "ERROR: ratio_weight_formula '" << expr
                      << "' does not compile; ratio weight disabled." << std::endl;
            return;
        }
        if (f->GetNdim() > kFormulaMaxDim) {
            std::cerr << "ERROR: ratio_weight_formula has " << f->GetNdim()
                      << " variables (max " << kFormulaMaxDim
                      << "); ratio weight disabled." << std::endl;
            return;
        }
        f_is_ = f->GetVarNumber("s");
        f_it_ = f->GetVarNumber("t");
        formula_ = std::move(f);
        std::cout << "Ratio weight enabled (carried as w_ratio): formula '"
                  << expr << "'  (TFormula in s, t)" << std::endl;
    }

    double evalFormula(double s, double t) const {
        std::array<double, kFormulaMaxDim> x{};
        x[0] = s; x[1] = t;          // x, y aliases
        x[f_is_] = s; x[f_it_] = t;  // named s, t
        return formula_->EvalPar(x.data());
    }

    WeightConfig cfg_;
    Surface2D q2ep_;   // weight_func
    Surface2D pp_;     // mom_weight
    Surface3D xsec_;   // xsec_weight
    Surface2D ratio_;  // ratio_weight (table)
    std::unique_ptr<TFormula> formula_;   // ratio_weight_formula
    int f_is_ = 4, f_it_ = 3;

    long long n_reject_electron_ = 0;
    long long n_reject_mom_      = 0;
    long long n_reject_xsec_     = 0;
    // eventWeight() is const (it does not change the physics); the
    // bookkeeping is mutable.
    mutable long long n_ratio_eval_     = 0;
    mutable long long n_ratio_out_      = 0;
    mutable long long n_ratio_bad_      = 0;
    mutable long long n_ratio_topology_ = 0;
    mutable double    sum_w_ratio_      = 0.0;
};

#endif // EVENT_WEIGHTER_H
