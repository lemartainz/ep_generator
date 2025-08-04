#%%
%matplotlib qt

import numpy as np
import vector as vec
import matplotlib.pyplot as plt
from scipy.stats import rel_breitwigner


### Constants ###
c = 30 # speed of light in cm/ns
PDG_ID = {2212: 0.9382720813,  # proton mass in GeV/c^2
          211: 0.13957039,    # pion mass in GeV/c^2
          11: 0.0005109989461, # electron mass in GeV/c^2
          22: 0.0,             # photon mass in GeV/c^2
          130: 0.497611,       # K_L mass in GeV/c^2
          310: 0.497611,       # K_S mass in GeV/c^2
          321: 0.493677,       # K+ mass in GeV/c^2
          -321: 0.493677,      # K- mass in GeV/c^2
          2112: 0.9395654133,  # neutron mass in GeV/c^2
          -211: 0.13957039,    # anti-pion mass in GeV/c^2
          3312: 1.321}        # Lambda mass in GeV/c^2

def plot_exp(hist, **kwargs):
    bin_centers = (hist[1][:-1] +
                       hist[1][1:]) / 2
    
    plt.errorbar(bin_centers, hist[0], yerr=np.sqrt(hist[0]), **kwargs)


class EventGenerator:
    def __init__(self, beam_energy=None, target_mass=None, target_pid = None, num_events=None, smear_sigma=None, input_file=None):
        """
        Initialize the EventGenerator with beam energy, target mass, number of events, and smear sigma.
        Parameters
        ----------
        beam_energy : float
            Beam energy in GeV.
        target_mass : float
            Target mass in GeV.
        num_events : int
            Number of events to generate.
        smear_sigma : float
            Smearing sigma for the generated events.
        input_file : str
            Path to the input file containing parameters.
        """ 
        ### Need to add a check for the input file to see if it exists ###
        ### Make sure to add a check on whether input parameters are valid ###

        if input_file is None:
            self.num_events = num_events
            self.c = 30  # cm/ns
            self.beam_energy = beam_energy
            self.target_mass = target_mass
            self.p_beam = vec.obj(px=0, py=0, pz=beam_energy, E=beam_energy) # type: ignore
            self.p_target = vec.obj(px=0, py=0, pz=0, E=target_mass) # type: ignore
            self.smear = np.random.normal(1, smear_sigma, self.num_events) # type: ignore
            if target_pid is None:
                self.target_pid = 2212
            elif target_pid is not None:
                self.target_pid = target_pid
            
        elif input_file is not None:
            with open(input_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    key, value = [x.strip() for x in line.split("=")]
                    if key == "beam_energy":
                        self.beam_energy = float(value)
                    elif key == "target_mass":
                        self.target_mass = float(value)
                    elif key == "num_events":
                        self.num_events = int(value)
                    elif key == "smear_sigma":
                        self.smear_sigma = float(value)
                    elif key == "t_slope":
                        self.t_slope = float(value)
                    elif key == "target_pid":
                        self.target_pid = int(value)
                    else:
                        raise ValueError(f"Unknown parameter: {key}")
                    
            self.p_beam = vec.obj(px=0, py=0, pz=self.beam_energy, E=self.beam_energy) # type: ignore
            self.p_target = vec.obj(px=0, py=0, pz=0, E=self.target_mass) # type: ignore
            self.smear = np.random.normal(1, self.smear_sigma, self.num_events)

    def momentum_magnitude(self, mass_system, mass1, mass2):
        """
        Calculate the momentum magntude for a two-body decay from a parent particle at rest.
        Parameters
        ---------- 
        mass_system : float or np.ndarray
            Mass of the system (parent particle).
        mass1 : float or np.ndarray
            Mass of the first daughter particle.
        mass2 : float or np.ndarray
            Mass of the second daughter particle.
        Returns
        -------
        p_mag : float or np.ndarray
            Momentum magnitude of the daughter particles in the parent rest frame.
        """

        mass_system = np.array(mass_system)
        mask = mass_system >= (mass1 + mass2)
        if isinstance(mass1, np.ndarray):
            mass1 = np.array(mass1)
            mass1 = mass1[mask]
        if isinstance(mass2, np.ndarray):
            mass2 = np.array(mass2)
            mass2 = mass2[mask]

        p_mag = np.full_like(mass_system, np.nan)
        p_mag[mask] = np.sqrt((mass_system[mask]**2 - (mass1 + mass2)**2) *
                              (mass_system[mask]**2 - (mass1 - mass2)**2)) / (2 * mass_system[mask])
        return p_mag
    
    def two_body_decay(self, parent, mass1, mass2, B=None, detector_mask=None):
        """
        Generate two-body decay of a parent particle into two daughter particles.
        Parameters
        ----------
        parent : vec.array
            4-momentum of the parent particle.
        mass1 : float
            Mass of the first daughter particle.
        mass2 : float
            Mass of the second daughter particle.
        B : float, optional
            T slope weighting parameter for the decay, determines the decay angular distribution in parent rest frame.
            If None, uniform distribution is used.
        detector_mask : np.ndarray, optional
            Mask to apply to the parent particles. If None, all parent particles are considered.
        Returns
        -------
        p1 : vec.array
            4-momentum of the first daughter particle. ### Need to boost to lab frame
        p2 : vec.array
            4-momentum of the second daughter particle. ### Need to boost to lab frame
        """

        if detector_mask is None:
            detector_mask = np.ones(len(parent), dtype=bool)

        parent = parent
        M = parent.M
        p_mag = self.momentum_magnitude(M, mass1, mass2)
        # p_mag = p_mag[np.isnan(p_mag) == False]  # Filter out NaN values
        # E1 = np.sqrt(p_mag**2 + mass1**2)
        N = len(parent)
        p_virtual_CM = self.p_virtual.boostCM_of(self.p_W) # Boost virtual photon to CM frame of W and virtual photon
        p_W_CM = self.p_W.boostCM_of(self.p_W) 
        p_target_CM = self.p_target.boostCM_of(self.p_W)  # Boost target to CM frame of W and virtual photon
        # print(f"p_virtual_CM: {p_virtual_CM.M}, p_target_CM: {p_target_CM.M}")

        t_max = ((p_virtual_CM.M2 - mass2**2 - p_target_CM.M2 + mass1**2) / (2 * p_W_CM.M))**2 - (p_virtual_CM.mag + p_mag)**2
        t_min = ((p_virtual_CM.M2 - mass2**2 - p_target_CM.M2 + mass1**2) / (2 * p_W_CM.M))**2 - (p_virtual_CM.mag - p_mag)**2
        self.t_min = t_min
        self.t_max = t_max
        # print(f"t_min: {t_min}, t_max: {t_max}")

        if (B is not None) and (B > 0):
            # Sample t-values with exp(-t*B) weighting
            cos_theta = np.empty(N)
            for i in range(N):
                if t_min[i] >= t_max[i]:
                    t_candidates = np.random.uniform(t_max[i], t_min[i], size=10000)  # broader pool

                    weights = np.exp(-np.abs(t_candidates - t_min[i]) * B)
                    probs = weights / np.sum(weights)
                    t_sampled = np.random.choice(t_candidates, size = 1, p=probs)
                    cos_theta[i] = 1 - 2 * (t_sampled - t_min[i]) / (t_max[i] - t_min[i])

            # plt.hist(t_sampled, bins=10, alpha=0.5, label='t candidates')

        elif (B is None) or (B == 0):
            # Uniform distribution for cos(theta)
            cos_theta = np.random.uniform(-1, 1, N)

        theta = np.arccos(cos_theta)
        phi = np.random.uniform(-np.pi, np.pi, N)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # 4-momentum of decay particle 1 in rest frame of parent
        p1 = vec.array({
            'px': p_mag * sin_theta * np.cos(phi),
            'py': p_mag * sin_theta * np.sin(phi),
            'pz': p_mag * cos_theta,
            'E': np.sqrt(p_mag**2 + mass1**2)
        })

        # Particle 2 gets opposite momentum
        p2 = vec.array({
            'px': -p1.px,
            'py': -p1.py,
            'pz': -p1.pz,
            'E': np.sqrt(p_mag**2 + mass2**2)
        })

        # # Boost both decay products to lab frame
        # boost_vector = parent.boostvec
        # p1 = p1.boost(boost_vector)
        # p2 = p2.boost(boost_vector)

        return p1, p2

    def in_acceptance(self, theta_deg, det="eFD"):
        if det == "eFT":
            return (theta_deg >= 2) & (theta_deg <= 6)
        elif det == "eFD":
            return (theta_deg > 6) & (theta_deg <= 35)
        else:
            return np.full_like(theta_deg, False, dtype=bool)

    # def generate_scattered_electron(self, Q2_range=(0, 6), E_range=(0, 6), W_min = 2, det=None):
    #     Q2 = np.random.uniform(*Q2_range, self.num_events) # type: ignore
    #     E_scattered = np.random.uniform(*E_range, self.num_events) # type: ignore
    #     theta = np.arccos(1 - Q2 / (2 * self.beam_energy * E_scattered)) # type: ignore
    #     phi = np.random.uniform(-np.pi, np.pi, self.num_events)

    #     if det is None:
    #         mask = np.full_like(theta, True, dtype=bool)
    #     else:
    #         mask = self.in_acceptance(np.degrees(theta), det=det)

    #     p_scattered = vec.array({
    #         'px': E_scattered[mask] * np.sin(theta[mask]) * np.cos(phi[mask]),
    #         'py': E_scattered[mask] * np.sin(theta[mask]) * np.sin(phi[mask]),
    #         'pz': E_scattered[mask] * np.cos(theta[mask]),
    #         'E': E_scattered[mask]
    #     })
    #     p_virtual = self.p_beam - p_scattered
    #     p_W = self.p_beam + self.p_target - p_scattered # type: ignore
    #     return p_scattered, p_virtual, p_W, mask
    
    def generate_scattered_electron( self, Q2_range=(0, 6), E_range=(0, 3), det=None, W_min=4.0):
        """
        Generate scattered electron 4-momenta.
        Parameters
        ----------
        Q2_range : tuple or float
            Range of Q2 values (in GeV^2) to sample from as (min, max) tuple,
            or a single Q2 value to use for all events.
        E_range : tuple or float
            Range of scattered electron energies (in GeV) to sample from as (min, max) tuple,
            or a single energy value to use for all events.
        det : str or None   
            Detector type for acceptance cuts. Options are "eFT" or "eFD".
            If None, no acceptance cuts are applied.
        W_min : float
            Minimum W value (in GeV) for the generated events.
        Returns
        -------
        p_scattered : vec.array
            4-momenta of the scattered electrons.
        p_virtual : vec.array
            4-momenta of the virtual photon.
        p_W : vec.array
            4-momenta of the W boson.
        """

        p_scattered_all = []
        p_virtual_all = []
        p_W_all = []
        num_events = self.num_events
        batch_size = max(4 * num_events, 10000)

        # Helper function to generate values based on whether input is single value or range
        def generate_values(param, batch_size):
            if isinstance(param, (tuple, list)) and len(param) == 2:
                # Range provided - sample uniformly
                return np.random.uniform(param[0], param[1], batch_size)
            else:
                # Single value provided - use for all events
                return np.full(batch_size, param)

        while sum(len(v) for v in p_scattered_all) < num_events:
            print(f"Generating events: {sum(len(v) for v in p_scattered_all)} / {num_events}")  
            Q2 = generate_values(Q2_range, batch_size)
            E_scattered = generate_values(E_range, batch_size)
            theta = np.arccos(1 - Q2 / (2 * self.beam_energy * E_scattered))
            phi = np.random.uniform(-np.pi, np.pi, batch_size)

            px = E_scattered * np.sin(theta) * np.cos(phi)
            py = E_scattered * np.sin(theta) * np.sin(phi)
            pz = E_scattered * np.cos(theta)

            p_scattered = vec.array({
                'px': px,
                'py': py,
                'pz': pz,
                'E': E_scattered
            })

            p_virtual = self.p_beam - p_scattered
            p_W = self.p_beam + self.p_target - p_scattered

            W = p_W.mass

            if det is None:
                det_mask = np.full(batch_size, True, dtype=bool)
            else:
                det_mask = self.in_acceptance(np.degrees(theta), det=det)

            good_mask = (W >= W_min) & det_mask
            
            if np.sum(good_mask) == 0:
                print("No events passed the acceptance cuts, retrying...")
                continue

            p_scattered_all.append(p_scattered[good_mask])
            p_virtual_all.append(p_virtual[good_mask])
            p_W_all.append(p_W[good_mask])

        def stack(vec_list):
            return vec.array({
                'px': np.concatenate([v.px for v in vec_list])[:num_events],
                'py': np.concatenate([v.py for v in vec_list])[:num_events],
                'pz': np.concatenate([v.pz for v in vec_list])[:num_events],
                'E':  np.concatenate([v.E  for v in vec_list])[:num_events],
            })
        self.p_W = stack(p_W_all)
        self.p_virtual = stack(p_virtual_all)
        self.p_scattered = stack(p_scattered_all)

        return self.p_scattered, self.p_virtual, self.p_W

    def write_LUND(self, particles: list, filename: str):
        """
        Write particles to a LUND file.

        Parameters
        ----------
        particles : list
            List of dictionaries containing particle information.
            Each dictionary should have the following keys:
                - 'vec': 4-momentum vector of the particle (vec.array)
                - 'pid': Particle ID (int)
                - 'charge': Charge of the particle (int)
                - 'mass': Mass of the particle (float)
                - 'vx', 'vy', 'vz': Vertex coordinates (float)
        filename : str
            Name of the output LUND file.
        """

        with open(filename, "w") as f:
            for i in range(len(p_scattered) - 1):
                # Write event header: number of particles, target/beam info (fill with dummy if unknown)
                num_particles = len(particles)
                f.write(f"\t{num_particles} 1 1 0.0 0.0 {self.target_pid} {self.beam_energy}\n")  # simple header

                # Randomly generate a vertex position (vx, vy, vz) for each event
                # vx = np.random.uniform(-1, 1)
                # vy = np.random.uniform(-1, 1)
                vz = np.random.uniform(-5, 0)
                # Write each particle
                for j, p in enumerate(particles, 1):
                    f.write(f"{j} 0.0 1 {p['pid']} 0 0 {p['vec'][i].px:.6f} {p['vec'][i].py:.6f} {p['vec'][i].pz:.6f} "
                            f"{p['vec'][i].E:.6f} {p['mass']:.6f} {p['vx']:.6f} {p['vy']:.6f} {vz:.6f}\n")


#%%

# EG = EventGenerator(beam_energy=10.2, target_mass=PDG_ID[2212], num_events=10, smear_sigma=0)
EG = EventGenerator(input_file='input.txt')
p_scattered, p_virtual, p_W = EG.generate_scattered_electron(Q2_range = 6, E_range = 4, W_min = 4, det=None)
p_p1, p_X = EG.two_body_decay(p_W, PDG_ID[2212], 2.325, B = 1, detector_mask=None)
p_X_RF = p_X.boostCM_of(p_X)
p_p2, p_pbar = EG.two_body_decay(p_X, PDG_ID[2212], PDG_ID[2212], B = None, detector_mask=None)
#%%
# Boost the particles to the lab frame
p_p1 = p_p1[~np.isnan(p_X.M)]
p_p2 = p_p2[~np.isnan(p_X.M)]
p_pbar = p_pbar[~np.isnan(p_X.M)]
p_virtual = p_virtual[~np.isnan(p_X.M)]
p_scattered = p_scattered[~np.isnan(p_X.M)]
p_W = p_W[~np.isnan(p_X.M)]
t_min = EG.t_min
t_min = t_min[~np.isnan(p_X.M)]
p_X = p_X[~np.isnan(p_X.M)]
# t_min = t_min[~np.isnan(p_X.M)]

p_X_LF = p_X.boost((p_virtual + EG.p_target).to_beta3())
p_p1_LF = p_p1.boost((p_virtual + EG.p_target).to_beta3())
p_p2_LF = p_p2.boost((p_X_LF).to_beta3())
p_pbar_LF = p_pbar.boost((p_X_LF).to_beta3())
# p_virtual_CM = p_virtual.boost((p_virtual + EG.p_target).to_beta3())
# p_target_CM = EG.p_target.boost((p_virtual + EG.p_target).to_beta3())

fig, ax = plt.subplots()
h_XM = plt.hist(p_X_LF.M, bins = 100, range = (2., 3), )
plot_exp(h_XM, color = 'black', fmt = '.', label=r'Generated X')
h_p2pbarM = plt.hist((p_p2_LF + p_pbar_LF).M, bins = 100, range = (2., 3), color = 'red', alpha = .5, label = r'$p_{M}\bar{p}$ Mass')
#%%
fig, ax = plt.subplots()
# plt.hist(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
plt.hist(np.cos(p_p2.theta), bins = 100, range = (-1, 1), label = r'$\cos\theta_{p_{XRF}}$')
plt.hist(-np.cos(p_pbar.theta), bins = 100, range = (-1, 1), histtype = 'step', label = r'$-\cos\theta_{\bar{p}_{XRF}}$')

fig, ax = plt.subplots()
# plt.hist(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
plt.hist((p_p2.phi), bins = 100, range = (-np.pi, np.pi), label = r'$\phi_{p_{XRF}}$')
plt.hist((p_pbar.phi), bins = 100, range = (-np.pi, np.pi), histtype = 'step', label = r'$\phi_{\bar{p}_{XRF}}$')

fig, ax = plt.subplots()
# plt.hist(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
plt.hist(np.cos(p_X.theta), bins = 100, range = (-1, 1), label = r'$\cos\theta_{p_{XRF}}$')
test = plt.hist(-np.cos(p_p1.theta), bins = 100, range = (-1, 1), histtype = 'step', label = r'$-\cos\theta_{\bar{p}_{XRF}}$')
#%%
t = p_virtual - p_p1_LF
t_prime = EG.p_target - p_X_LF
fig, ax = plt.subplots()
h_t = plt.hist(-t.M2+t_min, bins=100, label=r't-Channel ($\gamma^{*}X$)')
# plt.hist(-t_prime.M2, bins=100, histtype = 'step', linewidth = 2)

def fit_func(t, a, b, c):
    return a * np.exp(-b * np.abs(t + c))

from scipy.optimize import curve_fit
# Fit the t-channel distribution
t_fit = h_t[1][:-1] + np.diff(h_t[1]) / 2
fit_range = (t_fit > 0) & (t_fit < 5)
t_fit = t_fit[fit_range]
h_t_fit = h_t[0][fit_range]
params, unc = curve_fit(fit_func, t_fit, h_t_fit, p0=[1, 0, 0], bounds=([0, 0, -np.inf], [np.inf, 10.0, np.inf]))

fitted_values = fit_func(t_fit, *params)
plt.plot(t_fit, fitted_values, color='red', label=f'B: {params[1]:.2f} ± {unc[1,1]:.5f}')
plt.xlabel(r'$-t$ (GeV$^2$)')
plt.ylabel('Counts')
plt.title('t-Channel Distribution')
plt.legend()

#%%

p_miss = EG.p_beam + EG.p_target - p_scattered - p_p1_LF - p_p2_LF

fig, ax = plt.subplots()
plt.hist(p_miss.M, bins=100, range=(-2, 2), label=r'$p_{miss}$ Mass')

# %%
particles = [
    {'vec': p_p2_LF, 'pid': 2212, 'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_pbar_LF, 'pid': -2212,  'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_p1_LF, 'pid': 2212, 'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_scattered, 'pid': 11, 'mass': PDG_ID[11], 'vx': 0, 'vy': 0, 'vz': 0}
]
# %%
EG.write_LUND(particles, "../Simulations/test.lund")
# %% This is a test for the van Hove plot

particles = {'p_e': p_scattered, 'p_p1': p_p1_LF, 'p_p2': p_p2_LF, 'p_pbar': p_pbar_LF}

E_beam = 10.2  # GeV
beam = vector.obj(px=0, py=0, pz=E_beam, E=E_beam)
target = vector.obj(px=0, py=0, pz=0, E=0.938)  # proton target

def compute_lps(res : dict):
    """
    event: dict with 4-momenta for scattered electron, proton1, proton2, antiproton
    Returns: (X, Y) coordinates in van Hove plot
    """

    # Final state particles
    eprime = res['p_e']
    p1     = res['p_p1']
    p2     = res['p_p2']
    pbar   = res['p_pbar']

    # Reconstruct X = p + p + pbar system
    X = beam + target - eprime

    # Boost hadrons to CM of hadronic system
    # boost_vector = -X.to_beta3()
    p1_cm = p1.boostCM_of(p1 + p2 + pbar)
    p2_cm = p2.boostCM_of(p1 + p2 + pbar)
    pbar_cm = pbar.boostCM_of(p1 + p2 + pbar)
    # p2_cm = p2.boost_p4(boost_vector)
    # pbar_cm = pbar.boost_p4(boost_vector)

    # Choose beam axis as z
    z_hat = np.array([0, 0, 1])
    def proj_longitudinal(p):
        p_vec = np.array([p.px, p.py, p.pz])
        return np.dot(p_vec, z_hat)

    pL1 = p1_cm.pz
    pL2 = p2_cm.pz
    pL3 = pbar_cm.pz
    # pL2 = proj_longitudinal(p2_cm)
    # pL3 = proj_longitudinal(pbar_cm)

    # Symmetrized van Hove coordinates
    X_vh = (pL1 - pL2) / np.sqrt(2)
    Y_vh = (pL1 + pL2 - 2 * pL3) / np.sqrt(6)

    return X_vh, Y_vh

# %%
import matplotlib as mpl
x, y = compute_lps(particles)
# plt.figure(figsize=(12,10))
hists.histo2d(x, y, bins = 100, range  = ((-4, 4), (-4, 4)), norm = mpl.colors.LogNorm())
plt.xlabel("X (van Hove)")
plt.ylabel("Y (van Hove)")
# Plot symmetry axes corresponding to q1=q2, q2=q3, q3=q1
length = 4.5  # Extend slightly beyond plot range

# q1 = q2 -> X = 0 (vertical line)
plt.plot([-length, length], [0, 0], 'k--', lw=1, label='$q_1 = q_2$')

# q2 = q3 -> theta = -120 degrees
theta2 = -2 * np.pi / 3
plt.plot([-length * np.cos(theta2), length * np.cos(theta2)], [-length * np.sin(theta2), length * np.sin(theta2)], 'r--', lw=1, label='$q_2 = q_3$')

# q3 = q1 -> theta = +120 degrees
theta3 = 2 * np.pi / 3
plt.plot([-length * np.cos(theta3), length * np.cos(theta3)], [-length * np.sin(theta3), length * np.sin(theta3)], 'b--', lw=1, label='$q_3 = q_1$')

# plt.legend(loc='upper right')
plt.title("Longitudinal Phase Space (van Hove plot)")
# plt.grid(True)
# plt.gca().set_aspect('equal')
plt.show()
# %%
hists.histo2d((particles['p_p1'] + particles['p_pbar']).M, (particles['p_p2'] + particles['p_pbar']).M, bins = 100, range = ((1.5, 4), (1.5, 4)), norm = mpl.colors.LogNorm())
# %%
