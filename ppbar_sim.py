#%%
%matplotlib qt

import numpy as np
import vector as vec
import matplotlib.pyplot as plt


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


class EventGenerator:
    def __init__(self, beam_energy, target_mass, num_events=1_000_000, smear_sigma=0.01):
        self.num_events = num_events
        self.c = 30  # cm/ns
        self.beam_energy = beam_energy
        self.target_mass = target_mass
        self.p_beam = vec.obj(px=0, py=0, pz=beam_energy, E=beam_energy)
        self.p_target = vec.obj(px=0, py=0, pz=0, E=target_mass)
        self.smear = np.random.normal(1, smear_sigma, self.num_events)

    def momentum_magnitude(self, mass_system, mass1, mass2):
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

    # def two_body_decay(self, parent, mass1, mass2, B = None, detector_mask=None):
    #     if detector_mask is None:
    #         detector_mask = np.ones(len(parent), dtype=bool)
    #     smear_mask = self.smear[detector_mask]
    #     p_mag = self.momentum_magnitude(parent.M, mass1, mass2) * smear_mask
    #     cos_theta = np.random.uniform(-1, 1, len(p_mag))
    #     theta = np.arccos(cos_theta)
    #     phi = np.random.uniform(-np.pi, np.pi, len(p_mag))

    #     p1 = vec.array({
    #         'px': p_mag * np.sin(theta) * np.cos(phi),
    #         'py': p_mag * np.sin(theta) * np.sin(phi),
    #         'pz': p_mag * np.cos(theta),
    #         'E': np.sqrt(p_mag**2 + mass1**2)
    #     })    

    #     if B is not None: 
    #         parent = parent.boostCM_of(parent + p1)
    #         p1 = p1.boostCM_of(parent + p1)

    #         t_slope = np.random.uniform(0, 10, len(parent))
    #         weight = np.exp(-t_slope / B)
    #         prob = weight / np.sum(weight)
    #         t_sample = np.random.choice(t_slope, size=len(parent), p=prob)
    #         cos_theta = (t_sample - (parent.M2 + p1.M2 - 2 * (parent.E * p1.E))) / (2 * parent.mag * p1.mag)
        
    #     theta = np.arccos(cos_theta) 
    #     p1 = vec.array({
    #         'px': p_mag * np.sin(theta) * np.cos(phi),
    #         'py': p_mag * np.sin(theta) * np.sin(phi),
    #         'pz': p_mag * np.cos(theta),
    #         'E': np.sqrt(p_mag**2 + mass1**2)
    #     })
    #     p2 = vec.array({
    #         'px': -p1.px,
    #         'py': -p1.py,
    #         'pz': -p1.pz,
    #         'E': np.sqrt(p_mag**2 + mass2**2)
    #     })
    #     return p1, p2
    
    def two_body_decay(self, parent, mass1, mass2, B=None, detector_mask=None):
        if detector_mask is None:
            detector_mask = np.ones(len(parent), dtype=bool)

        parent = parent
        smear_mask = self.smear[detector_mask]
        M = parent.M
        p_mag = self.momentum_magnitude(M, mass1, mass2) * smear_mask
        E1 = np.sqrt(p_mag**2 + mass1**2)
        C = M**2 + mass1**2 - 2 * M * E1
        N = len(parent)

        if B is not None:
            # Sample t-values with exp(-t/B) weighting
            t_candidates = np.random.uniform(0, 10, size=5 * N)  # broader pool
            weights = np.exp(-t_candidates * B)
            probs = weights / np.sum(weights)
            t_sampled = np.random.choice(t_candidates, size=N, replace=False, p=probs)

            # Compute cos(theta) from sampled t
            cos_theta = (t_sampled - C) / (2 * p_mag**2)
            # cos_theta = t_sampled / (2 * p_mag**2) - 1
            cos_theta = np.clip(cos_theta, -1, 1)
        else:
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

    def generate_scattered_electron(self, Q2_range=(0, 6), W_range=(0.938, 6), det=None):
        Q2 = np.random.uniform(*Q2_range, self.num_events)
        W = np.random.uniform(*W_range, self.num_events)
        E_transfer = (Q2 + W**2 - self.target_mass**2) / (2 * self.target_mass)
        E_scattered = np.abs(self.beam_energy - E_transfer)
        theta = np.arccos(1 - Q2 / (2 * self.beam_energy * E_scattered))
        phi = np.random.uniform(-np.pi, np.pi, self.num_events)

        if det is None:
            mask = np.full_like(theta, True, dtype=bool)
        else:
            mask = self.in_acceptance(np.degrees(theta), det=det)

        p_scattered = vec.array({
            'px': E_scattered[mask] * np.sin(theta[mask]) * np.cos(phi[mask]),
            'py': E_scattered[mask] * np.sin(theta[mask]) * np.sin(phi[mask]),
            'pz': E_scattered[mask] * np.cos(theta[mask]),
            'E': E_scattered[mask]
        })
        p_virtual = self.p_beam - p_scattered
        p_W = self.p_beam + self.p_target - p_scattered
        return p_scattered, p_virtual, p_W, mask
    
    

#%%

EG = EventGenerator(beam_energy=10.2, target_mass=PDG_ID[2212], num_events=1_000_000, smear_sigma=0)
p_scattered, p_virtual, p_W, detector_mask = EG.generate_scattered_electron(det=None)

p_p1, p_X = EG.two_body_decay(p_W, PDG_ID[2212], 3, B = None, detector_mask=detector_mask)
p_X_RF = p_X.boostCM_of(p_X)
p_p2, p_pbar = EG.two_body_decay(p_X_RF, PDG_ID[2212], PDG_ID[2212], B = None, detector_mask=detector_mask)
# Boost the particles to the lab frame
p_p1 = p_p1[~np.isnan(p_X.M)]
p_p2 = p_p2[~np.isnan(p_X.M)]
p_pbar = p_pbar[~np.isnan(p_X.M)]
p_virtual = p_virtual[~np.isnan(p_X.M)]
p_scattered = p_scattered[~np.isnan(p_X.M)]
p_X = p_X[~np.isnan(p_X.M)]

p_X_LF = p_X.boost((p_virtual + EG.p_target).to_beta3())
p_p1_LF = p_p1.boost((p_virtual + EG.p_target).to_beta3())
p_p2_LF = p_p2.boost((p_X_LF).to_beta3())
p_pbar_LF = p_pbar.boost((p_X_LF).to_beta3())

fig, ax = hists.plt.subplots()
h_XM = hists.histo(p_X_LF.M, bins = 100, range = (2., 4), color = 'white', ax = ax)
h_XM.plot_exp(color = 'black', fmt = '.', label=r'Generated X', ax = ax)
h_p2pbarM = hists.histo((p_p2_LF + p_pbar_LF).M, bins = 100, range = (2., 4), color = 'red', alpha = .5, label = r'$p_{M}\bar{p}$ Mass', ax = ax)

fig, ax = hists.plt.subplots()
# hists.histo(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
hists.histo(np.cos(p_p2.theta), bins = 100, range = (-1, 1), label = r'$\cos\theta_{p_{XRF}}$', ax = ax)
hists.histo(-np.cos(p_pbar.theta), bins = 100, range = (-1, 1), histtype = 'step', label = r'$-\cos\theta_{\bar{p}_{XRF}}$', ax = ax)

fig, ax = hists.plt.subplots()
# hists.histo(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
hists.histo((p_p2.phi), bins = 100, range = (-np.pi, np.pi), label = r'$\phi_{p_{XRF}}$', ax = ax)
hists.histo((p_pbar.phi), bins = 100, range = (-np.pi, np.pi), histtype = 'step', label = r'$\phi_{\bar{p}_{XRF}}$', ax = ax)

fig, ax = hists.plt.subplots()
# hists.histo(np.cos(p_X.theta), bins = 100, range = (0, np.pi), label=r'Generated X')
hists.histo(np.cos(p_X.theta), bins = 100, range = (-1, 1), label = r'$\cos\theta_{p_{XRF}}$', ax = ax)
hists.histo(-np.cos(p_p1.theta), bins = 100, range = (-1, 1), histtype = 'step', label = r'$-\cos\theta_{\bar{p}_{XRF}}$', ax = ax)
#%%
# p_miss = EG.p_beam + EG.p_target - p_scattered - p_p1_LF - p_p2_LF
# hists.histo(p_miss.M, bins=100, range=(0, 2), alpha=0.7, label='Missing Mass')

t = p_virtual - p_X_LF
t_prime = EG.p_target - p_p1_LF
fig, ax = plt.subplots()
hists.histo(-t.M2, bins=100, label=r't-Channel ($\gamma^{*}X$)', ax = ax)
hists.histo(-t_prime.M2, bins=100, histtype = 'step', label=r't-Channel ($p_{target}p_{1}}$)', ax = ax)

import scipy.optimize as opt

def exp_func(t, A, B):
    return A * np.exp(-t / B)

hist, bins = np.histogram(-t.M2, bins=100, range=(0, 3))
bin_centers = 0.5 * (bins[1:] + bins[:-1])

popt, _ = opt.curve_fit(exp_func, bin_centers[bin_centers > 1], hist[bin_centers > 1], p0=[1e4, 1.0])
A_fit, B_fit = popt

plt.figure()
plt.plot(bin_centers, hist, label="Data")
plt.plot(bin_centers, exp_func(bin_centers, *popt), label=f"Fit: B={B_fit:.2f}")
plt.yscale("log")
plt.legend()




#%%
fig, ax = plt.subplots()
hists.histo(p_p2.phi, bins = 100, range = (-np.pi, np.pi), ax=ax)
hists.histo(p_pbar.phi, bins = 100, range = (-np.pi, np.pi), ax=ax, histtype='step')

fig, ax = plt.subplots()
hists.histo(np.cos(p_p2.theta), bins = 100, ax = ax)
hists.histo(np.cos(p_pbar.theta), bins = 100, ax = ax, histtype='step')
#%%
hists.histo((p_p2_LF + p_pbar_LF).M, bins=100, range=(1.5, 3))

# %%
# EG = EventGenerator(beam_energy=10.2, target_mass=PDG_ID[2212], num_events=1000000, smear_sigma=0.1)
# p_scattered, p_virtual, p_W, detector_mask = EG.generate_scattered_electron(det="eFT")

# p_p1, p_X0 = EG.two_body_decay(p_W, PDG_ID[2212], np.random.normal(2.5, 0.2, EG.num_events)[detector_mask], detector_mask)
# p_pim, p_X = EG.two_body_decay(p_X0, PDG_ID[211], np.random.normal(2.0, 0.2, EG.num_events)[detector_mask], detector_mask)
# p_p2, p_nbar = EG.two_body_decay(p_X, PDG_ID[2212], PDG_ID[2112], detector_mask)
# # Boost the particles to the lab frame
# p_p1 = p_p1[~np.isnan(p_X.M)]
# p_p2 = p_p2[~np.isnan(p_X.M)]
# p_nbar = p_nbar[~np.isnan(p_X.M)]
# p_virtual = p_virtual[~np.isnan(p_X.M)]
# p_scattered = p_scattered[~np.isnan(p_X.M)]
# p_X = p_X[~np.isnan(p_X.M)]


# p_X_LF = p_X.boost((p_virtual + EG.p_target).to_beta3())
# p_p1_LF = p_p1.boost((p_virtual + EG.p_target).to_beta3())
# p_p2_LF = p_p2.boost((p_X_LF).to_beta3())
# p_nbar_LF = p_nbar.boost((p_X_LF).to_beta3())
# %%
particles = [
    {'vec': p_p2_LF, 'pid': 2212, 'charge': 1, 'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_pbar_LF, 'pid': -2212, 'charge': -1, 'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_p1_LF, 'pid': 2212, 'charge': 1, 'mass': PDG_ID[2212], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_scattered, 'pid': 11, 'charge': 1, 'mass': PDG_ID[11], 'vx': 0, 'vy': 0, 'vz': 0}
]

with open("output.dat", "w") as f:
    for i in range(len(p_scattered) - 1):
        # Write event header: number of particles, target/beam info (fill with dummy if unknown)
        num_particles = len(particles)
        f.write(f"\t{num_particles} 0 0 0 0 0 0\n")  # simple header

        # Write each particle
        for j, p in enumerate(particles, 1):
            charge3 = int(p['charge'])
            f.write(f"{j} {charge3} {p['pid']} 1 0 0 {p['vec'][i].px:.6f} {p['vec'][i].py:.6f} {p['vec'][i].pz:.6f} "
                    f"{p['vec'][i].E:.6f} {p['mass']:.6f} {p['vx']:.6f} {p['vy']:.6f} {p['vz']:.6f} 0.0\n")
# %%
