#%%
#%matplotlib qt
import numpy as np
import vector as vec
import matplotlib.pyplot as plt
from breit_wigner import breit_wigner
import random


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
    def __init__(self, beam_energy=None, target_mass=None, num_events=None, smear_sigma=None, input_file=None):
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
        E1 = np.sqrt(p_mag**2 + mass1**2)
        C = M**2 + mass1**2 - 2 * M * E1
        N = len(parent)

        if B is not None:
            # Sample t-values with exp(-t*B) weighting
            t_candidates = np.random.uniform(0, 10, size=N)  # broader pool
            weights = np.exp(-t_candidates * B)
            probs = weights / np.sum(weights)
            t_sampled = np.random.choice(t_candidates, size=N, replace=False, p=probs)

            # Compute cos(theta) from sampled t
            cos_theta = t_sampled / (2 * p_mag**2) - 1
            # cos_theta = np.clip(cos_theta, -1, 1)  # Ensure within valid range
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


    #def generate_scattered_electron( self, Q2_range=(0, 6), E_range=(0, 6), det=None, W_min=4.0):
    def generate_scattered_electron( self, Q2_range=(0, 4), E_range=(0, 6), det=None, W_min=1.5):
        """
        Generate scattered electron 4-momenta.
        Parameters
        ----------
        Q2_range : tuple
            Range of Q2 values (in GeV^2) to sample from.
        E_range : tuple
            Range of scattered electron energies (in GeV) to sample from.
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

        while sum(len(v) for v in p_scattered_all) < num_events:
            Q2 = np.random.uniform(*Q2_range, batch_size)
            E_scattered = np.random.uniform(*E_range, batch_size)
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

        return stack(p_scattered_all), stack(p_virtual_all), stack(p_W_all)
    
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
               # p['vz'] = random.uniform(-10, 2)  # Random vertex z-coordinate for each particle
                #f.write(f"\t{num_particles} 0 0 0 0 0 0\n")  # simple header
                f.write(f"\t{num_particles} 1 1 0 0 11 6.5 2212 0.938272 0\n")  # header with beam and target info
                # Write each particle
                for j, p in enumerate(particles, 1):
                    charge3 = int(p['charge'])
                    type = 1.0 #GEMC only accepts type 1 for particles
                    lifetime = 0.0 # placeholder for lifetime
                    vz = random.uniform(-10, 2)  # Random vertex z-coordinate for each particle; placeholder for vertex z-coordinate; GEMC corrects this when reconstructed
                    #vz = 0.0 # placeholder for vertex z-coordinate
                    #vz = -3.0 # placeholder for vertex z-coordinate; GEMC corrects this when reconstructed
                    f.write(f"{j} {lifetime:.1f} {type:.1f} {p['pid']} 0 0 {p['vec'][i].px:.6f} {p['vec'][i].py:.6f} {p['vec'][i].pz:.6f} "
                            f"{p['vec'][i].E:.6f} {p['mass']:.6f} {p['vx']:.6f} {p['vy']:.6f} {vz:.6f}\n")
# def write_LUND(self, particles: list, filename: str):
#     """
#     Write particles to a LUND file with randomized vz for each event.
#     """
#     with open(filename, "w") as f:
#         num_events = len(particles[0]['vec'])  # Number of events
#         num_particles = len(particles)         # Particles per event

#         for i in range(num_events):
#             vz = random.uniform(-10, 2)  # Random vz for this event
#             f.write(f"\t{num_particles} 1 1 0 0 11 6.5 2212 0.938272 0\n")  # event header

#             for j, p in enumerate(particles, 1):
#                 charge3 = int(p['charge'])
#                 f.write(f"{j} {charge3} {p['pid']} 1 0 0 {p['vec'][i].px:.6f} {p['vec'][i].py:.6f} {p['vec'][i].pz:.6f} "
#                         f"{p['vec'][i].E:.6f} {p['mass']:.6f} 0.000000 0.000000 {vz:.6f}\n")


# #%%

EG = EventGenerator(beam_energy=6.5, target_mass=PDG_ID[2212], num_events=10_000, smear_sigma=0)



#EG = EventGenerator(input_file='/Users/biancagualtieri/Desktop/researchSpring25/simulations/input.txt')
p_scattered, p_virtual, p_W = EG.generate_scattered_electron(W_min = 3.0, det=None)
#p_Km, p_X = EG.two_body_decay(p_W, PDG_ID[321], np.random.normal(2.5, 0., EG.num_events), B = None, detector_mask=None)
#mass2_vals = breit_wigner(mean=2.0, width=0.2, size=len(p_W))
p_Km, p_X = EG.two_body_decay(p_W, PDG_ID[321], np.random.laplace(2.5, 0., len(p_W)) , B = None, detector_mask=None)


p_X_RF = p_X.boostCM_of(p_X) 
p_Kb, p_Xi = EG.two_body_decay(p_X, PDG_ID[321], PDG_ID[3312], B = None, detector_mask=None)
#%%
# Boost the particles to the lab frame
p_Km = p_Km[~np.isnan(p_X.M)]
p_Kb = p_Kb[~np.isnan(p_X.M)]
p_Xi = p_Xi[~np.isnan(p_X.M)]
p_virtual = p_virtual[~np.isnan(p_X.M)]
p_scattered = p_scattered[~np.isnan(p_X.M)]
p_W = p_W[~np.isnan(p_X.M)]
p_X = p_X[~np.isnan(p_X.M)]

p_X_LF = p_X.boost((p_virtual + EG.p_target).to_beta3())
p_Km_LF = p_Km.boost((p_virtual + EG.p_target).to_beta3())
p_Kb_LF = p_Kb.boost((p_X_LF).to_beta3())
p_Xi_LF = p_Xi.boost((p_X_LF).to_beta3())


#Plot 1: Invariant mass of X and Kb+Xi
fig, ax = plt.subplots()
ax.hist(p_X_LF.M, bins=100, range=(2.4, 2.6), label=r'Generated X')
ax.hist((p_Kb_LF + p_Xi_LF).M, bins=100, range=(2.4, 2.6), histtype='step', label=r'Kb+Xi Mass')
ax.legend()
ax.set_xlabel('Mass (GeV/c²)')
ax.set_ylabel('Counts')

# Plot 2: Cos(theta) in X rest frame
fig, ax = plt.subplots()
ax.hist(np.cos(p_Kb.theta), bins=100, range=(-1, 1), label=r'$\cos\theta_{Kb_{XRF}}$')
ax.hist(-np.cos(p_Xi.theta), bins=100, range=(-1, 1), histtype='step', label=r'$-\cos\theta_{Xi_{XRF}}$')
ax.legend()
ax.set_xlabel(r'$\cos\theta$')
ax.set_ylabel('Counts')

fig, ax = plt.subplots()
ax.hist(np.degrees(p_Km.phi), bins=100, range=(-180, 180), label=r'$\phi_{Km_{XRF}}$')
ax.hist(np.degrees(p_X.phi), bins=100, range=(-180, 180), histtype='step', label=r'$\phi_{X_{XRF}}$')
ax.legend()
ax.set_xlabel(r'$\phi$ (rad)')
ax.set_ylabel('Counts')

# Plot 3: Phi in X rest frame
fig, ax = plt.subplots()
ax.hist(np.degrees(p_Kb.phi), bins=100, range=(-180, 180), label=r'$\phi_{Kb_{XRF}}$')
ax.hist(np.degrees(p_Xi.phi), bins=100, range=(-180, 180), histtype='step', label=r'$\phi_{Xi_{XRF}}$')
ax.legend()
ax.set_xlabel(r'$\phi$ (rad)')
ax.set_ylabel('Counts')

# Plot 4: Cos(theta) of X and Km
fig, ax = plt.subplots()
ax.hist(np.cos(p_X.theta), bins=100, range=(-1, 1), label=r'$\cos\theta_{X}$')
ax.hist(-np.cos(p_Km.theta), bins=100, range=(-1, 1), histtype='step', label=r'$-\cos\theta_{Km}$')
ax.legend()
ax.set_xlabel(r'$\cos\theta$')
ax.set_ylabel('Counts')
plt.show()



p_miss = EG.p_beam + EG.p_target - p_scattered - p_Km_LF - p_Kb_LF
# Plot 5: Missing Mass
fig, ax = plt.subplots()
ax.hist(p_miss.M, bins=100, range=(0, 2), alpha=0.7, label='Missing Mass')
ax.set_xlabel('Missing Mass (GeV/c²)')
ax.set_ylabel('Counts')
plt.show()
# hists.histo(p_miss.M, bins=100, range=(0, 2), alpha=0.7, label='Missing Mass')

#t = p_virtual - p_X_LF
t = p_virtual - p_Km_LF
t_prime = EG.p_target - p_X_LF
fig, ax = plt.subplots()


#hists.histo(-t.M2, bins=100, label=r't-Channel ($\gamma^{*}X$)', ax = ax)
# hists.histo(-t_prime.M2, bins=100, histtype = 'step', label=r't-Channel ($p_{target}p_{1}}$)', ax = ax)

fig, ax = plt.subplots()

# Histogram for -t.M2
ax.hist(-t.M2, bins=100, label=r't-Channel ($\gamma^{*}X$)', alpha=0.7)

# Step-style histogram for -t_prime.M2
ax.hist(-t_prime.M2, bins=100, histtype='step', label=r't-Channel ($p_{target}}p_X)')

# Add labels and legend
ax.set_xlabel(r'$-t$ [GeV$^2$]')
ax.set_ylabel('Counts')
ax.legend()

plt.show()
# %%
particles = [
    {'vec': p_Kb_LF, 'pid': 321, 'charge': 1, 'mass': PDG_ID[321], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_Xi_LF, 'pid': 3312, 'charge': -1, 'mass': PDG_ID[3312], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_Km_LF, 'pid': 321, 'charge': 1, 'mass': PDG_ID[321], 'vx': 0, 'vy': 0, 'vz': 0},
    {'vec': p_scattered, 'pid': 11, 'charge': 1, 'mass': PDG_ID[11], 'vx': 0, 'vy': 0, 'vz': 0}
]
# %%
EG.write_LUND(particles, "ep10K_.lund")

#%%
