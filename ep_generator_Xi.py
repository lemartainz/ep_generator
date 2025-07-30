#%%

import numpy as np
import vector as vec
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
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
    
    plt.errorbar(
        bin_centers, hist[0], yerr=np.sqrt(hist[0]), **kwargs)

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
    
    def two_body_decay(self, parent, mass1, mass2, B = 4, detector_mask=None):
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

        # if B is not None:
        #     # Sample t-values with exp(-t*B) weighting
        #     t_candidates = np.random.uniform(0, 10, size=N)  # broader pool
        #     weights = np.exp(-t_candidates * B)
        #     probs = weights / np.sum(weights)
        #     t_sampled = np.random.choice(t_candidates, size=N, replace=False, p=probs)

        #     # Compute cos(theta) from sampled t
        #     cos_theta = t_sampled / (2 * p_mag**2) - 1
        #     cos_theta = np.clip(cos_theta, -1, 1) 
        #     # cos_theta = np.clip(cos_theta, -1, 1)  # Ensure within valid range
        # else:
        #     cos_theta = np.random.uniform(-1, 1, N)

        # theta = np.arccos(cos_theta)
        # phi = np.random.uniform(-np.pi, np.pi, N)
        # sin_theta = np.sqrt(1 - cos_theta**2)

        if B is not None:
            ## will be implementing this calculation for each bin later ##
            t_min = -3.1037
            t_max = -0.3874

            exp_max = np.exp(-t_max * B)
            exp_min = np.exp(-t_min * B)
            exp_t = np.random.uniform(exp_max, exp_min, N)
            t = -np.log(exp_t) / B  

            

            # Convert t to cos(theta)
            cos_theta = 1 - 2 * (t - t_min) / (t_max - t_min)
            cos_theta = np.clip(cos_theta, -1, 1)
            
        else:
            cos_theta = np.random.uniform(-1, 1, N)

            # Recalculate theta/phi using sampled subset
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(-np.pi, np.pi, len(cos_theta))
        sin_theta = np.sqrt(1 - cos_theta**2)    


    # Recalculate daughter 4-vectors here for the masked events only...
    # (you’ll also need to apply this mask to the parent array accordingly)



    # Recalculate daughter 4-vectors here for the masked events only...
    # (you’ll also need to apply this mask to the parent array accordingly)


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


        return p1, p2

    def in_acceptance(self, theta_deg, det="eFD"):
        if det == "eFT":
            return (theta_deg >= 2) & (theta_deg <= 6)
        elif det == "eFD":
            return (theta_deg > 6) & (theta_deg <= 35)
        else:
            return np.full_like(theta_deg, False, dtype=bool)


    def generate_scattered_electron( self, Q2_range=(0, 4), E_range=(0, 6), det=None, W_min=2.35):
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
    
    # def write_LUND(self, particles: list, filename: str):
    #     """
    #     Write particles to a LUND file.

    #     Parameters
    #     ----------
    #     particles : list
    #         List of dictionaries containing particle information.
    #         Each dictionary should have the following keys:
    #             - 'vec': 4-momentum vector of the particle (vec.array)
    #             - 'pid': Particle ID (int)
    #             - 'charge': Charge of the particle (int)
    #             - 'mass': Mass of the particle (float)
    #             - 'vx', 'vy', 'vz': Vertex coordinates (float)
    #     filename : str
    #         Name of the output LUND file.
    #     """

    #     with open(filename, "w") as f:
    #         for i in range(len(p_scattered) - 1):
    #             # Write event header: number of particles, target/beam info (fill with dummy if unknown)
    #             num_particles = len(particles)
    #             f.write(f"\t{num_particles} 0 0 0 0 0 0\n")  # simple header

    #             # Write each particle
    #             for j, p in enumerate(particles, 1):
    #                 charge3 = int(p['charge'])
    #                 f.write(f"{j} {charge3} {p['pid']} 1 0 0 {p['vec'][i].px:.6f} {p['vec'][i].py:.6f} {p['vec'][i].pz:.6f} "
    #                         f"{p['vec'][i].E:.6f} {p['mass']:.6f} {p['vx']:.6f} {p['vy']:.6f} {p['vz']:.6f} 0.0\n")

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

for i in range(10):
    EG = EventGenerator(beam_energy=6.5, target_mass=PDG_ID[2212], num_events=5000, smear_sigma=0)
    p_scattered, p_virtual, p_W = EG.generate_scattered_electron(W_min=2.4, det=None)
    p_k1, p_X = EG.two_body_decay(p_W, PDG_ID[321], np.random.normal(2.5, 0.2, EG.num_events), B = 4)
    p_X_RF = p_X.boostCM_of(p_X)
    p_k2, p_Xi = EG.two_body_decay(p_X, PDG_ID[321], PDG_ID[3312], B = 4)

    # Remove events with invalid X mass
    valid_mask = ~np.isnan(p_X.M)
    p_k1 = p_k1[valid_mask]
    p_k2 = p_k2[valid_mask]
    p_Xi = p_Xi[valid_mask]
    p_virtual = p_virtual[valid_mask]
    p_scattered = p_scattered[valid_mask]
    p_W = p_W[valid_mask]
    p_X = p_X[valid_mask]

    # Boost particles to lab frame
    p_X_LF = p_X.boost((p_virtual + EG.p_target).to_beta3())
    p_k1_LF = p_k1.boost((p_virtual + EG.p_target).to_beta3())
    p_k2_LF = p_k2.boost(p_X_LF.to_beta3())
    p_Xi_LF = p_Xi.boost(p_X_LF.to_beta3())

    particles = [
        {'vec': p_k2_LF, 'pid': 321, 'charge': 1, 'mass': PDG_ID[321], 'vx': 0, 'vy': 0, 'vz': 0},
        {'vec': p_Xi_LF, 'pid': 3312, 'charge': -1, 'mass': PDG_ID[3312], 'vx': 0, 'vy': 0, 'vz': 0},
        {'vec': p_k1_LF, 'pid': 321, 'charge': 1, 'mass': PDG_ID[321], 'vx': 0, 'vy': 0, 'vz': 0},
        {'vec': p_scattered, 'pid': 11, 'charge': 1, 'mass': PDG_ID[11], 'vx': 0, 'vy': 0, 'vz': 0}
    ]

    filename = f"test_B4_25_2{i}.lund"
    EG.write_LUND(particles, filename)
    print(f"Saved {filename}")


