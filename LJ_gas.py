#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas.py

Core module for molecular dynamics simulations of Lennard-Jones gases in the 
NVE and NVT ensembles. Defines data structures (ParticleSystem, SimulationParameters), 
integration schemes (Velocity Verlet, Langevin BAOAB), and energy/force calculations 
based on Lennard-Jones interactions.

Author: Bettina Keller
Created: May 28, 2025

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R, Avogadro
import matplotlib.pyplot as plt

#----------------------------------------------------------------
#   C L A S S E S
#----------------------------------------------------------------
class ParticleSystem:
    def __init__(self, n_particles):
        self.n = n_particles
        
        # Properties for each particle
        self.mass = np.zeros(n_particles)
        self.sigma = np.zeros(n_particles)
        self.epsilon = np.zeros(n_particles)
        
        # 3D positions, velocities, forces, and random numbers (shape: n_particles x 3)
        self.position = np.zeros((n_particles, 3))
        self.unwrapped_position = np.zeros((n_particles, 3)) #position without PBC
        self.velocity = np.zeros((n_particles, 3))
        self.force = np.zeros((n_particles, 3))
        self.random_number = np.zeros((n_particles, 3))
    
    #---------------------
    # With these functions the parameters and states of individual atoms can be changed.
    # In vectorized programming, they will not be used very often
    #
    def set_parameters(self, i, mass, sigma, epsilon):
        """Set the paramters of the i-th particle
            mass in units of u 
            sigma in units of nm 
            epsilon in units of kJ/mol 
        """
        self.mass[i] = mass
        self.sigma[i] = sigma
        self.epsilon[i] = epsilon

    def set_position(self, i, position):
        """Set the paramters of the i-th particle"""
        self.position[i] = position
        
    def set_velocity(self, i, velocity):
        """Set the paramters of the i-th particle"""    
        self.velocity[i] = velocity            

    def set_force(self, i, force):
        """Set the paramters of the i-th particle"""    
        self.force[i] = force            

    def set_random_number(self, i, random_number):
        """Set the paramters of the i-th particle"""    
        self.random_number[i] = random_number            

    def __repr__(self):
        return f"<ParticleSystem with {self.n} particles>"


class SimulationParameters:
    def __init__(self, dt, n_steps, temperature, box_length, sd_eta, tau_thermostat = None, rij_min=0.0):
        """
        Parameters:
            dt (float): Time step in ps.
            n_steps (int): Number of time steps.
            temperature (float): Temperature in K.
            box_length (float): Length of the (cubic) simulation box in nm.

        Parameters with default values: 
            tau_thermostat (float or None) = None: Thermostat coupling constant in ps
                                                   If None, not thermostat is applied 
            rij_min (float) = 0.0: Lower cutoff for interparticle distances (in nm).
        """
        self.dt = dt
        self.n_steps = n_steps
        self.temperature = temperature
        self.box_length = box_length  # in nm
        self.tau_thermostat = tau_thermostat  # thermostat coupling time in ps
        self.rij_min = rij_min        # minimum allowed pairwise distance
        self.sd_eta = sd_eta            #eta used for steepest descent

        # Optional: friction coefficient for Langevin or stochastic thermostats
        self.xi = None
        if self.tau_thermostat is not None and self.tau_thermostat > 0.0:
            self.xi = 1.0 / self.tau_thermostat


#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------

def update_rdf_histogram(ps, sim, hist, dr):
    """
    Add one simulation snapshot to the RDF histogram (vectorized).
    """
    N = ps.n
    L = sim.box_length
    r_max = L / 2.0

    i_upper = np.triu_indices(N, k=1)
    rij = ps.position[i_upper[1]] - ps.position[i_upper[0]]
    rij -= L * np.round(rij / L)          # minimum image convention
    r = np.linalg.norm(rij, axis=1)

    mask = r < r_max
    r_valid = r[mask]

    indices = np.clip((r_valid / dr).astype(int), 0, len(hist) - 1)
    np.add.at(hist, indices, 2.0)

def finalize_rdf(hist, n_samples, sim, ps, dr):
    """
    Normalize the accumulated RDF histogram.
    """

    N = ps.n
    L = sim.box_length

    rho = N / L**3

    r_max = L / 2.0

    bins = np.arange(0, r_max + dr, dr)

    g = np.zeros_like(hist)

    for i in range(len(hist)):

        r1 = bins[i]
        r2 = bins[i+1]

        shell_volume = (4.0*np.pi/3.0)*(r2**3-r1**3)

        ideal = rho * shell_volume * N

        g[i] = hist[i] / (ideal * n_samples)

    r = 0.5*(bins[:-1]+bins[1:])

    return r, g

#--------------------------------------
# Initialization
#--------------------------------------
def initialize_positions(ps: ParticleSystem, box_length_in_nm: float):
    """Initialize particle positions uniformly in a cubic box."""
    ps.position[:] = np.random.uniform(0, box_length_in_nm, size=(ps.n, 3))

def initialize_velocities(ps: ParticleSystem, temperature: float):
    """
    Initializes velocities of a ParticleSystem according to the Maxwell-Boltzmann
    distribution at a given temperature T (in Kelvin), using vectorized NumPy operations.

    Each velocity component is sampled from a Gaussian with:
        variance = sigma^2 = R*T / M
    
    Velocities are returned in units of nm/ps.
    """
    # molar masses in kg/mol (convert from u)
    M = ps.mass * 1e-3  # shape: (n,)
    
    # Compute standard deviations σ = sqrt(RT/M) in m/s
    stddev = np.sqrt(R * temperature / M)  # shape: (n,) 
    
    # Sample velocities: each component independently, shape (n, 3)
    velocities_m_s = np.random.normal(0.0, stddev[:, np.newaxis], size=(ps.n, 3))  # m/s

    # Convert to nm/ps
    velocities_nm_ps = velocities_m_s * 1e-3

    # Set velocities
    ps.velocity[:] = velocities_nm_ps

    # Remove center-of-mass velocity
    v_cm = np.average(ps.velocity, axis=0, weights=ps.mass)
    ps.velocity -= v_cm
    

#--------------------------------------
# Energies
#--------------------------------------

def calculate_force_and_energy(ps: ParticleSystem, sim: SimulationParameters):
    n_particles = ps.n
    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    L = sim.box_length

    rij_matrix = ps.position[:, np.newaxis, :] - ps.position[np.newaxis, :, :]
    rij_matrix -= L * np.rint(rij_matrix / L)
    r_matrix = np.linalg.norm(rij_matrix, axis=-1)

    i_upper = np.triu_indices(n_particles, k=1)
    rij = rij_matrix[i_upper]
    r = np.clip(r_matrix[i_upper], sim.rij_min, None)

    sr6 = (sigma / r)**6
    E_pot = np.sum(4 * epsilon * (sr6**2 - sr6))
    dV_dr = 24 * epsilon / r * (-2 * sr6**2 + sr6)
    f_ij = (dV_dr[:, np.newaxis] / r[:, np.newaxis]) * rij   # force from j on i

    force = np.zeros((n_particles, 3))
    i_idx, j_idx = i_upper
    for k in range(3):
        force[:, k] += np.bincount(i_idx, weights=-f_ij[:, k], minlength=n_particles)
        force[:, k] += np.bincount(j_idx, weights= f_ij[:, k], minlength=n_particles)
    
    ps.force = force

    return E_pot

def kinetic_energy(ps: ParticleSystem) -> float:
    """
   Computes the total kinetic energy of the system in units of kJ/mol.

    Assumes:
    - Mass is in u = 1e-3 g/mol
    - Velocity is in nm/ps = 1e3 m/s

    Returns:
        Kinetic energy in kJ/mol.

    """
    # unit: (1e3 ms/s)^2  = 1e6 m^2/s^2        
    v_squared = np.sum(ps.velocity**2, axis=1)   # shape (N,)    
    # unit: 1e-3 kg/mol * 1e6 m^2/s^2 = 1e3 J/mol = 1 kJ/mol
    return 0.5 * np.sum(ps.mass * v_squared)      

def instantaneous_temperature(ps: ParticleSystem) -> float:
    """
    Computes the instantaneous temperature of the particle system 
    from the total kinetic energy using the equipartition theorem.

    Formula:
        T = (2 * E_kin) / (dof * R)

    Where:
        - E_kin is the total kinetic energy in kJ/mol
        - dof is the number of degrees of freedom
        - R is the gas constant in J/(mol·K)

    Returns:
        Temperature in Kelvin (K).
    """
    # kinetic energy is returned in kJ/mol, convert to J/mol
    E_kin = kinetic_energy(ps)*1e3
    # degrees of freedom: 3 per particle
    dof = ps.n*3
        
    return (2* E_kin) / (dof *R)


def density(ps: ParticleSystem, sim: SimulationParameters) -> float: 
    """
    Computes the density of the system in g/cm^3.

    Assumes:
        - box_length is in nm
        - mass is in atomic mass units (g/mol)

    Returns:
        - Density in g/cm^3
    """
    L_in_nm = sim.box_length
    # nm^3 = 10^{-27} m^3 = 10^{-27} m^3* 1000 L/m^3 = 10^{-24} L
    V_in_cm3 = L_in_nm**3 * 1e-21 
    # Mass is stored in u = g/mol
    # Total mass in g (sum of all molar masses divided by Avogadro)
    m_in_g = np.sum(ps.mass) / Avogadro 

    return m_in_g/V_in_cm3

def ideal_gas_pressure(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """
    Computes the instantaneous ideal gas pressure of the system in Pascals (Pa),
    using the ideal gas law: P = nRT/V.

    Assumes:
    - Positions are in nanometers (nm), volume is converted to m³.
    - Temperature is in Kelvin.
    - Returns pressure in SI units (Pa = J/m^3 = N/m^2).
    """
    L_in_nm = sim.box_length
    V_in_m3 = L_in_nm**3 * 1e-27  # Convert volume to m³
    n_mol = ps.n / Avogadro  # Amount of substance in mol
    T = instantaneous_temperature(ps)  # in Kelvin

    return n_mol * R * T / V_in_m3  # Pressure in Pascals (Pa)
    
#--------------------------------------
# MD integrators
#--------------------------------------



def A_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the A-step (position update) of an MD integration scheme.

    This step updates particle positions using the current velocities:
    r(t + Δt) = r(t) + v(t) * Δt

    Parameters:
        - ps (ParticleSystem): The particle system containing positions and velocities.
        - sim (SimulationParameters): Simulation settings, including the time step.
        - half_step (bool): If True, performs a half step (Δt / 2) instead of a full step.

    Returns:
        None. Updates ps.position in-place.
    """
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt
        
    ps.position = ps.position + ps.velocity * dt
    ps.unwrapped_position = ps.unwrapped_position + ps.velocity * dt
    
    return None    

def B_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the B-step (velocity update) of an MD integration scheme.

    This step updates particle velocities using the current forces:
    v(t + Δt) = v(t) + 1/m * Δt * F(t) 
 
    Parameters:
        - ps (ParticleSystem): The particle system containing positions and velocities.
        - sim (SimulationParameters): Simulation settings, including the time step.
        - half_step (bool): If True, performs a half step (Δt / 2) instead of a full step.

    Returns:
        None. Updates ps.velocity in-place.
    """
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt
        
    # (1/ps.mass)[:, np.newaxis] = explicit reshaping to avoid
    # broadcasting issues when multiplying (N,) with (N,3) elementwise
    # now it is explicit: (N,1) * (N,3)
    ps.velocity = ps.velocity + (1/ps.mass)[:, np.newaxis]* dt * ps.force 
    
    return None    

def O_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the O-step (velocity update) in Langevin dynamics.

    The update integrates the effect of the stochastic (random) and friction forces:
        v ← exp(-ξ Δt) * v + sqrt(RT/m * (1 - exp(-2ξΔt))) * η

    Parameters:
        ps (ParticleSystem): Contains velocities, masses, and random number storage.
        sim (SimulationParameters): Contains xi, temperature, dt, and constants.
        half_step (bool): If True, use half the time step Δt / 2.

    Returns:
        None. Updates ps.velocity in-place.
    """

    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt

    # Draw random numbers from Gaussian normal distribution for stochastic term
    ps.random_number = np.random.normal(size=(ps.n,3))
    
    # dissipation term
    d = np.exp(- sim.xi * dt)

    # fluctuation term
    scalar = sim.temperature * R * (1.0 - np.exp(-2 * sim.xi * dt))
    # mass is stored in units of u ~ g/mol, but needs to be converted to kg/mol
    mass = ps.mass *1e3
    f = np.sqrt(scalar / mass)[:, np.newaxis]  # now shape (N, 1)
    f = np.broadcast_to(f, ps.random_number.shape)  # ensures (N, 3)
 
    ps.velocity = d * ps.velocity + f * ps.random_number 
    
    return None    

def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters):
    """
    Performs a single time step of molecular dynamics in the NVE ensemble
    using the velocity Verlet integrator in BAB form (half-step B, full-step A, half-step B).

    The steps are:
    1. Half-step velocity update (B-step)
    2. Full-step position update (A-step)
    3. Force recalculation based on new positions
    4. Second half-step velocity update (B-step)
    5. Apply periodic boundary conditions

    This corresponds to a time-symmetric, second-order accurate integrator for Newtonian dynamics.

    Parameters:
        - ps (ParticleSystem): The particle system containing positions, velocities, and forces.
        - sim (SimulationParameters): Simulation parameters including time step.

    Returns:
        Potential energy. Updates ps.position, ps.velocity, and ps.force in-place.
    """
    B_step(ps, sim, half_step=True)   # update velocity by a half-step
    A_step(ps, sim, half_step=False)  # update position by a full time step
    E_pot = calculate_force_and_energy(ps, sim)          # udpate force  
    B_step(ps, sim, half_step=True)   # update velocity by a second half-step

    apply_periodic_boundary(ps, sim)
        
    return E_pot      

def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters):
    """
    Performs a single time step of molecular dynamics in the NVT ensemble
    using the BAOAB Langevin integrator.

    The steps are:
    1. Half-step velocity update from force (B)
    2. Half-step position update (A)
    3. Full-step velocity update via Langevin thermostat (O)
    4. Second half-step position update (A)
    5. Force recalculation
    6. Second half-step velocity update from force (B)
    7. Apply periodic boundary conditions

    Parameters:
        ps (ParticleSystem): Particle data including velocity, position, mass, etc.
        sim (SimulationParameters): Simulation control parameters.

    Returns:
        potential energy
    """
    
    if sim.tau_thermostat is None:
        raise ValueError("Thermostat coupling time (tau_thermostat) is not set. Cannot run NVT simulation.")
    
    B_step(ps, sim, half_step=True)   # update velocity by a half-step
    A_step(ps, sim, half_step=True)  # update position by a half-step
    # thermostat
    O_step(ps, sim, half_step=False)  # Full-step velocity update using the Langevin thermostat (friction + noise)
    A_step(ps, sim, half_step=True)  # update position by a half-step
    E_pot = calculate_force_and_energy(ps, sim)          # udpate force  
    B_step(ps, sim, half_step=True)   # update velocity by a second half-step

    apply_periodic_boundary(ps, sim)
        
    return E_pot 

def apply_periodic_boundary(ps: ParticleSystem, sim: SimulationParameters): 
    """
    Applies periodic boundary conditions to all particle positions.
    Wraps positions into the interval (-L/2, L/2] using centered PBC.
    """
    L = sim.box_length
    # modulus
    # x < L: x/L = -1*L + remainder => return remainder => shifts x by L to the right
    # x in[ 0, L[ : x/L = 0*L + remainder => return remainder => leaves x where it is
    # x >= L : x/L = 1*L + remainder => return remainder => shifts x by L to the left
    ps.position = np.mod(ps.position, L)
    
def steepest_descent_step(ps, sim, fmax_threshold=100.0, max_steps=1000):
    """
    Performs steepest descent energy minimization until the max force
    drops below fmax_threshold (or max_steps is reached).

    Intended as a quick "de-clash" phase before handing off to FIRE,
    not as a full minimizer.

    Parameters:
        ps (ParticleSystem): Particle data including velocity, position, mass, etc.
        sim (SimulationParameters): Simulation control parameters.
        fmax_threshold (float): Stop once max force drops below this value.
        max_steps (int): Safety cap on number of iterations.

    Returns:
        E (np.ndarray): Array of shape (n_steps_taken, 2) with (step, energy).
    """
    energy = calculate_force_and_energy(ps, sim)
    forces = ps.force
    fmax = np.max(np.linalg.norm(forces, axis=1))

    E_list = []
    step = 0

    while fmax > fmax_threshold and step < max_steps:
        if fmax < 1e-12:
            break

        # stable step size
        eta = sim.sd_eta / (fmax + 1e-12)
        eta = min(eta, 0.01)

        # position update
        ps.position += eta * forces
        ps.position %= sim.box_length

        E_list.append((step, energy))

        energy = calculate_force_and_energy(ps, sim)
        forces = ps.force
        fmax = np.max(np.linalg.norm(forces, axis=1))
        step += 1

    print(f"SD finished after {step} steps, fmax = {fmax:.3e}")
    return np.array(E_list)


#--------------------------------------
# Output
#--------------------------------------
def write_xyz_trajectory(filename, trajectory, atom_symbol="Ar"):
    """
    Writes a trajectory to an .xyz file.

    Parameters:
        filename (str): Name of the output .xyz file.
        trajectory (np.ndarray): Array of shape (n_frames, n_particles, 3)
                                 containing atomic positions.
        atom_symbol (str): Element symbol to use for all atoms (default: "Ar").

    Returns:
        None. Writes file to disk.
    """
    
    trajectory = 10.0 * trajectory  # convert nm to Å
    n_frames, n_atoms, _ = trajectory.shape

    with open(filename, "w") as f:
        for frame in trajectory:
            f.write(f"{n_atoms}\n")
            f.write("Generated by write_xyz_trajectory\n")
            for pos in frame:
                f.write(f"{atom_symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
 
def fire_minimize(ps, sim, 
                  dt_init=0.002, dt_max=0.02,
                  alpha_start=0.5, 
                  n_min=5, 
                  f_tol=1e-4, 
                  max_steps=5000):

    dt = dt_init
    alpha = alpha_start
    n_pos = 0

    # Zero velocities
    ps.velocity[:] = 0.0

    # Allocate energy array
    E = np.zeros((max_steps, 2))

    # Initial force
    energy = calculate_force_and_energy(ps, sim)

    converged = False

    for step in range(max_steps):

        forces = ps.force
        v = ps.velocity

        # Compute potential energy
        

        # Log energy
        E[step, 0] = step
        E[step, 1] = energy

        # Check convergence
        fmax = np.max(np.linalg.norm(forces, axis=1))
        if fmax < f_tol:
            print(f"FIRE converged in {step} steps, fmax = {fmax:.3e}")
            converged = True
            break

        # Gradient descent velocity update
        v += forces * dt / ps.mass[:, None]

        # Power P = v·F
        P = np.sum(v * forces)

        if P > 0:
            n_pos += 1

            Fnorm = np.linalg.norm(forces)
            Vnorm = np.linalg.norm(v)

            # mix velocity using the CURRENT alpha, before decaying it
            if Fnorm > 0 and Vnorm > 0:
                v[:] = (1 - alpha) * v + alpha * (forces / Fnorm) * Vnorm

            if n_pos > n_min:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99

        else:
            n_pos = 0
            dt *= 0.5
            alpha = alpha_start
            v[:] = 0.0

        # Update positions
        ps.position += v * dt

        # Apply PBC
        apply_periodic_boundary(ps, sim)

        # Recompute forces
        energy = calculate_force_and_energy(ps, sim)
        if step % 50 == 0:
            maxF = np.max(np.linalg.norm(ps.force, axis=1))
            print("step", step, "max|F| =", maxF)

    else:
        # loop completed without break -> did not converge
        print(f"FIRE did NOT converge within {max_steps} steps, fmax = {fmax:.3e}")

    # Plot energy curve once, after the loop
    n_recorded = step + 1
    plt.figure()
    plt.plot(E[:n_recorded, 0], E[:n_recorded, 1])
    plt.xlabel("Iteration")
    plt.ylabel("Potential Energy")
    plt.title("FIRE Energy Minimization")
    plt.grid(True)
    
    plt.savefig("fire_minimization.png", dpi=300, bbox_inches='tight')
    plt.close()

    return converged


#--------------------------------------
# Helper Functions / Additional Calculations
#--------------------------------------

def calculate_msd(ps: ParticleSystem, ref_position: np.ndarray) -> float:
    """
    Computes the mean squared displacement relative to reference unwrapped positions
    
    MSD(t) = (1/N) * sum_i | r_i^unwrapped(t) - r_i^unwrapped(0) |^2
    """
    dr = ps.unwrapped_position - ref_position
    squared_displacement = np.sum(dr**2, axis=1)
    return np.mean(squared_displacement)

def calculate_diffusion_coefficient(msd_time: list, msd_trajectory: list, fit_start_ratio=0.25):
    """
    Calculates the self-diffusion coefficient D using Einstein's relation in 3D:
        MSD(t) = 6 * D * t  =>  D = slope / 6
        
    Parameters:
        msd_time (list or np.ndarray): Array of relative times in ps.
        msd_trajectory (list or np.ndarray): Array of MSD values in nm^2.
        fit_start_ratio (float): Fractional index where linear regime starts (default: 0.25 to skip ballistic phase).
        
    Returns:
        D_nm2_ps (float): Diffusion coefficient in nm^2/ps.
        D_cm2_s (float): Diffusion coefficient in cm^2/s.
        slope (float): Slope of the fit line (nm^2/ps).
        intercept (float): Intercept of the fit line.
    """
    t_arr = np.array(msd_time)
    msd_arr = np.array(msd_trajectory)
    
    # Skip the short ballistic regime (e.g. initial 25% of data)
    start_idx = int(len(t_arr) * fit_start_ratio)
    
    # Perform linear fit: msd = slope * t + intercept
    slope, intercept = np.polyfit(t_arr[start_idx:], msd_arr[start_idx:], 1)
    
    # D in 3D: D = slope / 6
    D_nm2_ps = slope / 6.0
    
    # Unit conversion: 1 nm^2/ps = 10^-18 m^2 / 10^-12 s = 10^-6 m^2/s = 10^-2 cm^2/s
    # Wait: 1 nm^2/ps = (10^-7 cm)^2 / 10^-12 s = 10^-14 cm^2 / 10^-12 s = 10^-2 cm^2/s ... Wait, 1 nm = 10^-7 cm => (10^-7)^2 = 10^-14 cm^2.
    # 10^-14 / 10^-12 = 10^-2 cm^2/s (or 10^-4 m^2/s).
    D_cm2_s = D_nm2_ps * 1e-2
    
    return D_nm2_ps, D_cm2_s, slope, intercept