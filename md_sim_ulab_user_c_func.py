# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)
# Based on the MD simulation concept by Ulf R. Pedersen (urp.dk/md)

"""
2D Molecular Dynamics Simulation for MicroPython with ulab + custom C optimization
C-code is in user.c

Lennard-Jones particles with Leap Frog integration.
Soft wall boundaries (1/r^6 repulsion) and Langevin thermostat.
O(N^2) force calculation (all pairs).

Optimized for Raspberry Pi Pico 2 using ulab (NumPy subset for MicroPython).
Uses separate 1D arrays for positions/velocities/forces (faster than 2D).

Requires: MicroPython with ulab and custom C-functions (user.c)

Reference: Ulf R. Pedersen's JavaScript MD simulation (urp.dk/md)
"""

import math

# Current version needs ulab and user.c (so micropython custom build)
from ulab import numpy as np
from ulab import user

class MDSimulation:
    """
    2D Lennard-Jones molecular dynamics simulation.

    Uses pixel-based units for direct display mapping:
      - sigma = 2 * atom_size (particle diameter)
      - dt = 0.005 * atom_size
      - cutoff = 3.5 * atom_size

    Particle data stored as separate 1D arrays (pos_x, pos_y, vel_x, etc.)
    for optimal performance on MicroPython/ulab.
    """

    def __init__(self, n_atoms_x=8, n_atoms_y=10, atom_size=10, box_size_x=400,
                 box_size_y=600, temperature=1.0):
        """
        Initialize MD simulation.

        Args:
            n_atoms_x: Number of atoms in x direction (grid)
            n_atoms_y: Number of atoms in y direction (grid)
            atom_size: Particle radius in pixels (sigma = 2*atom_size)
            box_size_x: Box width in pixels
            box_size_y: Box height in pixels
            temperature: Target temperature for Langevin thermostat

        Key attributes after init:
            stride: MD steps computed per call to step() (default 20)
            box_vel: [vx, vy] box frame velocity, used by accelerometer integration
"""

        _dtype = np.float64 if hasattr(np, 'float64') else np.float
        self._dtype = _dtype
        
        self.t = 0.0
        self.Nsteps = 0
        self.n_atoms_x = n_atoms_x
        self.n_atoms_y = n_atoms_y
        self.n_atoms = n_atoms_x * n_atoms_y
        self.atom_size = atom_size
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.target_temp = temperature

        # Lennard-Jones parameters (pixel-based units)
        self.sigma = 2 * self.atom_size  # Particle diameter
        self.epsilon = 1.0  # LJ energy scale
        self.cutoff = 3.5 * self.atom_size

        # Pre-computed simulation parameters
        self.sigma2 = self.sigma ** 2
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma ** 12
        self.lj_c12 = -48.0 * self.epsilon * self.sigma12  # Repulsive
        self.lj_c6 = 24.0 * self.epsilon * self.sigma6     # Attractive

        self.cutoff_sq = self.cutoff * self.cutoff

        self.epsilon_wall = 10  # Wall softness parameter
        self.box_vel = np.zeros(2, dtype=_dtype)  # Box velocity [vx, vy] (set by accelerometer)

        # Integrator time step
        self.dt = 0.005 * self.atom_size

        # Number of MD steps computed in C per call to step() (avoids Python overhead)
        self.stride = 20

        # Langevin thermostat
        self.langevin_friction = 0.01
        self._update_thermostat()  # Pre-calc std for thermostat

        # Gravity
        self.gravity_x = 0.0
        self.gravity_y = 0.005

        # Pre-allocate Particle data arrays
        self.pos_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.pos_y = np.zeros(self.n_atoms, dtype=_dtype)
        self.vel_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.vel_y = np.zeros(self.n_atoms, dtype=_dtype)
        self.force_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.force_y = np.zeros(self.n_atoms, dtype=_dtype)

        # Initialize positions on grid
        self._initialize_particles()


    def _initialize_particles(self):
        """Initialize particle positions on a uniform grid, zero velocities."""
        count = 0
        for i in range(self.n_atoms_x):
            for j in range(self.n_atoms_y):
                self.pos_x[count] = (i + 0.5) * self.box_size_x / self.n_atoms_x
                self.pos_y[count] = (j + 0.5) * self.box_size_y / self.n_atoms_y
                self.vel_x[count] = 0.0
                self.vel_y[count] = 0.0
                count += 1

    def _calculate_temperature(self):
        """Calculate instantaneous temperature from kinetic energy."""
        # Mean kinetic energy
        v_sq = 1/2 * (np.sum(self.vel_x * self.vel_x) + np.sum(self.vel_y * self.vel_y))
        # T = V_sq / N for 2D
        return v_sq / self.n_atoms

    def _update_thermostat(self):
        """Pre-calculate thermostat noise standard deviation from friction and temperature."""
        self.std = math.sqrt(2 * self.langevin_friction * self.target_temp / self.dt)
 
    def step(self):
        """Perform one Leap Frog integration step. using C function

        Leap Frog scheme:
          1. x(t+dt) = x(t) + v(t+dt/2) * dt
          2. F(t+dt) = forces at new positions
          3. v(t+3dt/2) = v(t+dt/2) + F(t+dt) * dt
        """

        self.t += self.dt * self.stride
        self.Nsteps += self.stride
 
        # one step includes all forces, inc thermostat
        user.lf_step(
            self.pos_x,            # array, float
            self.pos_y,            # array, float
            self.vel_x,            # array, float
            self.vel_y,            # array, float
            self.force_x,          # array, float (changed in place)
            self.force_y,          # array, float (changed in place)
            self.cutoff_sq,        # float
            self.lj_c12,           # float
            self.lj_c6,            # float
            self.box_size_x,       # int
            self.box_size_y,       # int
            self.epsilon_wall,     # float
            self.box_vel,          # array, float  
            self.dt,               # float
            self.std,              # float
            self.langevin_friction,# float
            self.gravity_x,        # float
            self.gravity_y,        # float
            self.stride            # int
        )

    def get_temperature(self):
        """Return current instantaneous temperature."""
        return self._calculate_temperature()

    def get_time_and_steps(self):
        """Return (simulation_time, step_count)."""
        return self.t, self.Nsteps

    def set_gravity(self, gx, gy):
        """Set gravity acceleration"""
        self.gravity_x = gx
        self.gravity_y = gy

    def set_target_temperature(self, temp):
        """Set target temperature for Langevin thermostat."""
        self.target_temp = temp
        self._update_thermostat()

    def set_box_vel(self, vel):
        """Set box velocity"""
        self.box_vel = np.array(vel, dtype=self._dtype)

    def get_target_temperature(self):
        """Get target temperature for Langevin thermostat."""
        return self.target_temp

    def set_thermostat_coupling(self, friction):
        """Set Langevin friction coefficient (coupling strength)."""
        self.langevin_friction = friction
        self._update_thermostat()

    def get_thermostat_coupling(self):
        """Get Langevin friction coefficient."""
        return self.langevin_friction

    def get_gravity(self):
        """Get gravity acceleration"""
        return self.gravity_x, self.gravity_y

    def get_box_vel(self):
        """Get box velocity"""
        return self.box_vel


# Simple test when run directly
if __name__ == "__main__":
    import time

    print("Starting C-ulab-optimized MD simulation...")

    sim = MDSimulation(
        n_atoms_x=10,
        n_atoms_y=10,
        atom_size=8,
        box_size_x=320,
        box_size_y=240,
        temperature=1.0
    )
    
    print(f"N atoms: {sim.n_atoms} ({sim.n_atoms_x}x{sim.n_atoms_y})")
    print(f"Box size: {sim.box_size_x}x{sim.box_size_y}")
    print(f"Atom size: {sim.atom_size}, sigma: {sim.sigma}")
    print(f"dt: {sim.dt}")
    print(f"Initial temperature: {sim.get_temperature():.3f}")
    print(f"Langevin friction: {sim.langevin_friction:.4f}")

    start_time = time.ticks_ms()
    n_calls = 100  # Each call does sim.stride MD steps
    print_every = 5  # Print every 5 calls (= 100 MD steps)

    for call in range(n_calls):
        sim.step()
        if call % print_every == 0:
            temp = sim.get_temperature()
            sim_time, n_md_steps = sim.get_time_and_steps()
            print(f"Step {n_md_steps}: t = {sim_time:.1f}, T={temp:.3f}")

    elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
    total_md_steps = n_calls * sim.stride
    steps_per_sec = total_md_steps / elapsed

    print(f"\nPerformance: {steps_per_sec:.1f} MD steps/second (stride={sim.stride})")
    print(f"Final temperature: {sim.get_temperature():.3f}")
    print(f"Target temperature: {sim.target_temp:.3f}")
