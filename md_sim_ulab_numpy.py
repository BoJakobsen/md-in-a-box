# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)
# Based on the MD simulation concept by Ulf R. Pedersen (urp.dk/md)

"""
2D Molecular Dynamics Simulation for MicroPython with ulab optimization

Lennard-Jones particles with Leap Frog integration.
Soft wall boundaries (1/r^6 repulsion) and Langevin thermostat.
O(N^2) force calculation (all pairs).

Optimized for Raspberry Pi Pico 2 using ulab (NumPy subset for MicroPython).
Uses separate 1D arrays for positions/velocities/forces (faster than 2D).
Wall forces are vectorized, LJ pair forces use Python loop.

Runs on both:
  - MicroPython with ulab (pico and unix port)
  - Standard Python with numpy (desktop testing)

Reference: Ulf R. Pedersen's JavaScript MD simulation (urp.dk/md)
"""

import math

_using_ulab = False
try:
    from ulab import numpy as np
    _using_ulab = True
except ImportError:
    import numpy as np


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
                 box_size_y=600, temperature=1.0, seed=None):
        """
        Initialize MD simulation.

        Args:
            n_atoms_x: Number of atoms in x direction (grid)
            n_atoms_y: Number of atoms in y direction (grid)
            atom_size: Particle radius in pixels (sigma = 2*atom_size)
            box_size_x: Box width in pixels
            box_size_y: Box height in pixels
            temperature: Target temperature for Langevin thermostat
            seed: Random seed (default 12345)
        """
        self.t = 0.0
        self.Nsteps = 0
        self.n_atoms_x = n_atoms_x
        self.n_atoms_y = n_atoms_y
        self.n_atoms = n_atoms_x * n_atoms_y
        self.atom_size = atom_size
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.target_temp = temperature

        # Random number generator (different API for ulab vs numpy)
        if seed is None:
            seed = 12345
        if _using_ulab:
            self.rng = np.random.Generator(seed)
            # First random numbers from ulab are off - discard them
            self.rng.random()
            self.rng.normal()
        else:
            self.rng = np.random.default_rng(seed)
        
        # Lennard-Jones parameters (pixel-based units)
        self.sigma = 2 * self.atom_size  # Particle diameter
        self.epsilon = 1.0  # LJ energy scale
        self.cutoff = 3.5 * self.atom_size
        self.cutoff_sq = self.cutoff * self.cutoff

        # Pre-computed LJ force constants
        self.sigma2 = self.sigma ** 2
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma ** 12
        self.lj_c12 = -48.0 * self.epsilon * self.sigma12  # Repulsive
        self.lj_c6 = 24.0 * self.epsilon * self.sigma6     # Attractive

        self.epsilon_wall = 10  # Wall softness parameter

        # Integrator timestep
        self.dt = 0.005 * self.atom_size

        # Langevin thermostat
        self.langevin_friction = 0.01
        self._update_thermostat()

        # Gravity
        self.gravity_x = 0.0
        self.gravity_y = 0.005

        # Particle data as separate 1D arrays (faster than 2D indexing in ulab)
        _dtype = np.float64 if hasattr(np, 'float64') else np.float
        self._dtype = _dtype
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
        """Calculate instantaneous temperature from kinetic energy.

        Uses Ulf's convention: T = sum(v^2) / N
        """
        v_sq = np.sum(self.vel_x * self.vel_x) + np.sum(self.vel_y * self.vel_y)
        return v_sq / self.n_atoms

    def _update_thermostat(self):
        """Pre-calculate thermostat standard deviation."""
        self.std = math.sqrt(self.langevin_friction * self.target_temp / self.dt)

    def _calculate_forces(self):
        """Calculate LJ pair forces and soft wall forces.

        Wall forces use 1/r^6 repulsion from each boundary (vectorized).
        LJ forces computed for all pairs O(N^2) with Python loop.
        """
        px = self.pos_x
        py = self.pos_y
        fx = self.force_x
        fy = self.force_y
        eps_w = self.epsilon_wall
        box_x = self.box_size_x
        box_y = self.box_size_y

        # Reset forces
        fx[:] = 0.0
        fy[:] = 0.0

        # Soft wall forces (1/x^6) - vectorized
        # Left wall (x -> 0)
        tmp = px * px
        tmp = tmp * tmp * tmp  # x^6
        fx += eps_w / tmp

        # Right wall (x -> box_x)
        tmp = (px - box_x)
        tmp = tmp * tmp
        tmp = tmp * tmp * tmp
        fx -= eps_w / tmp

        # Top wall (y -> 0)
        tmp = py * py
        tmp = tmp * tmp * tmp
        fy += eps_w / tmp

        # Bottom wall (y -> box_y)
        tmp = (py - box_y)
        tmp = tmp * tmp
        tmp = tmp * tmp * tmp
        fy -= eps_w / tmp
        
        # LJ pair forces - O(N^2) loop over all pairs
        cutoff_sq = self.cutoff_sq
        c12 = self.lj_c12
        c6 = self.lj_c6
        n_atoms = self.n_atoms


        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                r2 = dx * dx + dy * dy

                if r2 < cutoff_sq and r2 > 0.01:
                    r6 = r2 * r2 * r2
                    r12 = r6 * r6
                    pre = (c12 / r12 + c6 / r6) / r2

                    f_x = pre * dx
                    f_y = pre * dy

                    fx[i] += f_x
                    fy[i] += f_y
                    fx[j] -= f_x
                    fy[j] -= f_y

        # Apply Langevin thermostat: friction + random forces
        n = self.n_atoms
        std = self.std
        friction = self.langevin_friction

        # Approximate Gaussian from sum of uniforms (faster than rng.normal)
        rand_x = 2 * (self.rng.random(size=(1, n)) + self.rng.random(size=(1, n)) + self.rng.random(size=(1, n)) - 1.5).flatten()
        rand_y = 2 * (self.rng.random(size=(1, n)) + self.rng.random(size=(1, n)) + self.rng.random(size=(1, n)) - 1.5).flatten()

        fx += std * rand_x - friction * self.vel_x
        fy += std * rand_y - friction * self.vel_y

        # Apply gravity
        fx += self.gravity_x
        fy += self.gravity_y

    def step(self):
        """Perform one Leap Frog integration step.

        Leap Frog scheme:
          1. x(t+dt) = x(t) + v(t+dt/2) * dt
          2. F(t+dt) = forces at new positions
          3. v(t+3dt/2) = v(t+dt/2) + F(t+dt) * dt
        """
        self.t += self.dt
        self.Nsteps += 1
        dt = self.dt
        
        # Step 1: Update positions
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

        
        # Step 2: Calculate forces (includes thermostat and gravity)
        self._calculate_forces()

        # Step 3: Update velocities
        self.vel_x += self.force_x * dt
        self.vel_y += self.force_y * dt

    def get_temperature(self):
        """Return current instantaneous temperature."""
        return self._calculate_temperature()

    def get_time_and_steps(self):
        """Return (simulation_time, step_count)."""
        return self.t, self.Nsteps

    def set_gravity(self, gx, gy):
        """Set gravity acceleration."""
        self.gravity_x = gx
        self.gravity_y = gy

    def set_target_temperature(self, temp):
        """Set target temperature for Langevin thermostat."""
        self.target_temp = temp
        self._update_thermostat()

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


# Simple test when run directly
if __name__ == "__main__":
    import time

    print("Starting ulab/numpy MD simulation (no C optimization)...")

    sim = MDSimulation(
        n_atoms_x=5,
        n_atoms_y=5,
        atom_size=10,
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
    
    # Use ticks_ms if available (MicroPython), else time.time()
    if hasattr(time, 'ticks_ms'):
        start_time = time.ticks_ms()
    else:
        start_time = time.time()

    n_steps = 500
    print_every = 50
    
    for step in range(n_steps):
        sim.step()
        if step % print_every == 0:
            temp = sim.get_temperature()
            sim_time, _ = sim.get_time_and_steps()
            print(f"Step {step}: t = {sim_time:.1f}, T={temp:.3f}")

    if hasattr(time, 'ticks_ms'):
        elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
    else:
        elapsed = time.time() - start_time

    steps_per_sec = n_steps / elapsed

    print(f"\nPerformance: {steps_per_sec:.1f} MD steps/second")
    print(f"Final temperature: {sim.get_temperature():.3f}")
    print(f"Target temperature: {sim.target_temp:.3f}")
