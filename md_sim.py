# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)
# Based on the MD simulation concept by Ulf R. Pedersen (urp.dk/md)

"""
Unified 2D Binary Lennard-Jones MD Simulation

Auto-detects runtime and selects the fastest available backend:

  BACKEND = "ulab_c"   MicroPython + ulab + user.lf_step C extension  (Pico, main)
  BACKEND = "ulab"     MicroPython + ulab, pure Python physics         (Pico, no C build)
  BACKEND = "cpython"  CPython + NumPy, desktop testing

The API is identical across all three backends.

Reference: Ulf R. Pedersen's JavaScript MD simulation (urp.dk/md)
"""

import math

_user_c = None
try:
    from ulab import numpy as np
    BACKEND = "ulab"
    try:
        from ulab import user as _user_c
        BACKEND = "ulab_c"
    except ImportError:
        pass
except ImportError:
    import numpy as np
    BACKEND = "cpython"


class MDSimulation:
    """
    2D binary Lennard-Jones molecular dynamics simulation.

    Two atom types: A occupies indices [0..n_atomsA-1],
      B occupies [n_atomsA..n_atoms-1].

    Three LJ interaction sets stored as 3-element arrays (index 0=AA, 1=BB, 2=AB):
      lj_c12, lj_c6  (pre-computed force constants).
      epsilon_AA = epsilon_BB = 1 (fixed); epsilon_AB = epsilonAB (tunable).

    Uses pixel-based units for direct display mapping:
      sigma_AA = 2 * atom_sizeA,  sigma_BB = 2 * atom_sizeB
      sigma_AB = atom_sizeA + atom_sizeB  (Lorentz-Berthelot mixing rule)
      dt = 0.005 * atom_size (fixed, based on atom_sizeA)
      cutoff = 3.5 * atom_size  (fixed, based on atom_sizeA)
    """

    def __init__(self, n_atoms_x=8, n_atoms_y=8, atom_size=8, box_size_x=320,
                 box_size_y=240, temperature=1.0, r_NBparticles=0.5, seed=None):
        """
        Initialize MD simulation.

        Args:
            n_atoms_x:     Number of atoms in x direction (grid)
            n_atoms_y:     Number of atoms in y direction (grid)
            atom_size:     Type A particle radius in pixels (sigma_AA = 2*atom_size)
            box_size_x:    Box width in pixels
            box_size_y:    Box height in pixels
            temperature:   Target temperature for Langevin thermostat
            r_NBparticles: Fraction of total atoms that are type B (0=all A, 1=all B)
            seed:          RNG seed (default 12345; ignored for ulab_c which uses C xorshift32)

        Key attributes adjustable after init:
            stride:           MD steps per call to step() (default 20)
            r_sizeBparticles: B radius relative to A (default 1.0)
            epsilonAB:        A-B LJ well depth (default 1.0 = symmetric)
            box_vel:          [vx, vy] box frame velocity (accelerometer integration)
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

        # Gravity
        self.gravity_x = 0.0
        self.gravity_y = 0.005

        # Wall parameters
        self.epsilon_wall = 10
        self.box_vel = np.zeros(2, dtype=_dtype)

        # RNG (not used by ulab_c backend; C uses its own xorshift32)
        if BACKEND != "ulab_c":
            if seed is None:
                seed = 12345
            if BACKEND == "ulab":
                self.rng = np.random.Generator(seed)
                # First values from ulab Generator are off — discard
                self.rng.random()
                self.rng.normal()
            else:
                self.rng = np.random.default_rng(seed)

        # LJ cutoff (based on A-particle size, fixed at init)
        self.cutoff = 3.5 * self.atom_size
        self.cutoff_sq = self.cutoff * self.cutoff

        # Binary LJ tunable parameters (adjustable via setters)
        self.r_NBparticles = r_NBparticles
        self.r_sizeBparticles = 1.0
        self.epsilonAB = 1.0

        # MD step size and stride
        self.dt = 0.005 * self.atom_size
        self.stride = 20

        # Langevin thermostat
        self.langevin_friction = 0.01

        # Compute binary LJ parameters and thermostat std
        self._update_binary_lj()
        self._update_thermostat()

        self.pos_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.pos_y = np.zeros(self.n_atoms, dtype=_dtype)
        self.vel_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.vel_y = np.zeros(self.n_atoms, dtype=_dtype)
        self.force_x = np.zeros(self.n_atoms, dtype=_dtype)
        self.force_y = np.zeros(self.n_atoms, dtype=_dtype)

        self._initialize_particles()

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _update_binary_lj(self):
        """Recompute all binary LJ parameters from r_NBparticles,
             r_sizeBparticles, epsilonAB.

        Updates n_atomsA/B, atom sizes, sigma [AA, BB, AB], epsilon [1, 1, epsilonAB],
        and the pre-computed force constants lj_c12, lj_c6 (3-element arrays).
        Called by __init__ and all set_r_* / set_epsilonAB setters.
        """
        self.n_atomsB = int(self.r_NBparticles * self.n_atoms)
        self.n_atomsA = self.n_atoms - self.n_atomsB
        self.atom_sizeA = self.atom_size
        # Keep as float so callers can cast for display as needed
        self.atom_sizeB = self.atom_size * self.r_sizeBparticles
        self.sigma = np.array([2 * self.atom_sizeA, 2 * self.atom_sizeB,
                               self.atom_sizeA + self.atom_sizeB], dtype=self._dtype)
        self.epsilon = np.array([1, 1, self.epsilonAB], dtype=self._dtype)
        self.sigma2 = self.sigma ** 2
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma ** 12
        self.lj_c12 = -48.0 * self.epsilon * self.sigma12  # Repulsive
        self.lj_c6 = 24.0 * self.epsilon * self.sigma6     # Attractive

    def _update_thermostat(self):
        """Pre-calculate thermostat noise standard deviation."""
        self.std = math.sqrt(2 * self.langevin_friction * self.target_temp / self.dt)

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

    # ------------------------------------------------------------------
    # Physics (Python/ulab backends only)
    # ------------------------------------------------------------------

    def _calculate_forces(self):
        """Calculate LJ pair forces and soft wall forces (Python/ulab backends).

        Wall forces: 1/r^6 repulsion from each boundary (vectorized).
        LJ forces: O(N^2) pair loop, honouring epsilonAB via epsilon array.
        Thermostat: approximate Gaussian (sum-of-3-uniforms).
        """
        px = self.pos_x
        py = self.pos_y
        fx = self.force_x
        fy = self.force_y
        eps_w = self.epsilon_wall
        box_x = self.box_size_x
        box_y = self.box_size_y

        fx[:] = 0.0
        fy[:] = 0.0

        # Soft wall forces (1/x^6), vectorized
        tmp = px * px
        tmp = tmp * tmp * tmp
        fx += eps_w / tmp

        tmp = px - box_x
        tmp = tmp * tmp
        tmp = tmp * tmp * tmp
        fx -= eps_w / tmp

        tmp = py * py
        tmp = tmp * tmp * tmp
        fy += eps_w / tmp

        tmp = py - box_y
        tmp = tmp * tmp
        tmp = tmp * tmp * tmp
        fy -= eps_w / tmp

        # LJ pair forces — O(N^2)
        cutoff_sq = self.cutoff_sq

        # AA interactions
        c12 = self.lj_c12[0]
        c6 = self.lj_c6[0]
        for i in range(self.n_atomsA):
            for j in range(i + 1, self.n_atomsA):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                r2 = dx * dx + dy * dy
                if r2 < cutoff_sq and r2 > 0.01:
                    r6 = r2 * r2 * r2
                    pre = (c12 / (r6 * r6) + c6 / r6) / r2
                    f_x = pre * dx
                    f_y = pre * dy
                    fx[i] += f_x
                    fy[i] += f_y
                    fx[j] -= f_x
                    fy[j] -= f_y

        # BB interactions
        c12 = self.lj_c12[1]
        c6 = self.lj_c6[1]
        for i in range(self.n_atomsA, self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                r2 = dx * dx + dy * dy
                if r2 < cutoff_sq and r2 > 0.01:
                    r6 = r2 * r2 * r2
                    pre = (c12 / (r6 * r6) + c6 / r6) / r2
                    f_x = pre * dx
                    f_y = pre * dy
                    fx[i] += f_x
                    fy[i] += f_y
                    fx[j] -= f_x
                    fy[j] -= f_y

        # AB interactions
        c12 = self.lj_c12[2]
        c6 = self.lj_c6[2]
        for i in range(self.n_atomsA):
            for j in range(self.n_atomsA, self.n_atoms):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                r2 = dx * dx + dy * dy
                if r2 < cutoff_sq and r2 > 0.01:
                    r6 = r2 * r2 * r2
                    pre = (c12 / (r6 * r6) + c6 / r6) / r2
                    f_x = pre * dx
                    f_y = pre * dy
                    fx[i] += f_x
                    fy[i] += f_y
                    fx[j] -= f_x
                    fy[j] -= f_y

        # Langevin thermostat: friction + Gaussian noise
        n = self.n_atoms
        std = self.std
        friction = self.langevin_friction

        if BACKEND == "ulab":
            # ulab Generator.random() requires 2D shape — flatten to 1D.
            # Approximate Gaussian via sum of 3 uniforms.
            rand_x = 2 * (self.rng.random(size=(1, n)) + self.rng.random(size=(1, n))
                          + self.rng.random(size=(1, n)) - 1.5).flatten()
            rand_y = 2 * (self.rng.random(size=(1, n)) + self.rng.random(size=(1, n))
                          + self.rng.random(size=(1, n)) - 1.5).flatten()
        else:
            # Approximate Gaussian via sum of 3 uniforms (same as ulab backend)
            rand_x = 2 * (self.rng.random(size=n) + self.rng.random(size=n)
                          + self.rng.random(size=n) - 1.5)
            rand_y = 2 * (self.rng.random(size=n) + self.rng.random(size=n)
                          + self.rng.random(size=n) - 1.5)

        fx += std * rand_x - friction * self.vel_x
        fy += std * rand_y - friction * self.vel_y

        # Gravity
        fx += self.gravity_x
        fy += self.gravity_y

        # Clamp forces to prevent overflow
        max_force = 1.0e6
        fx[:] = np.minimum(np.maximum(fx, -max_force), max_force)
        fy[:] = np.minimum(np.maximum(fy, -max_force), max_force)

    def _py_step(self):
        """Single Leap Frog integration step (Python/ulab backends).

        Leap Frog scheme:
          1. x(t+dt) = x(t) + v(t+dt/2)*dt - box_vel  (moving frame)
          2. Emergency hard wall: clamp positions, reflect velocities
          3. F(t+dt) = forces at new positions (includes thermostat + gravity)
          4. v(t+3dt/2) = v(t+dt/2) + F(t+dt)*dt
        """
        dt = self.dt
        box_vel = self.box_vel

        self.pos_x += self.vel_x * dt - box_vel[0]
        self.pos_y += self.vel_y * dt - box_vel[1]

        # Emergency hard wall
        margin = 1.0
        mask = self.pos_x < margin
        self.pos_x[mask] = margin
        self.vel_x[mask] = np.abs(self.vel_x[mask])
        mask = self.pos_x > self.box_size_x - margin
        self.pos_x[mask] = self.box_size_x - margin
        self.vel_x[mask] = -np.abs(self.vel_x[mask])
        mask = self.pos_y < margin
        self.pos_y[mask] = margin
        self.vel_y[mask] = np.abs(self.vel_y[mask])
        mask = self.pos_y > self.box_size_y - margin
        self.pos_y[mask] = self.box_size_y - margin
        self.vel_y[mask] = -np.abs(self.vel_y[mask])

        self._calculate_forces()

        self.vel_x += self.force_x * dt
        self.vel_y += self.force_y * dt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by stride MD steps.

        ulab_c: delegates to user.lf_step (C extension, stride steps in one call).
        ulab/cpython: Python loop calling _py_step() stride times.
        """
        if BACKEND == "ulab_c":
            self.t += self.dt * self.stride
            self.Nsteps += self.stride
            _user_c.lf_step(
                self.pos_x,
                self.pos_y,
                self.vel_x,
                self.vel_y,
                self.force_x,
                self.force_y,
                self.cutoff_sq,
                self.lj_c12,
                self.lj_c6,
                self.box_size_x,
                self.box_size_y,
                self.epsilon_wall,
                self.box_vel,
                self.dt,
                self.std,
                self.langevin_friction,
                self.gravity_x,
                self.gravity_y,
                self.n_atomsA,
                self.stride,
            )
        else:
            for _ in range(self.stride):
                self.t += self.dt
                self.Nsteps += 1
                self._py_step()

    def get_temperature(self):
        """Return instantaneous temperature from kinetic energy."""
        v_sq = 0.5 * (np.sum(self.vel_x * self.vel_x) +
                      np.sum(self.vel_y * self.vel_y))
        return v_sq / self.n_atoms

    def get_target_temperature(self):
        """Return target temperature."""
        return self.target_temp

    def get_time_and_steps(self):
        """Return (simulation_time, step_count)."""
        return self.t, self.Nsteps

    def get_gravity(self):
        """Return (gravity_x, gravity_y)."""
        return self.gravity_x, self.gravity_y

    def get_box_vel(self):
        """Return box velocity array [vx, vy]."""
        return self.box_vel

    def set_target_temperature(self, temp):
        """Set thermostat target temperature."""
        if temp >= 0:
            self.target_temp = temp
            self._update_thermostat()

    def set_thermostat_coupling(self, friction):
        """Set Langevin friction coefficient."""
        if friction >= 0:
            self.langevin_friction = friction
            self._update_thermostat()

    def set_gravity(self, gx, gy):
        """Set gravity acceleration (gx, gy)."""
        self.gravity_x = gx
        self.gravity_y = gy

    def set_box_vel(self, vel):
        """Set box velocity [vx, vy] for moving frame."""
        if len(vel) == 2:
            self.box_vel = np.array(vel, dtype=self._dtype)

    def set_r_NBparticles(self, r_NBparticles):
        """Set fraction of atoms that are type B.
           0 <= r_NBparticles <= 1
        """
        if 0 <= r_NBparticles <= 1:
            self.r_NBparticles = r_NBparticles
            self._update_binary_lj()

    def set_r_sizeBparticles(self, r_sizeBparticles):
        """Set B atom radius relative to A (0 < r <= 5)."""
        if 0 < r_sizeBparticles <= 5:
            self.r_sizeBparticles = r_sizeBparticles
            self._update_binary_lj()

    def set_epsilonAB(self, epsilonAB):
        """Set A-B LJ well depth (>= 0). Low values cause A/B phase separation."""
        if epsilonAB >= 0:
            self.epsilonAB = epsilonAB
            self._update_binary_lj()

# ------------------------------------------------------------------
# Self-test / benchmark when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("BACKEND =", BACKEND)

    sim = MDSimulation(
        n_atoms_x=10,
        n_atoms_y=10,
        atom_size=8,
        box_size_x=320,
        box_size_y=240,
        temperature=1.0,
        r_NBparticles=0.5,
    )

    print("N atoms:", sim.n_atoms, "(", sim.n_atoms_x, "x", sim.n_atoms_y, ")")
    print("n_atomsA:", sim.n_atomsA, "  n_atomsB:", sim.n_atomsB)
    print("atom_sizeA:", sim.atom_sizeA, "  atom_sizeB:", sim.atom_sizeB)
    print("stride:", sim.stride)
    print("dt:", sim.dt)
    print("Initial T:", sim.get_temperature())

    if hasattr(time, 'ticks_ms'):
        t0 = time.ticks_ms()
    else:
        t0 = time.time()

    n_calls = 100
    print_every = 10

    for call in range(n_calls):
        sim.step()
        if call % print_every == 0:
            _, n_md = sim.get_time_and_steps()
            print("step", n_md, "T =", sim.get_temperature())

    if hasattr(time, 'ticks_ms'):
        elapsed = time.ticks_diff(time.ticks_ms(), t0) / 1000
    else:
        elapsed = time.time() - t0

    total_md = n_calls * sim.stride
    print("\nPerformance:", total_md / elapsed, "MD steps/sec  (stride =", sim.stride, ")")
    print("Final T:", sim.get_temperature(), "  target:", sim.target_temp)
