# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
Desktop test and visualization for the MD simulation (numpy version).
Runs a warmup phase, then the simulation with an optional live plot.

Requires: numpy (always), matplotlib (optional, for live plot)

Usage:
    python3 run_show_sim_numpy_pc.py
"""
import time
from md_sim_ulab_numpy import MDSimulation

import numpy as np

# --- Configuration ---
NX, NY = 8, 8
ATOM_SIZE = 10
BOX_X, BOX_Y = 320, 240
TEMPERATURE = 1.0

WARMUP_STEPS = 2000
SIM_STEPS = 5000
NPLOT = 20           # Update plot every N steps (0 to disable)

# --- Setup ---
sim = MDSimulation(
    n_atoms_x=NX, n_atoms_y=NY,
    atom_size=ATOM_SIZE,
    box_size_x=BOX_X, box_size_y=BOX_Y,
    temperature=TEMPERATURE,
)

print(f"N={sim.n_atoms} atoms ({NX}x{NY}), box={BOX_X}x{BOX_Y}, T_target={TEMPERATURE}")

# Try to import matplotlib for plotting
_can_plot = False
if NPLOT > 0:
    try:
        import matplotlib.pyplot as plt
        _can_plot = True
    except ImportError:
        print("matplotlib not available - skipping plots")

# --- Warmup ---
print(f"Warmup: {WARMUP_STEPS} steps ...")
for _ in range(WARMUP_STEPS):
    sim.step()
print(f"  T after warmup: {sim.get_temperature():.3f}")

# --- Set up live plot ---
if _can_plot:
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    scatter = ax.scatter(sim.pos_x, sim.pos_y, s=ATOM_SIZE**2,
                         c="steelblue", edgecolors="k", linewidths=0.5)
    ax.set_xlim(0, BOX_X)
    ax.set_ylim(BOX_Y, 0)  # inverted y to match display coordinates
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    title = ax.set_title("")
    fig.tight_layout()
    fig.show()

# --- Simulation ---
print(f"Running {SIM_STEPS} steps ...")
t0 = time.time()

for i in range(1, SIM_STEPS + 1):
    sim.step()
    if _can_plot and i % NPLOT == 0:
        scatter.set_offsets(np.column_stack((sim.pos_x, sim.pos_y)))
        T = sim.get_temperature()
        title.set_text(f"step {sim.Nsteps}, T={T:.3f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

elapsed = time.time() - t0
print(f"\nPerformance: {SIM_STEPS / elapsed:.0f} MD steps/sec")
print(f"Final T={sim.get_temperature():.3f}  (target {sim.target_temp})")

if _can_plot:
    plt.ioff()
    plt.show()
