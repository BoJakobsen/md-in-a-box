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
from md_sim import MDSimulation
import numpy as np

# --- Configuration ---
NX, NY = 8, 8
ATOM_SIZE = 8
BOX_X, BOX_Y = 320, 240
R_NBPARTICLES = 0.5
R_SIZEBPARTICLES = 1.5

WARMUP_TEMPERATURE = 1
WARMUP_STEPS = 50  # Each step is stride (20) MD steps

SIM_TEMPERATURE = 0.1
SIM_STEPS = 50
NPLOT = 1           # Update plot every N stride steps (0 to disable)

# Moving box: constant box frame velocity (set both to 0.0 to disable)
BOX_VEL_X = 0.0   # Box velocity in x (pixels/step)
BOX_VEL_Y = 0.0   # Box velocity in y (pixels/step)

# --- Setup ---
sim = MDSimulation(
    n_atoms_x=NX, n_atoms_y=NY,
    atom_size=ATOM_SIZE,
    box_size_x=BOX_X, box_size_y=BOX_Y,
    temperature=SIM_TEMPERATURE,
    r_NBparticles=R_NBPARTICLES
)

print(f"N={sim.n_atoms} atoms ({NX}x{NY}), box={BOX_X}x{BOX_Y}")
sim.set_box_vel([BOX_VEL_X, BOX_VEL_Y])
sim.set_r_sizeBparticles(R_SIZEBPARTICLES)

# Try to import matplotlib for plotting
_can_plot = False
if NPLOT > 0:
    try:
        import matplotlib.pyplot as plt
        _can_plot = True
    except ImportError:
        print("matplotlib not available - skipping plots")

# --- Warmup ---
print(f"Warmup: {WARMUP_STEPS} step() calls "
      f"= {WARMUP_STEPS * sim.stride} MD steps ...")
sim.set_target_temperature(WARMUP_TEMPERATURE)
for _ in range(WARMUP_STEPS):
    sim.step()
print(f"  T after warmup: {sim.get_temperature():.3f}")

# --- Set up live plot ---
if _can_plot:
    nA = sim.n_atomsA
    sA = np.pi*sim.atom_sizeA ** 2  # marker area scales with radius^2
    sB = np.pi*sim.atom_sizeB ** 2

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    scatterA = ax.scatter(sim.pos_x[:nA], sim.pos_y[:nA], s=sA,
                          c="steelblue", edgecolors="k", linewidths=0.5,
                          label="A")
    scatterB = ax.scatter(sim.pos_x[nA:], sim.pos_y[nA:], s=sB,
                          c="tomato", edgecolors="k", linewidths=0.5,
                          label="B")
    ax.set_xlim(0, BOX_X)
    ax.set_ylim(BOX_Y, 0)  # inverted y to match display coordinates
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    if sim.n_atomsB > 0:
        ax.legend(loc="upper right")
    title = ax.set_title("")
    fig.tight_layout()
    fig.show()

# --- Simulation ---
print(f"Running {SIM_STEPS} step() calls "
      f"= {SIM_STEPS * sim.stride} MD steps ...")
sim.set_target_temperature(SIM_TEMPERATURE)
t0 = time.time()

for i in range(1, SIM_STEPS + 1):
    sim.step()
    if _can_plot and i % NPLOT == 0:
        nA = sim.n_atomsA
        scatterA.set_offsets(np.column_stack((sim.pos_x[:nA], sim.pos_y[:nA])))
        scatterB.set_offsets(np.column_stack((sim.pos_x[nA:], sim.pos_y[nA:])))
        T = sim.get_temperature()
        title.set_text(f"step {sim.Nsteps}, T={T:.3f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

elapsed = time.time() - t0
total_md = SIM_STEPS * sim.stride
print(f"\nPerformance: {total_md / elapsed:.0f} MD steps/sec"
      f"  (stride={sim.stride}, {SIM_STEPS / elapsed:.1f} step() calls/sec)")
print(f"Final T={sim.get_temperature():.3f}  (target {sim.target_temp})")

if _can_plot:
    plt.ioff()
    plt.show()
