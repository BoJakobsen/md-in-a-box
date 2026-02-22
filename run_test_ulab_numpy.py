# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
MD Simulation benchmark script for Pico 2 (no display)
Pure ulab/numpy version (no C optimization) for speed comparison.
Tests multiple atom configurations (4x4 to 8x8) with Pico Display dimensions.

Runs on:
  - MicroPython + ulab (Pico or Unix port)
  - CPython + numpy

Run with: mpremote run run_test_ulab_numpy.py
      or: ./micropython-2 run_test_ulab_numpy.py
      or: python3 run_test_ulab_numpy.py
"""
import time
from md_sim_ulab_numpy import MDSimulation

# Timing helper - works on both MicroPython and standard Python
_use_ticks = hasattr(time, 'ticks_ms')


def get_time_ms():
    if _use_ticks:
        return time.ticks_ms()
    else:
        return time.time() * 1000


def get_elapsed_sec(start_ms):
    if _use_ticks:
        return time.ticks_diff(time.ticks_ms(), start_ms) / 1000
    else:
        return (time.time() * 1000 - start_ms) / 1000

# Pico Display 2.0 dimensions
BOX_X = 320
BOX_Y = 240

# Test configurations (matching run_test_C_pico.py)
CONFIGS = [
    {"nx": 4, "ny": 4, "as": 10},
    {"nx": 5, "ny": 5, "as": 10},
    {"nx": 6, "ny": 6, "as": 10},
    {"nx": 8, "ny": 8, "as" : 10},
#    {"nx": 10, "ny": 10, "as": 10},
#    {"nx": 10, "ny": 10, "as": 8},
]

# Note: numpy version has no stride, each step() is 1 MD step
# Use fewer steps than C version since it's much slower
WARMUP_STEPS = 10
BENCH_STEPS = 500

print("=" * 60)
print("MD Simulation Benchmark (Pico 2)")
print("=" * 60)
print(f"Box size: {BOX_X} x {BOX_Y} pixels")
print("=" * 60)
print("micropython + ulab (numpy vectorized, no C)")
print("=" * 60)

for config in CONFIGS:
    nx, ny, asiz = config["nx"], config["ny"], config["as"]
    n_atoms = nx * ny

    sim = MDSimulation(
        n_atoms_x=nx,
        n_atoms_y=ny,
        atom_size=asiz,
        box_size_x=BOX_X,
        box_size_y=BOX_Y,
        temperature=1.0
    )

    print(f"\n{nx}x{ny} = {n_atoms} atoms")
    print(f"Atom size: {asiz} pixels")
    print(f"Warmup: {WARMUP_STEPS} steps, Benchmark: {BENCH_STEPS} steps")
    print("-" * 40)

    # Warmup
    for _ in range(WARMUP_STEPS):
        sim.step()

    # Benchmark
    start = get_time_ms()
    for _ in range(BENCH_STEPS):
        sim.step()
    elapsed = get_elapsed_sec(start)

    steps_per_sec = BENCH_STEPS / elapsed
    ms_per_step = elapsed / BENCH_STEPS * 1000
    final_temp = sim.get_temperature()
    target_temp = sim.get_target_temperature()

    print(f"  Performance: {steps_per_sec:8.1f} MD steps/sec")
    print(f"  Time/step:   {ms_per_step:8.3f} ms")
    print(f"  Final T:     {final_temp:8.3f} (target: {target_temp:.1f})")

print("\n" + "=" * 60)
print("Benchmark complete")
print("=" * 60)
