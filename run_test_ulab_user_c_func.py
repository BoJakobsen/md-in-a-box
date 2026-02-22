# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
MD Simulation benchmark script for Pico 2 (no display)
ulab + custom C version for speed measurement.
Tests multiple atom configurations (4x4 to 10x10) with Pico Display dimensions.

Run with: mpremote run run_test_ulab_user_c_func.py
"""
import time
from md_sim_ulab_user_c_func import MDSimulation

# Pico Display 2.0 dimensions
BOX_X = 320
BOX_Y = 240

# Test configurations
CONFIGS = [
    {"nx": 4, "ny": 4, "as" : 10},
    {"nx": 5, "ny": 5, "as" : 10},
    {"nx": 6, "ny": 6, "as" : 10},
    {"nx": 8, "ny": 8, "as" : 10},
    {"nx": 10, "ny": 10, "as" : 10},
    {"nx": 10, "ny": 10, "as" : 8},
]

WARMUP_CALLS = 50
BENCH_CALLS = 500

print("=" * 60)
print("MD Simulation Benchmark (Pico 2)")
print("=" * 60)
print(f"Box size: {BOX_X} x {BOX_Y} pixels")
print("=" * 60)
print(f"micropython + ulab + custom C")
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

    stride = sim.stride
    warmup_md_steps = WARMUP_CALLS * stride
    bench_md_steps = BENCH_CALLS * stride

    print(f"\n{nx}x{ny} = {n_atoms} atoms (stride={stride})")
    print(f"Atom size: {asiz} pixels")
    print(f"Warmup: {warmup_md_steps} steps, Benchmark: {bench_md_steps} steps")
    print("-" * 40)

    # Warmup
    for _ in range(WARMUP_CALLS):
        sim.step()

    # Benchmark
    start = time.ticks_ms()
    for _ in range(BENCH_CALLS):
        sim.step()
    elapsed = time.ticks_diff(time.ticks_ms(), start) / 1000

    steps_per_sec = bench_md_steps / elapsed
    ms_per_step = elapsed / bench_md_steps * 1000
    final_temp = sim.get_temperature()
    target_temp = sim.get_target_temperature()

    print(f"  Performance: {steps_per_sec:8.1f} MD steps/sec")
    print(f"  Time/step:   {ms_per_step:8.3f} ms")
    print(f"  Final T:     {final_temp:8.3f} (target: {target_temp:.1f})")

print("\n" + "=" * 60)
print("Benchmark complete")
print("=" * 60)
