# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
MD Simulation benchmark — using md_sim auto-backend.

Auto-detects and reports backend (ulab_c / ulab / cpython).
Tests multiple atom configurations with Pico Display 2.0 dimensions.
Each step() call runs stride MD steps internally.

Run with: mpremote run run_test_all.py
      or: ./micropython-2 run_test_all.py
      or: python3 run_test_all.py
"""
import time
from md_sim import MDSimulation, BACKEND

# Timing helper — works on both MicroPython and CPython
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

# Test configurations
CONFIGS = [
    {"nx": 4,  "ny": 4,  "as": 10},
    {"nx": 5,  "ny": 5,  "as": 10},
    {"nx": 6,  "ny": 6,  "as": 10},
    {"nx": 8,  "ny": 8,  "as": 10},
    {"nx": 10, "ny": 10, "as": 10},
    {"nx": 10, "ny": 10, "as": 8},
]

WARMUP_CALLS = 50   # step() calls for warmup  = WARMUP_CALLS * stride MD steps
BENCH_CALLS  = 500  # step() calls for benchmark = BENCH_CALLS * stride MD steps

print("=" * 60)
print("MD Simulation Benchmark")
print("=" * 60)
print(f"Backend:  {BACKEND}")
print(f"Box size: {BOX_X} x {BOX_Y} pixels")
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
    warmup_md = WARMUP_CALLS * stride
    bench_md  = BENCH_CALLS * stride

    print(f"\n{nx}x{ny} = {n_atoms} atoms  (stride={stride})")
    print(f"Atom size: {asiz} pixels")
    print(f"Warmup: {warmup_md} MD steps, Benchmark: {bench_md} MD steps")
    print("-" * 40)

    # Warmup
    for _ in range(WARMUP_CALLS):
        sim.step()

    # Benchmark
    start = get_time_ms()
    for _ in range(BENCH_CALLS):
        sim.step()
    elapsed = get_elapsed_sec(start)

    steps_per_sec = bench_md / elapsed
    ms_per_step   = elapsed / bench_md * 1000
    final_temp    = sim.get_temperature()
    target_temp   = sim.get_target_temperature()

    print(f"  Performance: {steps_per_sec:8.1f} MD steps/sec")
    print(f"  Time/step:   {ms_per_step:8.3f} ms")
    print(f"  Final T:     {final_temp:8.3f}  (target: {target_temp:.1f})")

print("\n" + "=" * 60)
print("Benchmark complete")
print("=" * 60)
