# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
MD Simulation benchmark script for Pico 2 with display output.
ulab + custom C version.
Tests multiple atom configurations (4x4 to 10x10).
Results shown on Pico Display 2.0.
Press any button to proceed between results, or any button on final screen to exit.

Run with: mpremote run run_test_ulab_user_c_func_pico_display.py
"""
import time
import gc
from pimoroni import Button
from picographics import PicoGraphics, DISPLAY_PICO_DISPLAY_2, PEN_P8
from md_sim_ulab_user_c_func import MDSimulation

# Pico Display 2.0 dimensions
BOX_X = 320
BOX_Y = 240

# Test configurations (matching run_test_C_pico.py)
CONFIGS = [
    {"nx": 5, "ny": 5, "as": 10},
    {"nx": 8, "ny": 8, "as": 10},
    {"nx": 10, "ny": 10, "as": 10},
    {"nx": 10, "ny": 10, "as": 8},
]

WARMUP_CALLS = 10
BENCH_CALLS = 100

# Setup display
display = PicoGraphics(display=DISPLAY_PICO_DISPLAY_2, pen_type=PEN_P8)
display.set_backlight(1.0)
WIDTH, HEIGHT = display.get_bounds()

# Setup buttons
button_a = Button(12)
button_b = Button(13)
button_x = Button(14)
button_y = Button(15)

# Colors
BG = display.create_pen(20, 20, 40)
TEXT_PEN = display.create_pen(255, 255, 255)
TITLE_PEN = display.create_pen(255, 200, 100)
RESULT_PEN = display.create_pen(100, 255, 100)

def check_any_button():
    """Return True if any button is pressed"""
    return button_a.read() or button_b.read() or button_x.read() or button_y.read()

def clear_display():
    display.set_pen(BG)
    display.clear()

def show_text(text, x, y, scale=2, pen=None):
    if pen:
        display.set_pen(pen)
    else:
        display.set_pen(TEXT_PEN)
    display.set_font("bitmap8")
    display.text(text, x, y, scale=scale)

def show_title():
    clear_display()
    show_text("MD BENCHMARK", 60, 10, scale=3, pen=TITLE_PEN)
    show_text("Pico 2 + ulab + custom C", 80, 50, scale=2)
    show_text(f"Box: {BOX_X}x{BOX_Y}", 10, 90, scale=2)
    show_text(f"{len(CONFIGS)} test configs", 10, 110, scale=2)
    show_text("Press a key to start", 10, 150, scale=2)

    display.update()

def show_testing(nx, ny, n_atoms, atom_size):
    clear_display()
    show_text("TESTING", 100, 10, scale=3, pen=TITLE_PEN)
    show_text(f"{nx}x{ny} = {n_atoms} atoms", 60, 60, scale=2)
    show_text(f"Atom size: {atom_size}", 80, 85, scale=2)
    show_text("Warming up...", 70, 120, scale=2)
    display.update()

def show_benchmarking(nx, ny):
    clear_display()
    show_text("TESTING", 100, 10, scale=3, pen=TITLE_PEN)
    show_text(f"{nx}x{ny} grid", 100, 60, scale=2)
    show_text("Benchmarking...", 60, 100, scale=2)
    display.update()

def show_result(nx, ny, n_atoms, atom_size, steps_per_sec, ms_per_step, final_temp):
    clear_display()
    show_text(f"{nx}x{ny} RESULT", 80, 10, scale=2, pen=TITLE_PEN)
    show_text(f"Atoms: {n_atoms}, size: {atom_size}", 10, 40, scale=2)
    show_text(f"MDPS: {steps_per_sec:.0f}", 10, 70, scale=2, pen=RESULT_PEN)
    show_text(f"ms/step: {ms_per_step:.3f}", 10, 100, scale=2)
    show_text(f"Final T: {final_temp:.2f}", 10, 130, scale=2)
    show_text("A/B/X/Y: next", 10, HEIGHT - 30, scale=2)
    display.update()

def show_final_results(results):
    clear_display()
    show_text("RESULTS", 100, 5, scale=2, pen=TITLE_PEN)

    y = 35
    for r in results:
        show_text(f"{r['nx']}x{r['ny']} (size {r['as']}): {r['mdps']:.0f} MD step/s", 10, y, scale=2, pen=RESULT_PEN)
        y += 22

    show_text("Press any key", 60, HEIGHT - 50, scale=2)
    show_text("to exit", 100, HEIGHT - 30, scale=2)
    display.update()

def wait_for_button():
    """Wait for any button press and release"""
    # Wait for press
    while not check_any_button():
        time.sleep(0.01)
    # Wait for release
    while check_any_button():
        time.sleep(0.01)
    time.sleep(0.1)  # Debounce

# Main benchmark
gc.collect()
show_title()
wait_for_button()
#time.sleep(1)

results = []

for config in CONFIGS:
    if check_any_button():
        break

    nx, ny, atom_size = config["nx"], config["ny"], config["as"]
    n_atoms = nx * ny

    show_testing(nx, ny, n_atoms, atom_size)

    sim = MDSimulation(
        n_atoms_x=nx,
        n_atoms_y=ny,
        atom_size=atom_size,
        box_size_x=BOX_X,
        box_size_y=BOX_Y,
        temperature=1.0
    )

    stride = sim.stride
    bench_md_steps = BENCH_CALLS * stride

    # Warmup
    for _ in range(WARMUP_CALLS):
        sim.step()

    show_benchmarking(nx, ny)

    # Benchmark
    start = time.ticks_ms()
    for _ in range(BENCH_CALLS):
        sim.step()
    elapsed = time.ticks_diff(time.ticks_ms(), start)

    if elapsed > 0:
        steps_per_sec = bench_md_steps * 1000 / elapsed
        ms_per_step = elapsed / bench_md_steps
    else:
        steps_per_sec = 0
        ms_per_step = 0
    final_temp = sim.get_temperature()

    results.append({
        'nx': nx,
        'ny': ny,
        'as': atom_size,
        'n_atoms': n_atoms,
        'mdps': steps_per_sec,
        'ms_per_step': ms_per_step,
        'final_temp': final_temp
    })

    show_result(nx, ny, n_atoms, atom_size, steps_per_sec, ms_per_step, final_temp)
    wait_for_button()

    # Clean up
    del sim
    gc.collect()

# Show final summary
show_final_results(results)
wait_for_button()

# Exit message
clear_display()
show_text("DONE", 120, HEIGHT // 2 - 10, scale=3, pen=TITLE_PEN)
display.update()
time.sleep(0.5)
