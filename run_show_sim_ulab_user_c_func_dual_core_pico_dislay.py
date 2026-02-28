# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
Dual-core MD simulation with display for Raspberry Pi Pico 2

Core 0: Display rendering (PicoGraphics) and user input (buttons)
Core 1: Physics simulation (continuous Leap Frog integration)

Hardware: Pico 2 (RP2350) + Pico Display 2.0 (320x240) + BNO055 accelerometer
Performance: ~20 FPS display, ~500 MD steps/second (100 atoms, C extension)

Controls:
  A: Cycle through options (Temp SP, Gravity, Pause, Coupling, Box accel. scale)
  X: Increase selected parameter
  Y: Decrease selected parameter
  X+Y: Take screenshot (saved as PPM)
  Hold B (1 sec): Shutdown

Run with: mpremote run run_show_sim_ulab_user_c_func_pico_screen_dual_core.py
"""

import time
import gc
import os
import _thread
from pimoroni import Button, RGBLED
from picographics import PicoGraphics, DISPLAY_PICO_DISPLAY_2, PEN_P8
from md_sim_ulab_user_c_func import MDSimulation
from bno055_handler import BNO055Handler

# ============================================================================
# SIMULATE PARAMETERS
# ============================================================================
N_ATOMS_X = 10
N_ATOMS_Y = 10
ATOM_SIZE = 8  # Radius for drawing the atoms, in MD LJ sigma = 2 * ATOM_SIZE

# ============================================================================
# UI PARAMETERS
# ============================================================================
EXPERT_MODE = True
N_NON_EXPERT_OPTIONS = 3  # number of options shown for non expert mode

OVERHEAT_TEMP = 1000  # Temperature threshold - pause sim and require restart

# ============================================================================
# VELOCITY INTEGRATION
# ============================================================================
ALPHA = 0.85  # Decay velocity

# ============================================================================
# SETUP THE HARDWARE
# ============================================================================

# Setup display
display = PicoGraphics(display=DISPLAY_PICO_DISPLAY_2, pen_type=PEN_P8)
display.set_backlight(1.0)
WIDTH, HEIGHT = display.get_bounds()

# Initialize buttons (Pico Display 2.0 layout)
button_a = Button(12)
button_b = Button(13)
button_x = Button(14)
button_y = Button(15)

# Initialize RGB LED (unused currently, but initialized)
led = RGBLED(0, 0, 0)

# Initialize BNO055 accelerometer
accel = BNO055Handler(
    gravity_sign=(1, 1, 1),  # per-axis sign to match display coordinate system
    lin_acc_sign=(-1, -1, 1))

# ============================================================================
# SETUP THE DISPLAY
# ============================================================================

# Create color pens (pre-create all to avoid memory allocation in loop)
# Track palette for screenshot function
palette = {}

BG = display.create_pen(20, 20, 40)
palette[BG] = (20, 20, 40)

ATOM_PEN = display.create_pen(255, 200, 100)
palette[ATOM_PEN] = (255, 200, 100)

TEXT_PEN = display.create_pen(255, 255, 255)
palette[TEXT_PEN] = (255, 255, 255)

RED_PEN = display.create_pen(255, 0, 0)
palette[RED_PEN] = (255, 0, 0)

BLUE_PEN = display.create_pen(0, 0, 255)
palette[BLUE_PEN] = (0, 0, 255)

GREEN_PEN = display.create_pen(0, 255, 0)
palette[GREEN_PEN] = (0, 255, 0)

ACCENT_PEN = display.create_pen(100, 150, 255)
palette[ACCENT_PEN] = (100, 150, 255)

HINT_PEN = display.create_pen(150, 150, 150)
palette[HINT_PEN] = (150, 150, 150)

# ============================================================================
# SHARED STATE (accessed by both cores)
# ============================================================================

# Simulation object - created on Core 1, read by Core 0 for display
sim = None

# User-adjustable parameters (Core 0 writes, Core 1 reads)
target_temp = 1.0
gravity = 0.005
coupling = 0.01
box_acc_scale = 0.04

# Status info (Core 1 writes, Core 0 reads for display)
current_temp = 0.0
current_sim_time = 0.0
mdsps = 0.0  # MD steps per second

# Control flags
running = True
physics_running = True
shutdown_requested = False
overheated = False
restart_requested = False

# debug
ax = 0
ay = 0

# ============================================================================
# HELP FUNCTIONS
# ============================================================================


def save_screenshot_ppm(display, palette, width, height):
    """
    Save framebuffer as PPM image file.
    Returns filename if successful, None if failed.

    PPM format: simple, portable, viewable on any OS.
    Files are numbered consecutively: shot_001.ppm, shot_002.ppm, etc.
    """
    # Find next available filename
    num = 1
    while True:
        filename = "shot_{:03d}.ppm".format(num)
        try:
            os.stat(filename)
            num += 1
        except OSError:
            break  # File doesn't exist, use this name

    try:
        fb = memoryview(display)

        with open(filename, "wb") as f:
            # PPM header (P6 = binary RGB)
            f.write("P6\n{} {}\n255\n".format(width, height).encode())

            # Convert palette indices to RGB pixels
            # Write in chunks to avoid memory issues
            chunk = bytearray(width * 3)
            for y in range(height):
                for x in range(width):
                    idx = fb[y * width + x]
                    r, g, b = palette.get(idx, (0, 0, 0))
                    chunk[x * 3] = r
                    chunk[x * 3 + 1] = g
                    chunk[x * 3 + 2] = b
                f.write(chunk)

        return filename
    except Exception as e:
        print("Screenshot error:", e)
        return None


def draw_calibration_bar(x, y, width, accel):
    """Draw calibration status bar."""
    labels = ["Sys", "Gyr", "Acc", "Mag"]
    bar_w = width // 4 - 5

    cal_status = accel.cal_status()

    for i, (label, val) in enumerate(zip(labels, cal_status)):
        bx = x + i * (bar_w + 5)

        # Background
        display.set_pen(ACCENT_PEN)
        display.rectangle(bx, y, bar_w, 12)

        # Fill based on calibration level (0-3)
        if val > 0:
            fill_w = int(bar_w * val / 3)
            if val == 3:
                display.set_pen(GREEN_PEN)
            elif val >= 2:
                display.set_pen(BLUE_PEN)
            else:
                display.set_pen(RED_PEN)
            display.rectangle(bx, y, fill_w, 12)

        # Label
        display.set_pen(TEXT_PEN)
        display.text(label, bx + 2, y + 2, scale=1)

# ============================================================================
# CORE 1: PHYSICS SIMULATION (continuous loop)
# ============================================================================


def physics_thread():
    """
    Runs on Core 1.
    Continuously steps the MD simulation.
    Every 20 MD step: reads accelerometer and updates box velocity (moving frame).
    Every 100 MD steps: applies user parameter changes, updates shared status variables.
    """
    global sim, current_temp, physics_running, shutdown_requested, mdsps, current_sim_time, box_acc_scale, ax, ay, restart_requested

    def create_sim():
        """Create a fresh MDSimulation instance."""
        return MDSimulation(
            n_atoms_x=N_ATOMS_X, n_atoms_y=N_ATOMS_Y,
            atom_size=ATOM_SIZE,
            box_size_x=WIDTH, box_size_y=HEIGHT,
            temperature=1.0
        )

    # Create simulation on Core 1
    sim = create_sim()

    call_count = 0
    steps_since_update = 0
    last_acc_x = 0
    last_acc_y = 0
    print("Core 1: Physics thread started")
    last_md_time = time.ticks_ms()

    try:
        while running and not shutdown_requested:
            # Handle restart request (overheat recovery)
            if restart_requested:
                print("Core 1: Restarting simulation...")
                sim = create_sim()
                call_count = 0
                steps_since_update = 0
                last_acc_x = 0
                last_acc_y = 0
                last_md_time = time.ticks_ms()
                mdsps = 0.0
                current_temp = 0.0
                restart_requested = False
                physics_running = True
                print("Core 1: Simulation restarted")

            if physics_running:  # Handle pause mode
                sim.step()
                call_count += 1
                steps_since_update += sim.stride

                # Update box velocity from accelerometer every stride (20) MD step
                if accel.connected:
                    ax, ay, _ = accel.lin_acc()
                    vx, vy = sim.get_box_vel()
                    vx = vx * ALPHA
                    vy = vy * ALPHA 

                    if max([abs(ax), abs(ay)]) > 0.5:  # Cap small signals
                        vx += (last_acc_x + ax)/2 * box_acc_scale  # all conversion and time is lumped into this one factor
                        vy += (last_acc_y + ay)/2 * box_acc_scale
                        last_acc_x = ax
                        last_acc_y = ay
                        # clamp wall speed
                        if abs(vx) > 1:
                            vx = vx/abs(vx)
                        if abs(vy) > 1:
                            vy = vy/abs(vy)

                    sim.set_box_vel([vx, vy])

                # Update status every ~100 MD steps (5 calls with stride=20)
                if call_count % 5 == 0:
                    elapsed_md = time.ticks_diff(time.ticks_ms(), last_md_time)
                    if elapsed_md > 0:
                        mdsps = steps_since_update * 1000 / elapsed_md
                    steps_since_update = 0
                    last_md_time = time.ticks_ms()

                    # Apply possible user parameter changes
                    sim.set_target_temperature(target_temp)
                    sim.set_thermostat_coupling(coupling)

                    # Update gravity orientation if accelerometer is present
                    if accel.connected:
                        gx, gy, _ = accel.get_normalize_gravity()
                        sim.set_gravity(gx * gravity, gy * gravity)

                    # Update display status
                    current_temp = sim.get_temperature()
                    current_sim_time, _ = sim.get_time_and_steps()
            else:
                time.sleep(0.001)  # Paused - yield CPU
    except Exception as e:
        print("Core 1 ERROR:", e)
        import sys
        sys.print_exception(e)

    print("Core 1: Physics thread exiting")


# ============================================================================
# CORE 0: DISPLAY AND USER INPUT
# ============================================================================


def main():
    """
    Runs on Core 0 (main thread).
    Handles display rendering and button input.
    Reads particle positions from Core 1's simulation for drawing.
    """
    global target_temp, gravity, coupling, box_acc_scale, physics_running, running, shutdown_requested, ax, ay, overheated, restart_requested

    gc.collect()  # Clean up after initialization

    # Screenshot state
    screenshot_message_until = 0
    screenshot_filename = ""

    # Menu state for parameter adjustment
    current_option = 0  # 0=Temp SP, 1=Gravity, 2=Pause, 3=Coupling, 4=Box accel. scale
    option_names = ["Temp SP", "Gravity", "Pause", "Coupling", "Box Accel. scale"]
    last_button_time = 0
    DEBOUNCE_MS = 200

    # Button B hold detection for shutdown
    button_b_hold_start = 0
    SHUTDOWN_HOLD_MS = 1000  # Hold B for 1 second to shutdown

    # Wait for Core 1 to initialize simulation
    print("Core 0: Waiting for simulation to initialize on Core 1...")
    while sim is None:
        time.sleep(0.01)

    print("Core 0: Simulation ready, starting render loop")

    BALL_SIZE = int(sim.atom_size)
    frame = 0

    while running and not shutdown_requested:
        current_time = time.ticks_ms()

        # ====================================================================
        # BUTTON HANDLING
        # ====================================================================

        # Check Button B hold for shutdown (always check, no debounce)
        if button_b.read():
            if button_b_hold_start == 0:
                button_b_hold_start = current_time
            elif time.ticks_diff(current_time, button_b_hold_start) > SHUTDOWN_HOLD_MS:
                print("\nButton B held - shutting down...")
                shutdown_requested = True
                running = False
                break
        else:
            button_b_hold_start = 0  # Reset if released

        # Check for overheat
        if not overheated and current_temp > OVERHEAT_TEMP:
            overheated = True
            physics_running = False
            print(f"OVERHEAT! T={current_temp:.1f} > {OVERHEAT_TEMP}")

        # Debounced button handling for A, X, Y
        if time.ticks_diff(current_time, last_button_time) > DEBOUNCE_MS:

            if overheated:
                # Only A is active during overheat - triggers restart
                if button_a.read():
                    print("Restart requested by user")
                    overheated = False
                    target_temp = 1.0
                    restart_requested = True
                    # Wait for Core 1 to finish restarting
                    time.sleep(0.1)
                    last_button_time = current_time
            else:
                x_pressed = button_x.read()
                y_pressed = button_y.read()

                # X+Y together = screenshot
                if x_pressed and y_pressed:
                    filename = save_screenshot_ppm(display, palette, WIDTH, HEIGHT)
                    if filename:
                        screenshot_filename = filename
                        screenshot_message_until = current_time + 1500  # Show for 1.5 sec
                        print("Screenshot saved:", filename)
                    last_button_time = current_time

                elif button_a.read():
                    if EXPERT_MODE:
                        current_option = (current_option + 1) % len(option_names)
                    else:
                        current_option = (current_option + 1) % N_NON_EXPERT_OPTIONS 
                    last_button_time = current_time

                elif x_pressed:
                    if current_option == 0:    # Temperature
                        target_temp += 0.2
                    elif current_option == 1:  # Gravity
                        gravity += 0.001
                    elif current_option == 2:  # Pause toggle
                        physics_running = not physics_running
                    elif current_option == 3:  # Coupling
                        coupling += 0.005
                    elif current_option == 4:  # Box accel. scale
                        box_acc_scale += 0.01
                    last_button_time = current_time

                elif y_pressed:
                    if current_option == 0:    # Temperature
                        target_temp = max(0, target_temp - 0.2)
                    elif current_option == 1:  # Gravity
                        gravity -= 0.001
                    elif current_option == 2:  # Pause toggle
                        physics_running = not physics_running
                    elif current_option == 3:  # Coupling
                        coupling = max(0.0, coupling - 0.005)
                    elif current_option == 4:  # Box accel. scale
                        box_acc_scale -= 0.01
                    last_button_time = current_time

        # ====================================================================
        # RENDER
        # ====================================================================

        display.set_pen(BG)
        display.clear()

        if overheated:
            # ---- OVERHEAT SCREEN ----
            display.set_pen(RED_PEN)
            display.text("OVERHEAT!", WIDTH // 2 - 60, 40, scale=3)

            display.set_pen(TEXT_PEN)
            display.set_font("bitmap8")
            display.text("Atoms overheated", WIDTH // 2 - 70, 100, scale=2)
            display.text("T = " + str(int(current_temp)), WIDTH // 2 - 35, 130, scale=2)

            display.set_pen(ACCENT_PEN)
            display.text("Press A to restart", WIDTH // 2 - 75, 180, scale=2)

            display.set_pen(HINT_PEN)
            display.text("Hold B to exit", WIDTH // 2 - 55, HEIGHT - 25, scale=2)

        else:
            # ---- NORMAL SIMULATION RENDER ----

            # Draw particles (read positions from Core 1's simulation)
            # No lock needed - array reads are atomic enough for visualization
            try:
                px = sim.pos_x
                py = sim.pos_y
                display.set_pen(ATOM_PEN)
                for i in range(sim.n_atoms):
                    display.circle(int(px[i]), int(py[i]), BALL_SIZE)
            except:
                pass  # Skip frame if race condition during init

            # Display UI text
            display.set_pen(TEXT_PEN)
            display.set_font("bitmap8")

            # Current option and value (top-left)
            option_text = option_names[current_option]
            display.text(option_text, 5, 5, scale=2)

            if current_option == 0:
                value_text = str(int(target_temp * 10) / 10)
            elif current_option == 1:
                value_text = str(int(gravity * 1000) / 1000)
                if accel.connected:
                    gx, gy = sim.get_gravity()
                    value_text += (" (" + str(int(gx * 10000) / 10000) + ", "
                                   + str(int(gy * 10000) / 10000) + ")")
            elif current_option == 2:
                value_text = "ON" if physics_running else "OFF"
            elif current_option == 3:
                value_text = str(int(coupling * 10000) / 10000)
            elif current_option == 4:
                if accel.connected:
                    value_text = str(int(box_acc_scale * 100) / 100)
                    vx, vy = sim.get_box_vel()
                    value_text += (" (" + str(int(vx * 10000) / 10000) + ", "
                                   + str(int(vy * 10000) / 10000) + ")")
                else:
                    value_text = "NaN"
            display.text(value_text, 5, 25, scale=2)

            # Simulation temperature (top-right)
            display.text("T:" + str(int(current_temp * 10) / 10), WIDTH - 60, 5, scale=2)

            # MDPS counters (bottom-right)
            display.text("MDPS: " + str(int(mdsps)), WIDTH - 90, HEIGHT - 15, scale=2)

            if not physics_running and accel.connected and not accel.calibrated:
                draw_calibration_bar(120, 5, 195, accel)

            # Shutdown progress bar (if Button B held)
            if button_b_hold_start > 0:
                hold_duration = time.ticks_diff(current_time, button_b_hold_start)
                progress = min(1.0, hold_duration / SHUTDOWN_HOLD_MS)
                bar_width = int(100 * progress)

                display.set_pen(RED_PEN)
                display.rectangle(WIDTH//2 - 50, HEIGHT//2 - 10, bar_width, 20)
                display.set_pen(TEXT_PEN)
                display.text("HOLD B TO EXIT", WIDTH//2 - 60, HEIGHT//2 - 30, scale=1)

            # Screenshot saved message (temporary overlay)
            if time.ticks_diff(screenshot_message_until, current_time) > 0:
                display.set_pen(GREEN_PEN)
                display.rectangle(WIDTH//2 - 70, HEIGHT//2 - 15, 140, 30)
                display.set_pen(BG)
                display.text("Saved: " + screenshot_filename, WIDTH//2 - 60, HEIGHT//2 - 5, scale=2)

        # Periodic garbage collection to prevent memory issues
        if frame % 100 == 0:
            gc.collect()

        display.update()
        frame += 1

    # Cleanup on exit
    print("\nCore 0: Display loop exiting")

    display.set_pen(BG)
    display.clear()
    display.set_pen(TEXT_PEN)
    display.text("SHUTDOWN", WIDTH//2 - 40, HEIGHT//2 - 10, scale=2)
    display.update()

    led.set_rgb(0, 0, 0)
    time.sleep(0.5)  # Give Core 1 time to exit
    print("Core 0: Shutdown complete")

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DUAL-CORE MD SIMULATION")
    print("=" * 60)
    print("Core 0: Display rendering + UI")
    print("Core 1: Physics simulation")
    print("=" * 60)

    # Start physics thread on Core 1
    print("Starting physics thread on Core 1...")
    _thread.start_new_thread(physics_thread, ())

    # Run display on Core 0 (main thread)
    print("Running display on Core 0...")
    print("Press X+Y together to take screenshot")
    print("Hold Button B for 1 second to exit")
    time.sleep(0.5)  # Give Core 1 time to initialize

    main()

    print("\n" + "=" * 60)
    print("Shutdown complete - safe to restart or exit REPL")
    print("=" * 60)
