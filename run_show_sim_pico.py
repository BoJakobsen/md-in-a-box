# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

"""
Unified binary LJ MD simulation: Demo + Manual modes on Pico 2 display.

Architecture
------------
Core 0  Display rendering, button input, demo sequencer (main thread).
Core 1  Physics — creates and steps the MDSimulation object.

All inter-core communication goes through module-level globals.  Each
variable is owned by one core (see SHARED STATE section); the only
exceptions are `running`, `shutdown_requested`, and `overheated`, which
either core may write.

Demo Mode  (default at startup)
  Autonomous sequencer cycling through scripted settings with linear T ramps.
  Each Tsetpoint is the *end* temperature for that phase; the thermostat ramps
  linearly from the previous phase's end temperature over Nsteps MD steps.
  A        : Cycle info view (setpoint T | LJ params | gravity)
  X        : Skip forward to next setting  (fresh sim)
  Y        : Skip backward to previous setting  (fresh sim)
  B press  : Switch to Manual mode  (keeps current sim state and LJ params)
  A+B      : Shutdown

Manual Mode
  Full parameter control; simulation continues from current demo state.
  A        : Cycle option  (expert: all 7 | basic: first 3)
  X / Y    : Increase / decrease selected option
             (Temp SP at 0: first Y arms quench prompt,
              second Y within 2 s instantly zeroes all velocities)
  B release: Back to Demo mode  (T-ramp resumes, LJ params kept)
  A+B      : Shutdown
"""

import time
import gc
import _thread
from pimoroni import Button, RGBLED
from picographics import PicoGraphics, DISPLAY_PICO_DISPLAY_2, PEN_P8
from md_sim import MDSimulation
from bno055_handler import BNO055Handler
from ulab import numpy as np


# ============================================================================
# DEMO SETTINGS
# Each dict is one scenario.  Tsetpoints and Nsteps are paired lists.
# Each Tsetpoint is the *end* temperature for that phase.
# A Tsetpoint of -1 means "ramp to 0 then zero all velocities" (hard quench).
# Settings loop continuously; add or reorder freely.
# ============================================================================

# Standard temperature protocol — edit here: [end_T, N_steps, 'subtitle']
# ~45 000 steps ≈ 1.5 minutes real time at ~500 MD steps/s on Pico 2
standard_protocol = [
    [ 5.0,  45000, 'Hot gas'          ],
    [ 0.5,  45000, 'Cooling'       ],
    [ 0.5,  45000, 'Liquid like'       ],
    [ 0.1,  45000, 'Cooling'],
    [ 0.1,  20000, 'Crystal'      ],
    [ -1 ,      1, ''             ],   # hard quench: ramp to 0, zero velocities
    [ 0.0,  25000, 'Crystal'      ],
    [ 2.0,  45000, 'Melting'      ],
]
standard_protocol_SP     = [r[0] for r in standard_protocol]
standard_protocol_Nsteps = [r[1] for r in standard_protocol]
standard_protocol_subtit = [r[2] for r in standard_protocol]

settings = [
    {
        'title':            'Monoatomic system',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0,
        'r_sizeBparticles': 1.0,
        'epsilonAB':        1,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Symmetric mixture',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0.5,
        'r_sizeBparticles': 1.0,
        'epsilonAB':        1,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Same size, AB disfavored',
        'gravity':          0.01,
        'coupling':         0.02,
        'r_NBparticles':    0.5,
        'r_sizeBparticles': 1.0,
        'epsilonAB':        0.1,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Same size, AB favored',
        'gravity':          0.01,
        'coupling':         0.02,
        'r_NBparticles':    0.5,
        'r_sizeBparticles': 1.0,
        'epsilonAB':        2.0,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Different size, AB disfavored',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0.3,
        'r_sizeBparticles': 1.5,
        'epsilonAB':        0.5,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Different size, AB favored.',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0.3,
        'r_sizeBparticles': 1.5,
        'epsilonAB':        2,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Few large, AB favored',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0.1,
        'r_sizeBparticles': 1.3,
        'epsilonAB':        2,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
    {
        'title':            'Few small, AB favored',
        'gravity':          0.01,
        'coupling':         0.01,
        'r_NBparticles':    0.1,
        'r_sizeBparticles': 0.6,
        'epsilonAB':        2,
        'Tsetpoints': standard_protocol_SP,
        'Nsteps':     standard_protocol_Nsteps,
        'subtitles':  standard_protocol_subtit,
    },
]

# ============================================================================
# SIMULATION CONSTANTS
# ============================================================================
N_ATOMS_X     = 10
N_ATOMS_Y     = 10
ATOM_SIZE     = 8     # Type A radius in pixels; sigma_AA = 2 * ATOM_SIZE
OVERHEAT_TEMP = 1000  # Temperature threshold that triggers the overheat screen


# ============================================================================
# UI CONSTANTS
# ============================================================================
EXPERT_MODE      = True   # False = show only first N_BASIC_MANUAL options
N_BASIC_MANUAL   = 3      # Tsp, Gravity, Pause  (non-expert mode)
ALPHA            = 0.85   # Accelerometer velocity decay coefficient
BOX_ACC_SCALE    = 0.04   # Lin-acceleration → box velocity scale factor
DEBOUNCE_MS      = 200    # Button debounce window in milliseconds

MANUAL_OPTIONS   = ["Tsp", "Gravity", "Pause", "Coupling",
                    "r_NB", "r_sizeB", "epsilon_AB"]
N_MANUAL_OPTIONS = len(MANUAL_OPTIONS)
N_DISPLAY_MODES  = 3      # Button A cycles 3 info views in Demo mode


# ============================================================================
# HARDWARE SETUP
# ============================================================================
display = PicoGraphics(display=DISPLAY_PICO_DISPLAY_2, pen_type=PEN_P8)
display.set_backlight(1.0)
WIDTH, HEIGHT = display.get_bounds()

button_a = Button(12)
button_b = Button(13)
button_x = Button(14)
button_y = Button(15)

led   = RGBLED(0, 0, 0)
accel = BNO055Handler(
    axis_map=(1, 0, 2),   # Configure for concrete chip orientation
    axis_sign=(1, -1, 1), #
)


# ============================================================================
# DISPLAY PENS
# ============================================================================
palette = {}

BG         = display.create_pen(20,  20,  40);  palette[BG]         = (20,  20,  40)
ATOM_A_PEN = display.create_pen(207, 23,  23);  palette[ATOM_A_PEN] = (207, 23,  23)
ATOM_B_PEN = display.create_pen(27,  63, 161);  palette[ATOM_B_PEN] = (27,  63, 161)
TEXT_PEN   = display.create_pen(255, 255, 255);  palette[TEXT_PEN]   = (255, 255, 255)
RED_PEN    = display.create_pen(255,   0,   0);  palette[RED_PEN]    = (255,   0,   0)
BLUE_PEN   = display.create_pen(  0,   0, 255);  palette[BLUE_PEN]   = (  0,   0, 255)
GREEN_PEN  = display.create_pen(  0, 255,   0);  palette[GREEN_PEN]  = (  0, 255,   0)
ACCENT_PEN = display.create_pen(100, 150, 255);  palette[ACCENT_PEN] = (100, 150, 255)
HINT_PEN   = display.create_pen(150, 150, 150);  palette[HINT_PEN]   = (150, 150, 150)


# ============================================================================
# GRADIENT PENS FOR DEMO PROGRESS BAR
# Index 0 = cold (dark blue), index N_GRAD-1 = hot (red).
# 320 px / 64 segments = 5 px/segment — smooth on a 6 px bar.
# ============================================================================
N_GRAD = 64
BAR_H  = 6

grad_pens = []
for _i in range(N_GRAD):
    _t = _i / (N_GRAD - 1)          # 0.0 = cold, 1.0 = hot
    grad_pens.append(display.create_pen(
        int(20 + 200 * _t),          # R:  20 → 220
        20,                          # G: flat
        int(180 - 160 * _t),         # B: 180 →  20
    ))


# ============================================================================
# SHARED STATE
# Variables are grouped by which core writes them.
# ============================================================================

# -- Core 0 writes → Core 1 reads --
target_temp        = 1.0    # current T setpoint (demo: ramp value; manual: user set)
gravity            = 0.01
coupling           = 0.01
epsilonAB          = 1.0
r_NBparticles      = 0.5
r_sizeBparticles   = 1.0
demo_mode          = True   # True = demo sequencer, False = manual control
physics_running    = True   # False = simulation paused
quench_requested   = False  # one-shot: Core 1 zeroes velocities then clears this
pending_sim_config = None   # dict: Core 1 creates a new sim from this, then clears it

# -- Core 1 writes → Core 0 reads --
sim          = None          # MDSimulation object (None until first sim is ready)
current_temp = 0.0           # instantaneous kinetic temperature
steps_done   = 0             # MD steps completed in the current demo phase

# -- Both cores may write --
running            = True
shutdown_requested = False
overheated         = False   # set by Core 1 on overheat; cleared by Core 0 on restart

# -- Demo sequencer state (Core 0 owns; Core 1 never writes these) --
demo_s_idx        = 0        # index into settings[]
demo_sp_i         = 0        # phase index within the current setting
demo_t_start      = -1.0     # ramp start T for this phase  (-1 = needs init)
demo_t_sp         = 0.0      # ramp end T for this phase
demo_n_st         = 1        # Nsteps for this phase
demo_phase_quench = False    # True when this phase ends with a velocity quench


# ============================================================================
# HELPER: SIM FACTORY
# ============================================================================

def create_sim(config):
    """
    Create a fresh MDSimulation from a parameter dict.

    Recognised keys (all optional):
        temperature      float  initial temperature          (default 1.0)
        r_NBparticles    float  fraction of B atoms          (default 0.5)
        r_sizeBparticles float  B radius relative to A       (default 1.0)
        epsilonAB        float  A-B LJ well depth            (default 1.0)
        coupling         float  Langevin thermostat coupling  (default 0.01)
        gravity          float  downward gravity magnitude   (default 0.01)
    """
    ns = MDSimulation(
        n_atoms_x=N_ATOMS_X, n_atoms_y=N_ATOMS_Y,
        atom_size=ATOM_SIZE,
        box_size_x=WIDTH, box_size_y=HEIGHT,
        temperature=float(config.get('temperature', 1.0)),
        r_NBparticles=config.get('r_NBparticles', 0.5),
    )
    ns.set_r_sizeBparticles(config.get('r_sizeBparticles', 1.0))
    ns.set_epsilonAB(config.get('epsilonAB', 1.0))
    ns.set_thermostat_coupling(config.get('coupling', 0.01))
    ns.set_gravity(0.0, config.get('gravity', 0.01))
    return ns


# ============================================================================
# HELPER: ACCELEROMETER BOX VELOCITY
# Integrates the BNO055 linear acceleration into the simulation box velocity.
# Called every physics step in both Demo and Manual modes when accel is present.
# ============================================================================

_last_acc_x = 0.0
_last_acc_y = 0.0


def update_accel_box_vel(sim_obj):
    """Update sim box velocity from accelerometer linear acceleration.

    Each call decays the current box velocity by ALPHA (<1) so the box
    gradually comes to rest when the physical device stops moving.
    Acceleration is integrated only above the noise threshold (0.5 m/s²);
    a trapezoidal rule (average of last and current sample) reduces impulse
    noise from single spiky readings.
    Box velocity is clamped to ±1 pixel/step to prevent runaway motion.
    """
    global _last_acc_x, _last_acc_y
    if not accel.connected:
        return
    ax, ay, _ = accel.lin_acc()
    vx, vy    = sim_obj.get_box_vel()
    # Decay: bleed off velocity each step so box drifts to rest
    vx       *= ALPHA
    vy       *= ALPHA
    if max(abs(ax), abs(ay)) > 0.5:
        # Integrate acceleration (trapezoidal: average of last + current sample)
        vx += (_last_acc_x + ax) / 2 * BOX_ACC_SCALE
        vy += (_last_acc_y + ay) / 2 * BOX_ACC_SCALE
        _last_acc_x = ax
        _last_acc_y = ay
        # Clamp to ±1 pixel/step to prevent runaway
        if abs(vx) > 1: vx = vx / abs(vx)
        if abs(vy) > 1: vy = vy / abs(vy)
    sim_obj.set_box_vel([vx, vy])


# ============================================================================
# HELPER: SIM REQUEST FUNCTIONS
# Core 0 fills pending_sim_config; Core 1 picks it up and creates the sim.
# ============================================================================

def request_demo_sim(s_idx):
    """
    Request a fresh sim for settings[s_idx] and reset the demo sequencer.

    Syncs all shared parameter globals from the chosen setting so both
    the physics thread and the manual-mode display show consistent values.

    Demo overheat restart: call with the current demo_s_idx to restart the
    same setting from the beginning of the temperature sequence.
    """
    global pending_sim_config, steps_done
    global demo_s_idx, demo_sp_i, demo_t_start
    global gravity, coupling, epsilonAB, r_NBparticles, r_sizeBparticles
    global target_temp, physics_running, overheated

    s                = settings[s_idx]
    demo_s_idx       = s_idx
    demo_sp_i        = 0
    demo_t_start     = -1.0
    steps_done       = 0
    gravity          = s['gravity']
    coupling         = s['coupling']
    epsilonAB        = s['epsilonAB']
    r_NBparticles    = s['r_NBparticles']
    r_sizeBparticles = s['r_sizeBparticles']
    target_temp      = float(s['Tsetpoints'][0])
    physics_running  = True
    overheated       = False
    pending_sim_config = {
        'temperature':      float(s['Tsetpoints'][0]),
        'r_NBparticles':    s['r_NBparticles'],
        'r_sizeBparticles': s['r_sizeBparticles'],
        'epsilonAB':        s['epsilonAB'],
        'coupling':         s['coupling'],
        'gravity':          s['gravity'],
    }
    print("Core 0: [Demo] Request sim for setting", s_idx, "—", s['title'])


def request_manual_sim():
    """
    Request a fresh sim grid using all current manual parameters, at T=1.0.

    Called after overheat in Manual mode so the user can continue their work
    without losing their parameter choices.
    TODO: could offer a choice between resetting T to 1.0 vs keeping T.
    """
    global pending_sim_config, target_temp, physics_running, overheated

    target_temp      = 1.0
    physics_running  = True
    overheated       = False
    pending_sim_config = {
        'temperature':      1.0,
        'r_NBparticles':    r_NBparticles,
        'r_sizeBparticles': r_sizeBparticles,
        'epsilonAB':        epsilonAB,
        'coupling':         coupling,
        'gravity':          gravity,
    }
    print("Core 0: [Manual] Request fresh sim (overheat restart)")


# ============================================================================
# HELPER: DRAW CALIBRATION BAR
# ============================================================================

def draw_calibration_bar(x, y, width):
    """Draw BNO055 calibration status as four colour-coded segments.
    """
    labels = ["Sys", "Gyr", "Acc", "Mag"]
    bar_w  = width // 4 - 5
    cal    = accel.cal_status()
    for i, (label, val) in enumerate(zip(labels, cal)):
        bx = x + i * (bar_w + 5)
        display.set_pen(ACCENT_PEN)
        display.rectangle(bx, y, bar_w, 12)
        if val > 0:
            fill_w = int(bar_w * val / 3)
            display.set_pen(GREEN_PEN if val == 3 else BLUE_PEN if val >= 2 else RED_PEN)
            display.rectangle(bx, y, fill_w, 12)
        display.set_pen(TEXT_PEN)
        display.text(label, bx + 2, y + 2, scale=1)


# ============================================================================
# HELPER: DRAW GRADIENT PROGRESS BAR  (Demo mode only)
# ============================================================================

def draw_progress_bar():
    """
    Draw a colour-gradient progress bar across the top of the screen.

    Bar length = fraction of the full setting timeline completed.
    Bar colour at each segment = temperature at that point in the protocol
    (cold dark-blue → hot red), using the grad_pens palette.
    """
    try:
        s           = settings[demo_s_idx]
        total_steps = sum(s['Nsteps'])
        if total_steps == 0:
            return
        steps_prev   = sum(s['Nsteps'][:demo_sp_i])
        filled_steps = steps_prev + steps_done
        filled_w     = WIDTH * filled_steps // total_steps

        t_vals  = [float(t) for t in s['Tsetpoints']]
        t_pos   = [max(0.0, t) for t in t_vals]   # treat quench sentinel as 0
        t_min   = min(t_pos)
        t_max   = max(t_pos)
        t_range = t_max - t_min if t_max > t_min else 1.0

        seg_w = WIDTH // N_GRAD
        cum   = 0
        for seg in range(N_GRAD):
            x = seg * seg_w
            if x >= filled_w:
                break
            mid_step = (x + seg_w // 2) * total_steps // WIDTH
            t_at_seg = t_pos[-1]
            cum      = 0
            for sp in range(len(t_vals)):
                t_sp_v = t_pos[sp]
                t_prev = t_pos[sp - 1] if sp > 0 else t_sp_v
                n_st_v = s['Nsteps'][sp]
                if mid_step < cum + n_st_v:
                    frac     = (mid_step - cum) / n_st_v
                    t_at_seg = t_prev + (t_sp_v - t_prev) * frac
                    break
                cum += n_st_v
            norm    = max(0.0, min(1.0, (t_at_seg - t_min) / t_range))
            pen_idx = int(norm * (N_GRAD - 1))
            draw_w  = min(seg_w, filled_w - x)
            display.set_pen(grad_pens[pen_idx])
            display.rectangle(x, 0, draw_w, BAR_H)
    except Exception:
        pass   # skip on race condition during setting transitions


# ============================================================================
# HELPER: BUILD DISPLAY FIELDS
# Returns a dict of (text, pen, x, y) tuples describing the text overlay.
# render_overlay() consumes this dict.
# ============================================================================

def build_display_fields(display_mode, current_option,
                         quench_pending, quench_pending_until, current_time):
    """
    Compute all text overlay fields for the current frame.

    Returns a dict with keys:
        mode_label, temp_label, info1, info2, bottom  — each (text, pen, x, y)
        alert  — string tag or None:
                 'overheat_demo', 'overheat_manual', 'quench_pending'
    """
    f = {}

    # Temperature readout — always shown top-right
    f['temp_label'] = ("T:" + str(int(current_temp * 10) / 10),
                       TEXT_PEN, WIDTH - 60, 8)

    # Overheat screen overrides all other content in both modes
    if overheated:
        label     = "AUTO"    if demo_mode else "MANUAL"
        label_pen = HINT_PEN  if demo_mode else ACCENT_PEN
        label_x   = WIDTH - 140 if demo_mode else WIDTH - 175
        f['mode_label'] = (label, label_pen, label_x, 8)
        f['info1']      = None
        f['info2']      = None
        f['bottom']     = None
        f['alert']      = 'overheat_demo' if demo_mode else 'overheat_manual'
        return f

    if demo_mode:
        f['mode_label'] = ("AUTO", HINT_PEN, WIDTH - 140, 8)

        if display_mode == 0:
            f['info1'] = ("Tsp=" + str(round(target_temp, 2)), TEXT_PEN, 5, 8)
            f['info2'] = None
        elif display_mode == 1:
            if sim is not None:
                f['info1'] = ("epsilon_AB=" + str(round(sim.epsilonAB, 2)),
                              TEXT_PEN, 5, 8)
                f['info2'] = ("r_NB=" + str(round(sim.r_NBparticles, 2))
                              + "  r_sizeB=" + str(round(sim.r_sizeBparticles, 2)),
                              TEXT_PEN, 5, 26)
            else:
                f['info1'] = f['info2'] = None
        else:   # display_mode == 2: dynamics
            if sim is not None:
                gx, gy = sim.get_gravity()
                f['info1'] = ("gravity=" + str(int(gravity * 10000) / 10000),
                              TEXT_PEN, 5, 8)
                if accel.connected:
                    f['info2'] = (" (" + str(int(gx * 10000) / 10000)
                                  + ", " + str(int(gy * 10000) / 10000) + ")",
                                  TEXT_PEN, 5, 26)
                else:
                    f['info2'] = None
            else:
                f['info1'] = f['info2'] = None

        try:
            subtitle = settings[demo_s_idx]['subtitles'][demo_sp_i]
        except (KeyError, IndexError):
            subtitle = ""
        title_text = settings[demo_s_idx]['title'] + (": " + subtitle if subtitle else "")
        f['bottom'] = (title_text, ACCENT_PEN, 5, HEIGHT - 17)
        f['alert']  = None

    else:   # Manual mode
        f['mode_label'] = ("MANUAL", ACCENT_PEN, WIDTH - 175, 8)
        f['info1']      = (MANUAL_OPTIONS[current_option], TEXT_PEN, 5, 8)

        if   current_option == 0:
            val = str(int(target_temp * 10) / 10)
        elif current_option == 1:
            val = str(int(gravity * 1000) / 1000)
            if accel.connected and sim is not None:
                gx, gy = sim.get_gravity()
                val += (" (" + str(int(gx * 10000) / 10000)
                        + ", " + str(int(gy * 10000) / 10000) + ")")
        elif current_option == 2:
            val = "ON" if physics_running else "OFF"
        elif current_option == 3:
            val = str(int(coupling * 10000) / 10000)
        elif current_option == 4:
            val = (str(round(r_NBparticles, 2))
                   + (" (" + str(sim.n_atomsB) + "B)" if sim is not None else ""))
        elif current_option == 5:
            val = str(round(r_sizeBparticles, 2))
        elif current_option == 6:
            val = str(round(epsilonAB, 2))
        else:
            val = ""

        f['info2']  = (val, TEXT_PEN, 5, 26)
        f['bottom'] = None
        f['alert']  = ('quench_pending'
                       if quench_pending
                       and time.ticks_diff(quench_pending_until, current_time) > 0
                       else None)

    return f


# ============================================================================
# HELPER: RENDER TEXT OVERLAY
# ============================================================================

def render_overlay(fields):
    """
    Draw the text overlay produced by build_display_fields().
    Particle rendering and the progress bar are handled separately in main().
    """
    display.set_font("bitmap8")

    for key in ('mode_label', 'temp_label', 'info1', 'info2', 'bottom'):
        entry = fields.get(key)
        if entry:
            text, pen, x, y = entry
            display.set_pen(pen)
            display.text(text, x, y, scale=2)

    alert = fields.get('alert')
    if alert in ('overheat_demo', 'overheat_manual'):
        display.set_pen(RED_PEN)
        display.text("OVERHEAT!", WIDTH // 2 - 60, 40, scale=3)
        display.set_pen(TEXT_PEN)
        display.text("T = " + str(int(current_temp)), WIDTH // 2 - 35, 100, scale=2)
        hint = "press any button: restart" if alert == 'overheat_demo' else "press any button: new sim"
        display.set_pen(ACCENT_PEN)
        display.text(hint, WIDTH // 2 - 90, 150, scale=2)
    elif alert == 'quench_pending':
        display.set_pen(ACCENT_PEN)
        display.text("Press down again to quench", 5, HEIGHT - 52, scale=2)


# ============================================================================
# CORE 1: PHYSICS — simple step loop
#
# Core 1 is responsible only for creating sims and running physics.
# All sequencer logic lives in Core 0 (update_demo_sequencer).
# Core 1 applies whatever parameter globals Core 0 has set.
# ============================================================================

def physics_thread():
    """
    Runs on Core 1.

    Each iteration:
      1. If pending_sim_config is set by Core 0, create a fresh sim.
      2. Apply a velocity quench if requested.
      3. If physics is running:
           a. Every 5 calls: apply all parameters from shared globals,
              read temperature, check for overheat.
           b. Update box velocity from accelerometer (if connected).
           c. Step the simulation; increment steps_done.
      4. If physics is paused, yield CPU briefly.

    Overheat stops physics and sets the overheated flag; Core 0 handles
    the appropriate restart for each mode.
    """
    global sim, current_temp, steps_done
    global physics_running, quench_requested, overheated, pending_sim_config

    call_count = 0
    print("Core 1: Physics thread started")

    try:
        while running and not shutdown_requested:

            # Create new sim if Core 0 has requested one
            if pending_sim_config is not None:
                sim                = create_sim(pending_sim_config)
                pending_sim_config = None
                call_count         = 0
                print("Core 1: New sim created")

            if sim is None:
                time.sleep(0.001)
                continue

            # One-shot velocity quench
            if quench_requested:
                sim.vel_x[:] = 0.0
                sim.vel_y[:] = 0.0
                quench_requested = False

            if not physics_running:
                time.sleep(0.001)
                continue

            # Apply parameters and check temperature (~every 100 MD steps)
            if call_count % 5 == 0:
                sim.set_target_temperature(target_temp)
                sim.set_thermostat_coupling(coupling)
                sim.set_r_sizeBparticles(r_sizeBparticles)
                sim.set_r_NBparticles(r_NBparticles)
                sim.set_epsilonAB(epsilonAB)
                if accel.connected:
                    gx, gy, _ = accel.get_normalize_gravity()
                    sim.set_gravity(gx * gravity, gy * gravity)
                else:
                    sim.set_gravity(0.0, gravity)
                current_temp = sim.get_temperature()
                if current_temp > OVERHEAT_TEMP:
                    overheated      = True
                    physics_running = False
                    print("Core 1: Overheat T=", current_temp)

            update_accel_box_vel(sim)
            sim.step()
            steps_done += sim.stride
            call_count += 1

    except Exception as e:
        print("Core 1 ERROR:", e)
        import sys
        sys.print_exception(e)

    print("Core 1: Physics thread exiting")


# ============================================================================
# DEMO SEQUENCER  (Core 0)
# Called once per display frame while in demo mode (and not overheated).
# Maintains the linear T-ramp and advances phases / settings as steps complete.
# ============================================================================

def update_demo_sequencer(skip_delta):
    """
    Advance the demo sequencer by one display frame.

    skip_delta: +1 = jump to next setting, -1 = jump to previous, 0 = normal.
    Always returns 0 so the caller can reset its skip_delta in one assignment.
    """
    global demo_sp_i, demo_t_start, demo_t_sp, demo_n_st
    global demo_phase_quench, target_temp, steps_done, quench_requested

    # Handle X / Y skip buttons
    if skip_delta != 0:
        request_demo_sim((demo_s_idx + skip_delta) % len(settings))
        return 0

    s = settings[demo_s_idx]

    # Initialise phase parameters when entering a new phase
    if demo_t_start < 0:
        t_sp_v = float(s['Tsetpoints'][demo_sp_i])
        n_st_v = s['Nsteps'][demo_sp_i]
        if t_sp_v < 0:                       # sentinel: hard quench
            demo_phase_quench = True
            t_sp_v = 0.0
        else:
            demo_phase_quench = False
        t_start_v = (float(s['Tsetpoints'][demo_sp_i - 1])
                     if demo_sp_i > 0 else t_sp_v)
        if t_start_v < 0:
            t_start_v = 0.0
        demo_t_start = t_start_v
        demo_t_sp    = t_sp_v
        demo_n_st    = n_st_v
        print("Core 0: [Demo] Phase", demo_sp_i,
              "T", t_start_v, "->", t_sp_v, "N", n_st_v,
              "(quench)" if demo_phase_quench else "")

    # Compute linear ramp temperature for this frame
    frac        = min(1.0, steps_done / demo_n_st) if demo_n_st > 0 else 1.0
    target_temp = demo_t_start + (demo_t_sp - demo_t_start) * frac

    # Phase complete?
    if steps_done >= demo_n_st:
        if demo_phase_quench:
            quench_requested  = True
            demo_phase_quench = False
        demo_sp_i    += 1
        steps_done    = 0
        demo_t_start  = -1.0
        if demo_sp_i >= len(s['Tsetpoints']):
            # Setting complete → advance to the next one
            request_demo_sim((demo_s_idx + 1) % len(settings))

    return 0


# ============================================================================
# BUTTON HANDLING  (Core 0)
# ============================================================================

# Persistent B-button state (survives across handle_buttons calls)
_b_was_pressed          = False
_b_ignore_until_release = False
_last_button_time       = 0


def handle_buttons(current_time, display_mode, current_option,
                   quench_pending, quench_pending_until):
    """
    Poll all buttons, apply debounce, and update shared state.

    Parameters:
        current_time          — ticks_ms() for this frame
        display_mode          — current demo info view (0-2)
        current_option        — current manual parameter index
        quench_pending        — True if the two-step quench is armed
        quench_pending_until  — ticks_ms() deadline for the quench prompt

    Returns:
        (display_mode, current_option, quench_pending, quench_pending_until,
         skip_delta, do_shutdown)

    Side-effects on shared globals:
        demo_mode, target_temp, gravity, coupling,
        r_NBparticles, r_sizeBparticles, epsilonAB,
        physics_running, quench_requested, shutdown_requested
    """
    global demo_mode, target_temp, gravity, coupling
    global r_NBparticles, r_sizeBparticles, epsilonAB
    global physics_running, quench_requested, shutdown_requested
    global _b_was_pressed, _b_ignore_until_release, _last_button_time

    b_held     = button_b.read()
    a_held_now = button_a.read()
    skip_delta  = 0
    do_shutdown = False

    # -- A+B simultaneously → shutdown (highest priority) --
    if a_held_now and b_held:
        if time.ticks_diff(current_time, _last_button_time) > DEBOUNCE_MS:
            print("Core 0: A+B — shutting down")
            shutdown_requested = True
            do_shutdown        = True
        return (display_mode, current_option, quench_pending,
                quench_pending_until, skip_delta, do_shutdown)

    # -- B button: mode switching --
    if _b_ignore_until_release:
        if not b_held:
            _b_ignore_until_release = False
            _b_was_pressed          = False
    else:
        if b_held:
            _b_was_pressed = True
            if demo_mode:
                # Sync manual globals from the current setting so displayed
                # values match what the sim is already doing.
                _s               = settings[demo_s_idx]
                gravity          = _s['gravity']
                coupling         = _s['coupling']
                r_NBparticles    = _s['r_NBparticles']
                r_sizeBparticles = _s['r_sizeBparticles']
                epsilonAB        = _s['epsilonAB']
                current_option   = 0
                quench_pending   = False
                demo_mode        = False
                _b_ignore_until_release = True
                print("Core 0: → Manual mode")
        elif _b_was_pressed:
            # B released → return to Demo
            _b_was_pressed = False
            if not demo_mode:
                demo_mode               = True
                _b_ignore_until_release = True
                quench_pending          = False
                print("Core 0: → Demo mode")

    # -- A / X / Y: debounced; skipped while B-release guard is active --
    if (_b_ignore_until_release or
            time.ticks_diff(current_time, _last_button_time) <= DEBOUNCE_MS):
        return (display_mode, current_option, quench_pending,
                quench_pending_until, skip_delta, do_shutdown)

    if demo_mode:
        if overheated:
            # Any button restarts the current setting from the beginning
            if a_held_now or button_x.read() or button_y.read():
                request_demo_sim(demo_s_idx)
                _last_button_time = current_time
        else:
            if a_held_now:
                display_mode      = (display_mode + 1) % N_DISPLAY_MODES
                _last_button_time = current_time
            elif button_x.read():
                skip_delta        = 1
                _last_button_time = current_time
            elif button_y.read():
                skip_delta        = -1
                _last_button_time = current_time

    else:   # Manual mode
        if overheated:
            # Any button creates a fresh sim with current params at T=1.0
            if a_held_now or button_x.read() or button_y.read():
                request_manual_sim()
                _last_button_time = current_time
        else:
            x_pressed = button_x.read()
            y_pressed = button_y.read()

            if a_held_now:
                n_opts         = N_MANUAL_OPTIONS if EXPERT_MODE else N_BASIC_MANUAL
                current_option = (current_option + 1) % n_opts
                _last_button_time = current_time

            elif x_pressed:
                if   current_option == 0: target_temp      += 0.1
                elif current_option == 1: gravity          += 0.001
                elif current_option == 2: physics_running   = not physics_running
                elif current_option == 3: coupling         += 0.005
                elif current_option == 4: r_NBparticles     = min(1.0, r_NBparticles + 0.01)
                elif current_option == 5: r_sizeBparticles += 0.05
                elif current_option == 6: epsilonAB        += 0.05
                quench_pending    = False
                _last_button_time = current_time

            elif y_pressed:
                if current_option == 0:
                    if target_temp > 0:
                        target_temp    = max(0.0, target_temp - 0.1)
                        quench_pending = False
                    elif quench_pending:
                        quench_requested = True
                        quench_pending   = False
                    else:
                        quench_pending       = True
                        quench_pending_until = current_time + 2000
                elif current_option == 1: gravity          -= 0.001
                elif current_option == 2: physics_running   = not physics_running
                elif current_option == 3: coupling          = max(0.0, coupling - 0.005)
                elif current_option == 4: r_NBparticles     = max(0.0, r_NBparticles - 0.01)
                elif current_option == 5: r_sizeBparticles  = max(0.5, r_sizeBparticles - 0.05)
                elif current_option == 6: epsilonAB         = max(0.0, epsilonAB - 0.05)
                _last_button_time = current_time

    return (display_mode, current_option, quench_pending,
            quench_pending_until, skip_delta, do_shutdown)


# ============================================================================
# CORE 0: DISPLAY + BUTTON INPUT + DEMO SEQUENCER
# ============================================================================

def main():
    """
    Runs on Core 0 (main thread).

    Each frame:
      1. handle_buttons() — read inputs, update shared globals, return UI state.
      2. update_demo_sequencer() — advance phase, compute target_temp (demo only).
      3. Render:
           a. Clear screen.
           b. Draw particles.
           c. Draw gradient progress bar (demo mode).
           d. Draw calibration bar (manual, paused, accel uncalibrated).
           e. build_display_fields() + render_overlay() — text overlay.
      4. display.update() — flush to screen.
    """
    global running, shutdown_requested

    gc.collect()

    # Local UI state
    display_mode         = 0     # demo: which info view (0=Tsp, 1=LJ, 2=dynamics)
    current_option       = 0     # manual: which parameter is selected
    quench_pending       = False
    quench_pending_until = 0
    skip_delta           = 0

    print("Core 0: Waiting for simulation on Core 1...")
    while sim is None:
        time.sleep(0.01)
    print("Core 0: Ready, starting render loop")

    frame = 0

    while running and not shutdown_requested:
        current_time = time.ticks_ms()

        # -- Button input --
        (display_mode, current_option,
         quench_pending, quench_pending_until,
         skip_delta, do_shutdown) = handle_buttons(
            current_time, display_mode, current_option,
            quench_pending, quench_pending_until)

        if do_shutdown:
            break

        # -- Demo sequencer: advance phase, update target_temp --
        if demo_mode and not overheated:
            skip_delta = update_demo_sequencer(skip_delta)

        # -- Render --
        display.set_pen(BG)
        display.clear()

        # Particles (both modes)
        try:
            px   = sim.pos_x
            py   = sim.pos_y
            sz_a = int(sim.atom_sizeA)
            sz_b = int(sim.atom_sizeB)
            display.set_pen(ATOM_A_PEN)
            for i in range(sim.n_atomsA):
                display.circle(int(px[i]), int(py[i]), sz_a)
            display.set_pen(ATOM_B_PEN)
            for i in range(sim.n_atomsA, sim.n_atoms):
                display.circle(int(px[i]), int(py[i]), sz_b)
        except Exception:
            pass   # skip frame on race condition during sim transitions

        # Gradient progress bar (demo mode only)
        if demo_mode:
            draw_progress_bar()

        # Calibration bar (manual, paused, accel not yet calibrated)
        if not demo_mode and not physics_running and accel.connected and not accel.calibrated:
            draw_calibration_bar(120, 5, 195)

        # Text overlay
        fields = build_display_fields(display_mode, current_option,
                                      quench_pending, quench_pending_until,
                                      current_time)
        render_overlay(fields)

        if frame % 100 == 0:
            gc.collect()

        display.update()
        frame += 1

    # -- Shutdown screen --
    print("\nCore 0: Display loop exiting")
    display.set_pen(BG)
    display.clear()
    display.set_pen(TEXT_PEN)
    display.text("SHUTDOWN", WIDTH // 2 - 40, HEIGHT // 2 - 10, scale=2)
    display.update()
    led.set_rgb(0, 0, 0)
    time.sleep(0.5)
    print("Core 0: Shutdown complete")


# ============================================================================
# STARTUP
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MD-IN-A-BOX — BINARY LJ SIMULATION")
    print("=" * 60)
    print("Core 0: Display + sequencer  |  Core 1: Physics")
    print("Demo:   A=info  X=next  Y=prev  B=manual")
    print("Manual: A=option  X/Y=adjust  B=auto  A+B=exit")
    print("=" * 60)

    request_demo_sim(0)   # pre-fill pending_sim_config before Core 1 starts

    _thread.start_new_thread(physics_thread, ())
    time.sleep(0.5)       # allow Core 1 to create the first simulation
    main()

    print("\n" + "=" * 60)
    print("Shutdown complete - safe to restart or exit REPL")
    print("=" * 60)
