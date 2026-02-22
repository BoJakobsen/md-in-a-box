# MD in a Box

Interactive 2D molecular dynamics simulation running on a Raspberry Pi Pico 2.
Based on the browser-based MD simulation by Ulf R. Pedersen: https://urp.dk/md/

---

## Physics

Atoms are modelled as classical particles interacting via pair forces, following
Newton's laws of motion. A Langevin thermostat maintains the temperature.


## Hardware

| Component | Details |
|-----------|---------|
| Raspberry Pi Pico 2 | RP2350, dual-core ARM Cortex-M33 @ 150 MHz, 520 KB RAM |
| Pico Display Pack 2.0 | 320×240 pixel display (Pimoroni) |
| BNO055 9-axis IMU | Gravity vector + linear acceleration (DFRobot breakout) |

Total hardware cost: ~€65

---

## Architecture

The simulation uses both CPU cores:

- **Core 0** — Display rendering (PicoGraphics) and button input
- **Core 1** — Physics simulation, running C-accelerated MD steps continuously

Particle positions are stored in shared memory arrays read by Core 0 for drawing.
The accelerometer feeds tilt (gravity direction) and movement (box linear acceleration) into the simulation.

---

## Performance

The MD force loop and Leap Frog integrator are implemented as a custom C
extension to ulab (`user.c`), called from MicroPython. This gives a large
speedup over pure Python/ulab:

| N atoms | ulab (MDPS) | ulab + C (MDPS) | Speedup |
|---------|------------|-----------------|---------|
| 16      | 78         | 12,674          | ~160×   |
| 25      | 35         | 6,816           | ~195×   |
| 36      | 18         | 3,960           | ~220×   |
| 64      | 5          | 1,553           | ~310×   |
| 100     | —          | 748             | —       |

*Measured on Pico 2 as pure MD run, with no graphics or physical input.

In the full demo (display + accelerometer + 100 atoms): ~500 MD steps/second.

---

## Controls

| Button | Action |
|--------|--------|
| A | Cycle options: Temperature → Gravity → Pause → Coupling → Box Vel scale |
| X | Increase selected parameter |
| Y | Decrease selected parameter |
| X + Y | Save screenshot (PPM file) |
| Hold B (1 sec) | Shutdown |

Physical interaction via the accelerometer:
- **Tilt** the device → gravity follows orientation
- **Shake** it → box walls move, moving the atoms, leading to heating

---

## Files

### Simulation cores

| File | Description |
|------|-------------|
| `md_sim_ulab_numpy.py` | Pure Python/ulab simulation class (no C), runs on MicroPython with ulab and standard Python with numpy |
| `md_sim_ulab_user_c_func.py` | Simulation class using C-accelerated force loop integrated with ulab|
| `user.c` | Custom ulab C extension for ulab: Leap Frog integrator + LJ forces |

### Pico applications

| File | Description |
|------|-------------|
| `run_show_sim_ulab_user_c_func_dual_core_pico_dislay.py` | Main demo: dual-core display + physics |
| `run_test_ulab_user_c_func.py` | Benchmark (no display) — C version |
| `run_test_ulab_user_c_func_pico_display.py` | Benchmark with display output — C version |
| `run_test_ulab_numpy.py` | Benchmark (no display) — pure ulab / numpy version |
| `bno055_handler.py` | BNO055 accelerometer wrapper |

### Desktop

| File | Description |
|------|-------------|
| `run_show_sim_numpy_pc.py` | Desktop visualization with optional matplotlib plot |

---

## Building the Firmware

### MicroPython / ulab / display

The C extension requires a custom MicroPython + ulab build.

`user.c` should be placed in the ulab folder:
```
ulab/code/user/user.c
```
and enabled it by setting `ULAB_HAS_USER_MODULE` in `ulab.h`.

For the full integration with pico display 2, pimoroni modules are also needed.
Easiest build strategy is to use a modified version of the `pimoroni-pico-rp2350` build. 

Pimoroni provides a CI setup at:
```
https://github.com/pimoroni/pimoroni-pico-rp2350.git
```

The script
```
setup_and_build_firmware.sh
```
utilizes this CI setup to perform a build (as of February 2026), might be broken in the future. 


### BNO055 accelerometer support 
For supporting the accelerometer the `micropython-bno055` library must be added to the device.

The library is found at:
```
https://github.com/micropython-IMU/micropython-bno055
```

---

## Quick Start

```bash
# Desktop test (numpy / matplotlib version )
python3 run_show_sim_numpy_pc.py

# Upload to Pico
mpremote fs cp md_sim_ulab_user_c_func.py :
mpremote fs cp bno055_handler.py :
mpremote fs cp run_show_sim_ulab_user_c_func_dual_core_pico_dislay.py :

# Run on Pico
mpremote run run_show_sim_ulab_user_c_func_dual_core_pico_dislay.py
```

---

## References

- Ulf R. Pedersen's browser MD simulation: https://urp.dk/md/
- ulab (numpy for MicroPython): https://github.com/v923z/micropython-ulab
- Pimoroni Pico libraries: https://github.com/pimoroni/pimoroni-pico
- Micropython-bno055: https://github.com/micropython-IMU/micropython-bno055
---

## License

MIT — see `LICENSE`.

## Authors

Bo Jakobsen, IMFUFA, Roskilde University (boj@ruc.dk)
Based on the browser-based MD simulation by Ulf R. Pedersen (urp@ruc.dk)
