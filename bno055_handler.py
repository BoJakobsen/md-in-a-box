# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)


import machine
import math


class BNO055Handler:
    """
    Wrapper class for BNO055 accelerometer.

    Checks for library availability and chip connection on init.
    Provides simplified access to acceleration and gravity data.

    Requires the BNO055 MicroPython library
    (https://github.com/micropython-IMU/micropython-bno055)
    """

    # For this project we use non default pins
    DEFAULT_SDA_PIN = 0
    DEFAULT_SCL_PIN = 1
    DEFAULT_I2C_ID = 0
    DEFAULT_ADDRESS = 0x28

    def __init__(self, sda_pin=None, scl_pin=None, i2c_id=None, address=None,
                 axis_map=(0, 1, 2), axis_sign=(1, 1, 1)):
        """
        Initialize the BNO055 handler.

        Args:
            sda_pin: I2C SDA pin number (default: 0)
            scl_pin: I2C SCL pin number (default: 1)
            i2c_id: I2C bus ID (default: 0)
            address: I2C address (default: 0x28)
            axis_map: Which chip axis maps to each output axis.
                      (0,1,2) = no change; (1,0,2) = swap x and y.
                      axis_map[i] is the chip-axis index for output axis i.
            axis_sign: Sign for each output axis after remapping, e.g. (-1,1,1).
                       Applied identically to accel, gravity, and lin_acc so
                       they remain consistent with each other.
        """
        self.sda_pin = sda_pin if sda_pin is not None else self.DEFAULT_SDA_PIN
        self.scl_pin = scl_pin if scl_pin is not None else self.DEFAULT_SCL_PIN
        self.i2c_id = i2c_id if i2c_id is not None else self.DEFAULT_I2C_ID
        self.address = address if address is not None else self.DEFAULT_ADDRESS

        self._imu = None
        self._i2c = None
        self._BNO055 = None

        self._axis_map = axis_map
        self._axis_sign = axis_sign
        
        # Step 1: Check if library is available
        self._check_library()

        # Step 2: Try to connect to chip
        self._connect()

    def _check_library(self):
        """Check if the bno055 library is available."""
        try:
            from bno055 import BNO055
            self._BNO055 = BNO055
        except ImportError:
            print( "bno055 library not found. ")
            print("Please copy bno055.py and bno055_base.py to the device.")
            return  # Early exit no need to continue.

    def _connect(self):
        """Attempt to connect to the BNO055 chip."""
        # Create I2C bus
        try:
            self._i2c = machine.I2C(
                self.i2c_id,
                sda=machine.Pin(self.sda_pin),
                scl=machine.Pin(self.scl_pin)
            )
        except Exception as e:
            print(f"Failed to create I2C bus: {e}")
            return  # Early exit no need to continue.

        # Check if device is on the I2C bus
        devices = self._i2c.scan()
        if self.address not in devices:
            if devices:
                print(f"BNO055 not found at address 0x{self.address:02X}.")
                print(f"Found devices at: {[hex(d) for d in devices]}")
                return  # Early exit no need to continue.
            else:
                print("No I2C devices found. Check wiring and connections.")
                return  # Early exit no need to continue.

        # Try to initialize the BNO055
        try:
            self._imu = self._BNO055(self._i2c,
                                     address=self.address)
        except RuntimeError as e:
            print(f"Failed to initialize BNO055: {e}")
            return  # Early exit no need to continue.
        except OSError as e:
            print(f"I2C communication error: {e}")
            return  # Early exit no need to continue.

    @property
    def connected(self):
        """Return True if connected to BNO055."""
        return self._imu is not None

    @property
    def calibrated(self):
        """Return True if the sensor is calibrated."""
        if not self._imu:
            return False
        return self._imu.calibrated()

    def cal_status(self):
        """
        Return calibration status.

        Returns:
            tuple: (sys, gyro, accel, mag) each 0-3, where 3 is fully calibrated
        """
        if not self._imu:
            return (0, 0, 0, 0)
        return tuple(self._imu.cal_status())

    def _remap(self, raw):
        """Apply axis_map then axis_sign to a raw (x, y, z) tuple."""
        mapped = (raw[self._axis_map[0]],
                  raw[self._axis_map[1]],
                  raw[self._axis_map[2]])
        return (mapped[0] * self._axis_sign[0],
                mapped[1] * self._axis_sign[1],
                mapped[2] * self._axis_sign[2])

    def accel(self):
        """
        Get acceleration vector in m/s^2.

        Returns:
            tuple: (x, y, z) acceleration
        """
        return self._remap(self._imu.accel())

    def gravity(self):
        """
        Get gravity vector in m/s^2.

        Returns:
            tuple: (x, y, z) gravity direction
        """
        gx, gy, gz  = self._remap(self._imu.gravity())
        # Sign swap required for correct simulation behaviour.
        # TODO: check BNO055 datasheet / coordinate convention to confirm why.
        return (-gx, -gy, -gz)
    
    def lin_acc(self):
        """
        Get linear acceleration (accel minus gravity) in m/s^2.

        Returns:
            tuple: (x, y, z) linear acceleration
        """
        return self._remap(self._imu.lin_acc())

    def euler(self):
        """
        Get Euler angles in degrees.

        Returns:
            tuple: (heading, roll, pitch)
        """
        return self._imu.euler()

    def temperature(self):
        """
        Get temperature in Celsius.

        Returns:
            int: Temperature
        """
        return self._imu.temperature()

    def get_normalize_gravity(self):
        """
        Get gravity direction suitable for MD simulation.

        Converts gravity vector to normalized x, y, z components
        that can be passed to sim.set_gravity().

        Returns:
            tuple: (gx, gy, gz) gravity components for simulation
        """
        gx, gy, gz = self.gravity()
        if math.isnan(gx) or math.isnan(gy):
            gx = 0
            gy = 0
        # Normalize to roughly -1 to 1 range (gravity is ~9.8 m/s^2)
        return (gx / 9.8, gy / 9.8, gz / 9.8)


# Simple test when run directly
if __name__ == "__main__":
    import time

    print("Testing BNO055Handler...")

    handler = BNO055Handler()
    if handler.connected:
        print("SUCCESS: Connected to BNO055")

        print("\nWaiting for calibration...")
        while not handler.calibrated:
            status = handler.cal_status()
            print(f"  Calibration: sys={status[0]} gyro={status[1]} accel={status[2]} mag={status[3]}")
            time.sleep(0.5)

        print("\nCalibrated! Reading data...")
        for _ in range(20):
            accel = handler.accel()
            gravity = handler.gravity()
            norm_g = handler.get_normalize_gravity()
            print(f"Accel: ({accel[0]:6.2f}, {accel[1]:6.2f}, {accel[2]:6.2f})  "
                  f"Gravity (norm): ({norm_g[0]:5.2f}, {norm_g[1]:5.2f})")
            time.sleep(0.25)

    else:
        print("Some error happened")

