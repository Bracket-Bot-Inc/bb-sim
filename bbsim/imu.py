"""Balance IMU: where the chip sits and what it reads"""

import numpy as np

DPS = np.pi / 180
OFFSET = np.array([0.0259, 0.0004, 0.0083])  # chip centre from the axle midpoint, base frame, m
GYRO_NOISE = np.array([1.5e-3, 1.6e-3, 1.6e-3])  # rad/s per tick, 1 sigma
ACCEL_NOISE = np.array([0.012, 0.009, 0.009])  # m/s^2 per tick, 1 sigma
GYRO_TEMP_SLOPE = np.array([-0.040, 0.026, 0.004]) * DPS  # rad/s per deg C from calibration
GYRO_BIAS_REPEAT = 0.02 * DPS  # rad/s, 1 sigma, drawn once per reset
ACCEL_BIAS = np.array([-0.02, 0.0, 0.03])  # m/s^2


class ImuSensor:
    """Turns the true rates and specific force at the chip into what the board reads."""

    def __init__(self, temp_offset=0.0, gyro_bias=(0.0, 0.0, 0.0), seed=0):
        """temp_offset: deg C the chip is from where its gyro was calibrated. gyro_bias: extra, rad/s."""
        self.offset = OFFSET
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = (
            np.array(gyro_bias, float)
            + GYRO_TEMP_SLOPE * temp_offset
            + self.rng.normal(0.0, GYRO_BIAS_REPEAT, 3)
        )

    def __call__(self, gyro, accel):
        gyro = gyro + self.gyro_bias + self.rng.normal(0.0, GYRO_NOISE)
        accel = accel + ACCEL_BIAS + self.rng.normal(0.0, ACCEL_NOISE)
        return gyro, accel
