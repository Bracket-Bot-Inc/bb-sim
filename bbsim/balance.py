"""Balance policy observations and action filtering."""

import mujoco
import numpy as np

from .config import ARM_JOINTS, DT, WHEELS
from .policy import HallEstimator, onnx_session

GRAVITY = 9.80665
TORQUE_NM = 5.5
ACTION_HISTORY_SCALE = 5.5 / 6.5
WHEEL_SIGNS = np.array([-1.0, 1.0])  # forward-positive [left, right]
WHEEL_RADIUS = 0.0825
CMD_INT_CLAMP = 0.1
IMU_OFFSET = np.array([0.04, 0.0, 0.0])  # from the axle, base frame
ATTITUDE_GAIN = 0.012
ATTITUDE_GATE = 0.25
AUX_COLUMNS = [2, 7, 0, 1, 15, 16, 31, 75]
AUX_TICK = 10
TURN_FILTER = (0.4182754168, -0.3573573609, -0.9390819441)
ARM_POSITION_SCALE = np.tile(
    [
        1.0,
        0.49822416967897676,
        0.6740679942715567,
        0.49822416967897676,
        0.520870722846203,
        0.6740679942715567,
        0.6740679942715567,
    ],
    2,
)
ARM_VELOCITY_SCALE = np.tile([2.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 2)


def specific_force(rotation, velocity, previous):
    """Body-frame accelerometer reading from a world-frame velocity difference."""
    return rotation.T @ ((velocity - previous) / DT + [0.0, 0.0, GRAVITY])


class AttitudeFilter:
    def __init__(self, quat):
        self.q = np.asarray(quat, float) / np.linalg.norm(quat)

    def update(self, gyro, accel):
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        qd = 0.5 * np.array(
            [
                -q1 * gx - q2 * gy - q3 * gz,
                q0 * gx + q2 * gz - q3 * gy,
                q0 * gy - q1 * gz + q3 * gx,
                q0 * gz + q1 * gy - q2 * gx,
            ]
        )
        norm = np.linalg.norm(accel)
        if norm > 1e-6 and abs(norm - GRAVITY) <= ATTITUDE_GATE:
            ax, ay, az = accel / norm
            f1 = 2 * (q1 * q3 - q0 * q2) - ax
            f2 = 2 * (q0 * q1 + q2 * q3) - ay
            f3 = 2 * (0.5 - q1 * q1 - q2 * q2) - az
            s = np.array(
                [
                    -2 * q2 * f1 + 2 * q1 * f2,
                    2 * q3 * f1 + 2 * q0 * f2 - 4 * q1 * f3,
                    -2 * q0 * f1 + 2 * q3 * f2 - 4 * q2 * f3,
                    2 * q1 * f1 + 2 * q2 * f2,
                ]
            )
            qd -= ATTITUDE_GAIN * s / max(np.linalg.norm(s), 1e-12)
        self.q = self.q + qd * DT
        self.q /= np.linalg.norm(self.q)

    def pitch(self):
        q0, q1, q2, q3 = self.q
        g0 = 2 * (q1 * q3 - q0 * q2)
        g1 = 2 * (q0 * q1 + q2 * q3)
        g2 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
        return float(np.arctan2(-g0, np.hypot(g1, g2)))


class StateEstimator:
    def __init__(self):
        self.session = onnx_session("balance_state")
        self.state = np.zeros((1, self.session.get_inputs()[0].shape[1]), np.float32)
        self.reset = True

    def __call__(self, pitch, gyro, accel, wheel, velocity, action):
        inputs = dict(gyro=gyro, accel=accel, wheel=wheel, vel=velocity, action=action)
        feed = {k: np.asarray(v, np.float32).reshape(1, -1) for k, v in inputs.items()}
        feed.update(
            state=self.state,
            pitch_in=np.array([pitch], np.float32),
            reset=np.array([float(self.reset)], np.float32),
        )
        self.state, pitch, speed, _, _ = self.session.run(None, feed)
        self.reset = False
        return float(pitch[0]), float(speed[0])


class AuxEstimator:
    def __init__(self):
        self.session = onnx_session("balance_aux")
        self.h = np.zeros((1, 32), np.float32)
        self.sum = np.zeros(11)
        self.count = 0
        self.estimate = 0.0
        self.previous_wheels = None

    def __call__(self, obs, accel_x):
        wheels = obs[:2]
        wheel_acc = (
            np.zeros(2) if self.previous_wheels is None else (wheels - self.previous_wheels) / DT
        )
        self.previous_wheels = wheels.copy()
        self.sum += np.r_[obs[AUX_COLUMNS], accel_x, wheel_acc]
        self.count += 1
        if self.count == AUX_TICK:
            x = (self.sum / self.count).astype(np.float32)[None]
            estimate, self.h = self.session.run(None, dict(x=x, h=self.h))
            self.estimate = float(estimate[0])
            self.sum[:] = 0
            self.count = 0


class ArmAngle:
    """Upright pitch shift (deg) the arm pose causes on this model, relative to zero arms."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base = model.body("base").id
        self.wheels = [model.body(f"{side}_wheel").id for side in WHEELS]
        self.joints = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.zero = self.theta(np.zeros(14))

    def theta(self, arms):
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qpos[self.joints] = arms
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        axle = d.xpos[self.wheels].mean(0)
        com = d.xmat[self.base].reshape(3, 3).T @ (d.subtree_com[self.base] - axle)
        return -np.rad2deg(np.arctan2(com[0], com[2]))

    def __call__(self, arms):
        return self.theta(arms) - self.zero


class TurnFilter:
    """Filters the differential (turn) part of the action; the common part passes through."""

    def __init__(self):
        self.z = 0.0

    def __call__(self, action):
        b0, b1, a1 = TURN_FILTER
        s = 0.5 * (action[0] + action[1])
        x = 0.5 * (action[1] - action[0])
        d = b0 * x + self.z
        self.z = b1 * x - a1 * d
        return np.clip([s - d, s + d], -1, 1)


class BalanceObservations:
    def __init__(self, model, arms):
        self.arm_angle = ArmAngle(model) if arms else None
        d = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d)
        base = model.body("base").id
        axle = d.xpos[[model.body(f"{side}_wheel").id for side in WHEELS]].mean(0)
        self.imu = d.xmat[base].reshape(3, 3).T @ (axle - d.xpos[base]) + IMU_OFFSET
        self.hall = HallEstimator()
        self.estimator = StateEstimator()
        self.aux = AuxEstimator()
        self.attitude = None
        self.previous = None  # last base / IMU-point world velocities
        self.actions = np.zeros((3, 2))
        self.applied = np.zeros(2)
        self.arm_position = np.zeros(14)
        self.arm_velocity = np.zeros((3, 14))
        self.pitch_rates = np.zeros(5)
        self.late = None  # previous tick's sensor terms (1-step lag)
        self.cmd_int = 0.0
        self.step = 0

    def build(
        self,
        rotation,
        quat,
        velocity,
        gyro,
        wheel_pos,
        wheel_vel,
        command,
        arm_position,
        arm_velocity,
        ideal_wheels=False,
    ):
        """rotation / quat: base attitude; velocity: base world velocity; gyro: base-frame rates."""
        first = self.step == 0
        wheel = self.hall(wheel_pos)
        if ideal_wheels:
            wheel = wheel_vel
        imu_velocity = velocity + np.cross(rotation @ gyro, rotation @ self.imu)
        if first:
            self.attitude = AttitudeFilter(quat)
            self.previous = velocity.copy(), imu_velocity.copy()
        accel = specific_force(rotation, velocity, self.previous[0])
        imu_accel = specific_force(rotation, imu_velocity, self.previous[1])
        self.previous = velocity.copy(), imu_velocity.copy()
        if not first:
            self.attitude.update(gyro, accel)
        pitch, speed = self.estimator(
            self.attitude.pitch(),
            gyro,
            imu_accel,
            wheel_pos * WHEEL_SIGNS,
            wheel * WHEEL_SIGNS,
            self.applied,
        )
        now = dict(wheel=wheel * [-0.05, 0.05], pitch=pitch * 4, rate=gyro[1], yaw=gyro[2] * 0.4)
        now["speed"] = speed * 4
        late, self.late = self.late or now, now
        if first:
            self.pitch_rates[:] = late["rate"]
        self.pitch_rates = np.r_[self.pitch_rates[1:], late["rate"]]
        if self.step % 4 == 0:
            self.arm_position[:] = arm_position
            self.arm_velocity = (
                np.tile(arm_velocity, (3, 1))
                if first
                else np.r_[self.arm_velocity[1:], [arm_velocity]]
            )
        odometry = 0.5 * WHEEL_RADIUS * np.sum(wheel * WHEEL_SIGNS)
        self.cmd_int = np.clip(
            self.cmd_int + (command[0] - odometry) * DT, -CMD_INT_CLAMP, CMD_INT_CLAMP
        )
        arm_angle = self.arm_angle(arm_position) if self.arm_angle else 0.0
        obs = np.r_[
            late["wheel"],
            late["pitch"],
            self.pitch_rates,
            late["yaw"],
            command * [4.0, 1.0],
            self.actions.ravel() * ACTION_HISTORY_SCALE,
            self.arm_position * ARM_POSITION_SCALE,
            self.cmd_int * 10,
            (self.arm_velocity * ARM_VELOCITY_SCALE).ravel(),
            late["speed"],
            arm_angle * 0.5,
            self.aux.estimate * 0.5,
        ]
        # Reads this tick's observation; a new estimate shows up from the next tick.
        self.aux(obs, accel[0])
        self.step += 1
        return obs.astype(np.float32)

    def remember_action(self, action):
        self.applied = np.asarray(action, float).copy()
        self.actions[:-1] = self.actions[1:]
        self.actions[-1] = action
