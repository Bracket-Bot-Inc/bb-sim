"""Balance policy observations and action filtering."""

import mujoco
import numpy as np

from .config import ARM_JOINTS, DT, WHEELS
from .imu import ImuSensor
from .policy import Actor, HallEstimator, onnx_session

GRAVITY = 9.80665
TORQUE_NM = 5.5
ACTION_HISTORY_SCALE = 5.5 / 6.5
WHEEL_SIGNS = np.array([-1.0, 1.0])  # forward-positive [left, right]
WHEEL_RADIUS = 0.0825
CMD_INT_CLAMP = 0.1
AUX_COLUMNS = [2, 7, 0, 1, 15, 16, 31, 75]
AUX_TICK = 10
TURN_FILTER = (0.4182754168, -0.3573573609, -0.9390819441)
LEAN_MAX_DEG = 15.0
LEAN_DEBOUNCE_TICKS = 5
LEAN_ABORT_PITCH = np.deg2rad(20.0)
LEAN_ABORT_TICKS = 60
LEAN_TRAVEL_M = 0.5
LEAN_FAST_MPS = 0.6
LEAN_FAST_MIN_M = 0.3
LEAN_FAST_TICKS = 10
LEAN_SPEED_TAU_S = 0.05
LEAN_RELEASE_TICKS = 200
LEAN_YAW_CAP = 0.3
HANDBACK_TICKS = 200
SWAP_GAP_TICKS = 20
SETTLED = (0.2, 0.4, 3.0)  # pitch rate rad/s, obs speed, balance-point angle deg
THETA0_DEG = 0.788988
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


class StateEstimator:
    """The pitch / speed estimator; its own pitch is the only pitch."""

    def __init__(self):
        self.session = onnx_session("balance_state")
        self.state = np.zeros((1, self.session.get_inputs()[0].shape[1]), np.float32)
        self.reset = True
        self.pitch = None

    def __call__(self, gyro, accel, wheel, velocity, action):
        if self.pitch is None:  # seed from the accelerometer's tilt (a still robot)
            self.pitch = float(np.arctan2(-accel[0], accel[2]))
        inputs = dict(gyro=gyro, accel=accel, wheel=wheel, vel=velocity, action=action)
        feed = {k: np.asarray(v, np.float32).reshape(1, -1) for k, v in inputs.items()}
        feed.update(
            state=self.state,
            pitch_in=np.array([self.pitch], np.float32),
            reset=np.array([float(self.reset)], np.float32),
        )
        self.state, _, speed, pitch, _ = self.session.run(None, feed)
        self.reset = False
        self.pitch = float(pitch[0])
        return self.pitch, float(speed[0])


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
    def __init__(self, model, arms, imu=None):
        self.arm_angle = ArmAngle(model) if arms else None
        d = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d)
        base = model.body("base").id
        axle = d.xpos[[model.body(f"{side}_wheel").id for side in WHEELS]].mean(0)
        self.sensor = imu or ImuSensor()
        self.imu = d.xmat[base].reshape(3, 3).T @ (axle - d.xpos[base]) + self.sensor.offset
        self.hall = HallEstimator()
        self.estimator = StateEstimator()
        self.aux = AuxEstimator()
        self.previous = None  # last base / IMU-point world velocities
        self.actions = np.zeros((3, 2))
        self.applied = np.zeros(2)
        self.arm_position = np.zeros(14)
        self.arm_velocity = np.zeros((3, 14))
        self.pitch_rates = np.zeros(5)
        self.late = None  # previous tick's sensor terms (1-step lag)
        self.cmd_int = 0.0
        self.accel_x = 0.0
        self.forward_speed = 0.0
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
        lean_input=0.0,
        ideal_wheels=False,
    ):
        """rotation / quat: base attitude; velocity: base world velocity; gyro: base-frame rates."""
        first = self.step == 0
        wheel = self.hall(wheel_pos)
        if ideal_wheels:
            wheel = wheel_vel
        imu_velocity = velocity + np.cross(rotation @ gyro, rotation @ self.imu)
        if first:
            self.previous = velocity.copy(), imu_velocity.copy()
        accel = specific_force(rotation, velocity, self.previous[0])
        imu_accel = specific_force(rotation, imu_velocity, self.previous[1])
        self.previous = velocity.copy(), imu_velocity.copy()
        gyro, imu_accel = self.sensor(gyro, imu_accel)
        pitch, speed = self.estimator(
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
        self.forward_speed = odometry
        self.accel_x = accel[0]
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
            lean_input,
        ]
        self.step += 1
        return obs.astype(np.float32)

    def remember_action(self, action):
        self.applied = np.asarray(action, float).copy()
        self.actions[:-1] = self.actions[1:]
        self.actions[-1] = action


class Controller:
    """One 5 ms base-board tick: lean state, actor selection, actor, turn filter."""

    def __init__(self, model, arms, imu=None):
        self.observations = BalanceObservations(model, arms, imu)
        self.balance = Actor("balance")
        self.lean = onnx_session("lean")
        self.lean_aux = onnx_session("lean_aux")
        self.turn_filter = TurnFilter()
        self.aux_h = np.zeros((1, 64), np.float32)
        self.aux_on = False
        self.aux_value = 0.0
        self.lean_active = False
        self.lean_target = False
        self.lean_angle = 0.0
        self.pending = self.abort_ticks = self.release_ticks = self.fast_ticks = 0
        self.latched_off = self.travel_latch = False
        self.travel_start = self.speed_f = 0.0
        self.lean_actor = False  # selected
        self.loaded = "balance"  # driving
        self.handback = self.swap_gap = 0

    def _lean_update(self, want, angle, pitch, travel, speed):
        if not want:
            self.latched_off = False
            if self.release_ticks < LEAN_RELEASE_TICKS:
                self.release_ticks += 1
            else:
                self.travel_latch = False
        else:
            self.release_ticks = 0
        self.lean_target = want and not self.latched_off and not self.travel_latch
        if self.lean_target != self.lean_active:
            self.pending += 1
            if self.pending >= LEAN_DEBOUNCE_TICKS:
                self.lean_active, self.pending = self.lean_target, 0
                if self.lean_active:
                    self.travel_start, self.speed_f, self.fast_ticks = travel, speed, 0
        else:
            self.pending = 0
        if not self.lean_active:
            self.abort_ticks, self.lean_angle = 0, 0.0
            return
        self.lean_angle = float(np.clip(angle, -LEAN_MAX_DEG, LEAN_MAX_DEG))
        if abs(pitch) > LEAN_ABORT_PITCH:
            self.abort_ticks += 1
            if self.abort_ticks >= LEAN_ABORT_TICKS:
                self.lean_active, self.latched_off = False, True
                self.abort_ticks = self.pending = 0
                self.lean_angle = 0.0
                return
        else:
            self.abort_ticks = 0
        moved = abs(travel - self.travel_start)
        self.speed_f += DT / LEAN_SPEED_TAU_S * (speed - self.speed_f)
        fast = abs(self.speed_f) > LEAN_FAST_MPS and moved > LEAN_FAST_MIN_M
        self.fast_ticks = self.fast_ticks + 1 if fast else 0
        if moved > LEAN_TRAVEL_M or self.fast_ticks >= LEAN_FAST_TICKS:
            self.lean_active, self.travel_latch = False, True
            self.fast_ticks = self.abort_ticks = self.pending = 0
            self.lean_angle = 0.0

    def _select(self, obs):
        if self.lean_target or self.lean_active:
            self.lean_actor, self.handback = True, 0
            return
        if not self.lean_actor:
            return
        if self.latched_off or self.travel_latch:
            self.lean_actor, self.handback = False, 0
            return
        rate, speed, angle = SETTLED
        phi = np.rad2deg(obs[2] / 4) - THETA0_DEG - 2 * obs[75] - 2 * obs[76]
        quiet = abs(obs[7]) < rate and abs(obs[74]) < speed and abs(phi) < angle
        self.handback = self.handback + 1 if quiet else 0
        if self.handback >= HANDBACK_TICKS:
            self.lean_actor, self.handback = False, 0

    def __call__(
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
        lean_request=False,
        lean_degrees=0.0,
        ideal_wheels=False,
    ):
        o = self.observations
        pitch = o.estimator.pitch  # the estimator's pitch from the last tick, as on the board
        if pitch is None:
            pitch = float(np.arctan2(-rotation[2, 0], rotation[2, 2]))
        travel = 0.5 * WHEEL_RADIUS * np.sum(wheel_pos * WHEEL_SIGNS)
        self._lean_update(lean_request, lean_degrees, pitch, travel, o.forward_speed)
        lean_input = self.lean_angle / LEAN_MAX_DEG if self.lean_active else 0.0
        obs = o.build(
            rotation,
            quat,
            velocity,
            gyro,
            wheel_pos,
            wheel_vel,
            command,
            arm_position,
            arm_velocity,
            lean_input,
            ideal_wheels,
        )
        yaw = obs[10]
        obs[10] = np.clip(yaw, -LEAN_YAW_CAP, LEAN_YAW_CAP) if self.lean_active else 0.0
        aux_on = self.loaded == "lean"
        self.aux_value = 0.0
        if aux_on:
            p, self.aux_h = self.lean_aux.run(
                None,
                dict(
                    obs=obs[None],
                    h=self.aux_h,
                    reset=np.array([float(not self.aux_on)], np.float32),
                ),
            )
            self.aux_value = float(p[0])
        self.aux_on = aux_on
        if not aux_on:
            obs[10] = yaw
        self._select(obs)
        if self.loaded == "lean":
            action = self.lean.run(
                None, dict(obs=obs[None], aux=np.array([self.aux_value], np.float32))
            )[0][0]
        else:
            action = self.balance(obs)
        if not self.lean_active:
            o.aux(obs, o.accel_x)
        action = self.turn_filter(np.clip(action, -1, 1))
        o.remember_action(action)
        self.swap_gap = max(self.swap_gap - 1, 0)
        want = "lean" if self.lean_actor else "balance"
        if want != self.loaded and self.swap_gap == 0:
            self.loaded, self.swap_gap = want, SWAP_GAP_TICKS
        return obs, action
