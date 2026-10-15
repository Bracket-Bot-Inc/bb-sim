"""Policy inference, training observation layout, and deterministic wheel plant."""

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.special import expit

from .config import ASSETS, DT, POLICY_INPUTS


def onnx_session(name):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(ASSETS / f"policies/{name}.onnx"), opts, providers=["CPUExecutionProvider"]
    )


class Actor:
    def __init__(self, name):
        self.name = name
        self.size = POLICY_INPUTS[name]
        self.session = onnx_session(name)
        self.input = self.session.get_inputs()[0].name
        assert self.session.get_inputs()[0].shape == [1, self.size]

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.size)
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite policy observation")
        y = self.session.run(None, {self.input: x})[0].reshape(2)
        if not np.isfinite(y).all() or np.max(np.abs(y)) > 1.00001:
            raise FloatingPointError("Invalid policy action")
        return np.clip(y, -1, 1)


class WheelMotor:
    """Residual LSTM motor model; torques use the MuJoCo joint frame.

    BW100 models also carry the fitted Stribeck assist (kinetic friction below the joint's
    static frictionloss) and the mechanical torque clamp.
    """

    PARAMETERS = (
        "hid",
        "layers",
        "pos_scale",
        "vel_scale",
        "torque_scale",
        "frictionloss",
        "dc_sat",
        "dc_vlim",
        "dc_elim",
        "w_eps",
        "sign",
    )
    OPTIONAL = ("stick_gate_w_eps", "kinetic", "stribeck_velocity", "effort_limit")

    def __init__(self, model, side):
        numerics = {model.numeric(i).name for i in range(model.nnumeric)}
        self.config = {
            name: float(model.numeric(f"wheel_{side}_{name}").data[0])
            for name in self.PARAMETERS + self.OPTIONAL
            if name in self.PARAMETERS or f"wheel_{side}_{name}" in numerics
        }
        for name in ("hid", "layers"):
            value = self.config[name]
            if not value.is_integer() or value < 1:
                raise ValueError(f"wheel_{side}_{name} must be a positive integer")
            self.config[name] = int(value)
        # One net for both wheels when the model names it, else per-wheel nets.
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "wheel_model")
        adr = model.text_adr[i] if i >= 0 else 0
        weights = (
            model.text_data[adr : adr + model.text_size[i] - 1].decode()
            if i >= 0
            else f"wheel_{side}"
        )
        with np.load(ASSETS / f"motors/{weights}.npz") as z:
            self.weights = {k: z[k] for k in z.files}
        self.reset()

    def reset(self):
        c = self.config
        self.h = np.zeros((c["layers"], c["hid"]), np.float32)
        self.c = np.zeros_like(self.h)

    def residual(self, inputs):
        x = np.asarray(inputs, np.float32)
        w = self.weights
        for i in range(self.config["layers"]):
            p = f"cells.{i}."
            gates = (
                w[p + "weight_ih"] @ x
                + w[p + "bias_ih"]
                + w[p + "weight_hh"] @ self.h[i]
                + w[p + "bias_hh"]
            )
            ig, fg, gg, og = np.split(gates, 4)
            self.c[i] = expit(fg) * self.c[i] + expit(ig) * np.tanh(gg)
            self.h[i] = expit(og) * np.tanh(self.c[i])
            x = self.h[i]
        x = w["head.0.weight"] @ x + w["head.0.bias"]
        x = x / (1 + np.abs(x))
        return float((w["head.2.weight"] @ x + w["head.2.bias"])[0])

    def __call__(self, torque, velocity):
        c = self.config
        sign = c.get("sign", 1.0)
        tau, vel = torque * sign, velocity * sign
        residual = self.residual([tau * c["pos_scale"], vel * c["vel_scale"]]) * c.get(
            "torque_scale", 1.0
        )
        top = min(c["dc_sat"] * (1 - vel / c["dc_vlim"]), c["dc_elim"])
        bottom = max(c["dc_sat"] * (-1 - vel / c["dc_vlim"]), -c["dc_elim"])
        base = min(max(tau, bottom), top)
        if (
            abs(vel) < c.get("stick_gate_w_eps", c.get("w_eps", 0.15))
            and abs(base) < c["frictionloss"]
        ):
            residual = 0.0
        torque = (base + residual) * sign
        if "kinetic" in c:
            assist = (c["frictionloss"] - c["kinetic"]) * (
                1 - np.exp(-((velocity / c["stribeck_velocity"]) ** 2))
            )
            torque = np.clip(
                torque + assist * np.sign(velocity), -c["effort_limit"], c["effort_limit"]
            )
        return torque


class HallEstimator:
    """Training's interpolated 90-count/rev, 125-us 1/T wheel estimator."""

    def __init__(self):
        self.previous = None
        self.age = np.zeros(2)
        self.period = np.full(2, 2.0)
        self.direction = np.zeros(2)

    def __call__(self, radians):
        x = np.asarray(radians) * (90 / (2 * np.pi))
        if self.previous is None:
            self.previous = x.copy()
        dx = x - self.previous
        delta = np.round(x) - np.round(self.previous)
        edge = delta != 0
        direction = np.sign(dx)
        safe_dx = np.where(edge, dx, 1.0)
        first = np.clip((np.round(self.previous) + 0.5 * direction - self.previous) / safe_dx, 0, 1)
        last = np.clip((np.round(x) - 0.5 * direction - self.previous) / safe_dx, 0, 1)
        period = np.where(
            np.abs(delta) >= 2, DT / np.maximum(np.abs(dx), 1e-6), self.age + DT * first
        )
        period = np.minimum(np.maximum(np.round(period / 125e-6), 1) * 125e-6, 2.0)
        self.period = np.where(edge, period, self.period)
        self.direction = np.where(edge, direction, self.direction)
        self.age = np.where(edge, DT * (1 - last), self.age + DT)
        self.previous = x.copy()
        velocity = self.direction / np.maximum(self.period, self.age) * (2 * np.pi / 90)
        return np.where(np.abs(velocity) < 0.044, 0.0, velocity)


class Observations:
    """Lean actor input (the balance modes use balance.BalanceObservations)."""

    def __init__(self):
        self.pitch_rates = np.zeros(5)
        self.actions = np.zeros((3, 2))
        self.arc_error = 0.0
        self.step = 0
        self.hall = HallEstimator()

    def build(
        self,
        wheel_pos,
        wheel_vel,
        pitch,
        gyro,
        body_velocity,
        command,
        lean_degrees=3.0,
        ideal_wheels=False,
    ):
        # Histories are oldest -> newest, including the current gyro sample.
        if self.step == 0:
            self.pitch_rates[:] = gyro[1]
        self.pitch_rates[:-1] = self.pitch_rates[1:]
        self.pitch_rates[-1] = gyro[1]
        # Training ARC uses true body-frame forward velocity, with ±0.5 m clamp.
        if self.step:
            self.arc_error = np.clip(self.arc_error + (command[0] - body_velocity) * DT, -0.5, 0.5)
        velocity = wheel_vel if ideal_wheels else self.hall(wheel_pos)
        obs = np.r_[
            velocity * np.array([-0.05, 0.05]),
            pitch * 4,
            self.pitch_rates,
            gyro[2] * 0.4,
            command * np.array([4.0, 1.0]),
            self.actions.ravel(),
            self.arc_error * 2,
            lean_degrees / 3.0,
        ]
        self.step += 1
        return obs.astype(np.float32)

    def remember_action(self, action):
        self.actions[:-1] = self.actions[1:]
        self.actions[-1] = action
