"""The arm daemon's IK, with the shoulder offset taken from the active MJCF."""

import collections

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .policy import onnx_session

DT = 1.0 / 60.0
SAMPLES = (1, 2, 4, 6, 8)  # earlier targets the network sees, in DT
HISTORY_RESET_S = 0.25  # a longer gap between solves restarts the history and the speed cap
MAX_SPEED = np.array([0.25] + [15.0] * 6)  # lift m/s, then rad/s
# j1 in the frame the network was trained in, base coordinates.
TRAINING_J1 = {
    "l": np.array([-0.007939332, 0.113248292, 1.222503448]),
    "r": np.array([-0.007939267, -0.113247529, 1.222502847]),
}


class ArmIK:
    """One arm: targets are the {side}_eef site pose in base coordinates (xyz, quat xyzw)."""

    def __init__(self, model, side, session=None):
        self.session = session or onnx_session("ik")
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        base = model.body("base").id
        j1 = model.joint(f"{side}j1").id
        anchor = data.xmat[base].reshape(3, 3).T @ (data.xanchor[j1] - data.xpos[base])
        self.offset = anchor - TRAINING_J1[side]
        joints = [model.joint(f"{side}j{i}").id for i in range(7)]
        self.limits = model.jnt_range[joints]
        self.left = np.array([float(side == "l")], np.float32)
        self.buffer = collections.deque(maxlen=256)  # (t, position, rotation)
        self.reset(np.zeros(7))

    def reset(self, joints):
        """Forget the target history; joints are held if the next target is non-finite."""
        self.buffer.clear()
        self.u = None
        self.last_q, self.last_t = np.array(joints[:7], float), None

    def _at(self, t):
        b = self.buffer
        if t <= b[0][0]:
            return b[0][1], b[0][2]
        for (t0, p0, R0), (t1, p1, R1) in zip(reversed(list(b)[:-1]), reversed(list(b)[1:])):
            if t0 <= t <= t1:
                s = (t - t0) / max(t1 - t0, 1e-6)
                step = Rotation.from_matrix(R0.T @ R1).as_rotvec()
                return p0 + s * (p1 - p0), R0 @ Rotation.from_rotvec(s * step).as_matrix()
        return b[-1][1], b[-1][2]

    def solve(self, position, quaternion, t):
        p = np.asarray(position, float)
        quat = np.asarray(quaternion, float)
        if not (np.isfinite(p).all() and np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-6):
            return self.last_q
        R = Rotation.from_quat(quat).as_matrix()
        p = p - self.offset
        if self.buffer and t - self.buffer[-1][0] > HISTORY_RESET_S:
            self.buffer.clear()
        fresh = not self.buffer or self.u is None
        self.buffer.append((t, p, R))
        poses = [(p, R)] + [self._at(t - k * DT) for k in SAMPLES]
        pose = np.array([np.r_[pk, Rk.ravel()] for pk, Rk in poses], np.float32)[None]
        q, u = self.session.run(
            None,
            dict(
                pose=pose,
                u=(np.zeros((1, 7)) if self.u is None else self.u).astype(np.float32),
                fresh=np.array([float(fresh)], np.float32),
                left=self.left,
            ),
        )
        q = q[0].astype(float)
        self.u = u if np.isfinite(q).all() else None
        if not np.isfinite(q).all():
            return self.last_q
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        if self.last_t is not None and t - self.last_t < HISTORY_RESET_S:
            step = MAX_SPEED * max(t - self.last_t, 1e-3)
            q = np.clip(q, self.last_q - step, self.last_q + step)
        self.last_q, self.last_t = q, t
        return q
