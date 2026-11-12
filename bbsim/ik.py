"""The arm IK: a 30 Hz closed-loop policy that steps a joint reference toward the target pose.

Each tick reads the measured joints and the target, and returns two goals: the half-way goal, sent at
once, and the new reference, sent HALF_S later. The left arm runs the right-arm policy mirrored.
"""

import json
import math

import mujoco
import numpy as np

from .policy import onnx_session

HALF_S = 6 / 11 / 30  # the new reference goes out this long after the half-way goal
# Command law gain beta = max(G(e_p), G(e_R)), G = BETA_MIN + (1 - BETA_MIN) clip((e - lo) / (hi - lo), 0, 1)
BETA_MIN = 0.1
POS_BAND = (0.0005, 0.00931)  # m
ROT_BAND = (math.radians(0.415), math.radians(2.328))  # rad
NEW_GOAL_P, NEW_GOAL_R = (
    0.01,
    math.radians(2.0),
)  # a target this far from the last goal is a new goal
# Servo position resolution (4096 counts per turn; the lift's 46.5 mm pinion).
ENC = np.array([2 * math.pi / 4096 * 0.0465] + [2 * math.pi / 4096] * 6)
MIRROR = np.array([1.0, 1, -1, 1, -1, -1, 1])  # left arm joints <-> the policy's (right arm) joints
# Where each arm's j0 frame sat in arm_base in the URDF the policy was trained on.
TRAINING_SHOULDER = {
    "r": np.array([7.84414955213e-09, -0.0274999999999, 1.36714092703]),
    "l": np.array([-7.84414955213e-09, 0.0274999999999, 1.36714107297]),
}


def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def _rotvec(R):
    """Rotation vector of R, robust near pi."""
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = math.atan2(s, c)
    if s > 1e-6:
        return v * (ang / s)
    if c >= 0:
        return v
    d = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
    k = int(np.argmax(d))
    a = (R + R.T)[k] / (4 * max(d[k], 1e-9))
    a[k] = d[k]
    return a / max(np.linalg.norm(a), 1e-9) * math.pi


def _quat_matrix(quat):
    x, y, z, w = quat
    n = x * x + y * y + z * z + w * w
    return (
        np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        / n
    )


class ArmIK:
    """One arm: targets are the {side}_eef site pose in base coordinates (xyz, quat xyzw)."""

    def __init__(self, model, side, session=None):
        self.session = session or onnx_session("ik")
        meta = json.loads(self.session.get_modelmeta().custom_metadata_map["arm"])
        self.steps = [
            (np.array(off), np.array(rot), idx, np.array(axis), _skew(axis))
            for off, rot, idx, axis in zip(
                meta["fk_off"], meta["fk_rot"], meta["fk_idx"], meta["fk_axis"]
            )
        ]
        self.rate = meta["rate"]
        self.eta, self.lead = np.array(meta["eta"]), np.array(meta["lead"])
        self.tol_p, self.tol_r = meta["tol_p"], meta["tol_r"]
        lo, hi = np.array(meta["lo"]), np.array(meta["hi"])
        self.q_nom = 0.5 * (lo + hi)
        self.sign = MIRROR if side == "l" else np.ones(7)
        self.left = side == "l"
        joints = [model.joint(f"{side}j{i}").id for i in range(7)]
        own = model.jnt_range[joints] * self.sign[:, None]
        self.lo = np.maximum(lo, own.min(1))
        self.hi = np.minimum(hi, own.max(1))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        base = model.body("base").id
        j0 = model.body(model.jnt_bodyid[joints[0]]).id
        shoulder = data.xmat[base].reshape(3, 3).T @ (data.xpos[j0] - data.xpos[base])
        self.offset = shoulder - TRAINING_SHOULDER[side]
        self.reset(np.zeros(7))

    def fk(self, q):
        """End-effector (p, R) of the policy's arm (model frame) at joints q."""
        R, p = np.eye(3), np.zeros(3)
        for off, rot, idx, axis, K in self.steps:
            p = p + R @ off
            R = R @ rot
            if idx == 0:
                p = p + (R @ axis) * q[0]
            elif idx > 0:
                R = R @ (np.eye(3) + math.sin(q[idx]) * K + (1 - math.cos(q[idx])) * (K @ K))
        return p, R

    def reset(self, joints):
        """Start a new segment from the measured joints (URDF j0..j6)."""
        q = np.asarray(joints[:7], float) * self.sign
        self.z, self.a_prev = q.copy(), np.zeros(7)
        self.q_m1 = self.q_m2 = q.copy()
        self.goal = None  # (p, R) of the current goal
        self.best = None  # the best normalised error reached on it

    def tick(self, joints, position, quaternion):
        """One policy tick from the measured joints and the target (base coordinates). Returns the
        half-way goal and the new reference (URDF j0..j6)."""
        gp = np.asarray(position, float) - self.offset
        gR = _quat_matrix(np.asarray(quaternion, float))
        if (
            self.left
        ):  # into the policy's (right arm) frame: p -> M p, R -> M R M, M = diag(1, -1, 1)
            gp[1] = -gp[1]
            gR = gR * np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        q = np.round(np.asarray(joints[:7], float) * self.sign / ENC) * ENC
        p, R = self.fk(q)
        e_p, e_r = float(np.linalg.norm(p - gp)), float(np.linalg.norm(_rotvec(gR.T @ R)))
        rho = max(e_p / self.tol_p, e_r / self.tol_r)
        new_goal = (
            self.goal is None
            or np.linalg.norm(gp - self.goal[0]) > NEW_GOAL_P
            or np.linalg.norm(_rotvec(self.goal[1].T @ gR)) > NEW_GOAL_R
        )
        best = rho if new_goal else self.best
        ep_ee, er_ee = R.T @ (gp - p), _rotvec(R.T @ gR)
        obs = np.concatenate(
            (
                q - self.q_nom,
                30 * (q - self.q_m1),
                30 * (self.q_m1 - self.q_m2),
                gp,
                gR[:, :2].T.reshape(6),
                self.z - q,
                np.clip(ep_ee / 0.01, -1, 1),
                np.clip(er_ee / math.radians(20), -1, 1),
                self.a_prev,
                [np.clip(best / 20, 0, 1), np.clip((rho - best) / 20, 0, 1)],
                ((gp[None] + 0.3 * gR.T) - (p[None] + 0.3 * R.T)).reshape(9),  # keypoints
                np.clip(ep_ee / (0.5 * self.tol_p), -1, 1),
                np.clip(er_ee / (1.25 * self.tol_r), -1, 1),
            )
        )
        if new_goal:
            self.goal = (gp.copy(), gR.copy())
        self.best = min(best, rho)
        (a,) = self.session.run(None, dict(obs=obs.astype(np.float32)[None]))
        a = a[0].astype(float)

        def gain(e, lo, hi):
            return BETA_MIN + (1 - BETA_MIN) * np.clip((e - lo) / (hi - lo), 0, 1)

        beta = max(gain(e_p, *POS_BAND), gain(e_r, *ROT_BAND))
        z = self.z + self.eta * beta * np.clip(a, -1, 1)
        z = np.clip(np.clip(z, self.lo, self.hi), q - self.lead, q + self.lead)
        half = self.z + 0.5 * (z - self.z)
        self.z, self.a_prev = z, a
        self.q_m2, self.q_m1 = self.q_m1, q
        return half * self.sign, z * self.sign
