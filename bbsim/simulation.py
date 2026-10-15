"""Local MuJoCo physics and policy execution."""

import mujoco
import numpy as np

from .balance import TORQUE_NM, BalanceObservations, TurnFilter
from .config import ARM_JOINTS, ASSETS, POLICIES, WHEELS
from .policy import Actor, Observations, WheelMotor


class Simulation:
    def __init__(
        self,
        kind="arms",
        velocity=0.0,
        yaw=0.0,
        lean_degrees=3.0,
        arm_motion=False,
        terrain="flat",
        ideal_wheels=False,
        start_pitch=0.0,
        wall_distance=0.32,
    ):
        self.kind = kind
        spec = mujoco.MjSpec.from_file(str(ASSETS / f"{kind}.xml"))
        if terrain == "bumps":
            for i in range(8):
                spec.worldbody.add_geom(
                    name=f"low_bump_{i}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.8 + i * 0.5, 0, 0.004],
                    size=[0.05, 1.5, 0.004],
                    contype=1,
                    conaffinity=1,
                    rgba=[0.36, 0.40, 0.45, 1],
                )
        elif terrain == "slope":
            floor = spec.geom("floor")
            floor.quat = [np.cos(np.deg2rad(3) / 2), 0, np.sin(np.deg2rad(3) / 2), 0]
        if kind == "lean":
            spec.body("table_edge").pos[0] = wall_distance
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("base").id
        self.wheel_qpos = [self.model.joint(n).qposadr[0] for n in WHEELS]
        self.wheel_dof = [self.model.joint(n).dofadr[0] for n in WHEELS]
        self.wheel_act = [self.model.actuator(n).id for n in WHEELS]
        self.arm_qpos = (
            [self.model.joint(n).qposadr[0] for n in ARM_JOINTS] if kind == "arms" else []
        )
        self.arm_dof = [self.model.joint(n).dofadr[0] for n in ARM_JOINTS] if kind == "arms" else []
        self.arm_act = [self.model.actuator(n).id for n in ARM_JOINTS] if kind == "arms" else []
        self.actor = Actor(POLICIES[kind])
        self.balance = self.actor.name == "balance"
        self.motors = [WheelMotor(self.model, s) for s in WHEELS]
        self.command = np.array([velocity, yaw], dtype=float)
        self.lean_degrees = lean_degrees
        self.arm_motion = arm_motion
        self.ideal_wheels = ideal_wheels
        self.start_pitch = np.deg2rad(start_pitch)
        self.reset()

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = [np.cos(self.start_pitch / 2), 0, np.sin(self.start_pitch / 2), 0]
        mujoco.mj_forward(m, d)
        bottoms = []
        for side in WHEELS:
            i = m.geom(side + "_wheel_col").id
            axis = d.geom_xmat[i].reshape(3, 3)[:, 2]
            extent = m.geom_size[i, 0] * np.sqrt(max(0, 1 - axis[2] ** 2)) + m.geom_size[
                i, 1
            ] * abs(axis[2])
            bottoms.append(d.geom_xpos[i, 2] - extent)
        d.qpos[2] = 0.004 - min(bottoms)
        mujoco.mj_forward(m, d)
        if self.balance:
            self.observations = BalanceObservations(m, arms=self.kind == "arms")
            self.turn_filter = TurnFilter()
        else:
            self.observations = Observations()
        self.pending_torques = None
        for motor in self.motors:
            motor.reset()
        self.last_action = np.zeros(2)
        self.last_obs = np.zeros(self.actor.size)
        self.failed = None
        self.stats = dict(
            steps=0,
            max_abs_pitch_deg=0.0,
            max_abs_roll_deg=0.0,
            max_abs_action=0.0,
            max_wall_force_n=0.0,
        )

    def state(self):
        d = self.data
        rotation = d.xmat[self.base].reshape(3, 3)
        pitch = np.arctan2(-rotation[2, 0], rotation[2, 2])
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        # Free-joint angular velocities are local; translation velocities are world.
        gyro = d.qvel[3:6].copy()
        velocity = float((rotation.T @ d.qvel[:3])[0])
        return pitch, roll, gyro, velocity

    def arm_target(self):
        q = np.zeros(14)
        if self.arm_motion:
            t = self.data.time
            ramp = min(t / 2.0, 1.0)
            q[[0, 7]] = -0.18 * ramp * (1 - np.cos(0.6 * t))
            q[[1, 8]] = 0.25 * ramp * np.sin(0.6 * t) * np.array([1, -1])
            q[[2, 9]] = 0.20 * ramp * np.sin(0.45 * t) * np.array([1, -1])
            q[[3, 10]] = 0.30 * ramp * np.sin(0.55 * t)
        return q

    def step(self):
        if self.failed:
            return
        m, d = self.model, self.data
        # Refresh transforms after the previous integration before observing.
        mujoco.mj_forward(m, d)
        pitch, roll, gyro, velocity = self.state()
        arm_pos = d.qpos[self.arm_qpos] if self.arm_qpos else np.zeros(14)
        arm_vel = d.qvel[self.arm_dof] if self.arm_dof else np.zeros(14)
        if self.balance:
            self.last_obs = self.observations.build(
                d.xmat[self.base].reshape(3, 3),
                d.qpos[3:7],
                d.qvel[:3],
                gyro,
                d.qpos[self.wheel_qpos],
                d.qvel[self.wheel_dof],
                self.command,
                arm_pos,
                arm_vel,
                ideal_wheels=self.ideal_wheels,
            )
            self.last_action = self.turn_filter(self.actor(self.last_obs))
            # The wheel command reaches the motor one tick late.
            command = self.last_action * np.array([-TORQUE_NM, TORQUE_NM])
            torques = command if self.pending_torques is None else self.pending_torques
            self.pending_torques = command
        else:
            self.last_obs = self.observations.build(
                d.qpos[self.wheel_qpos],
                d.qvel[self.wheel_dof],
                pitch,
                gyro,
                velocity,
                self.command,
                lean_degrees=self.lean_degrees,
                ideal_wheels=self.ideal_wheels,
            )
            self.last_action = self.actor(self.last_obs)
            torques = self.last_action * np.array([-6.5, 6.5])
        for i, motor in enumerate(self.motors):
            d.ctrl[self.wheel_act[i]] = motor(torques[i], d.qvel[self.wheel_dof[i]])
        if self.arm_act:
            d.ctrl[self.arm_act] = self.arm_target()
        self.observations.remember_action(self.last_action)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        pitch, roll, _, _ = self.state()
        self.stats["steps"] += 1
        self.stats["max_abs_pitch_deg"] = max(
            self.stats["max_abs_pitch_deg"], float(abs(np.rad2deg(pitch)))
        )
        self.stats["max_abs_roll_deg"] = max(
            self.stats["max_abs_roll_deg"], float(abs(np.rad2deg(roll)))
        )
        self.stats["max_abs_action"] = max(
            self.stats["max_abs_action"], float(np.max(np.abs(self.last_action)))
        )
        wall_force = 0.0
        if self.kind == "lean":
            wall_id = m.geom("table_edge_geom").id
            for i in range(d.ncon):
                if wall_id in d.contact[i].geom:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, force)
                    wall_force += np.linalg.norm(force[:3])
        self.stats["max_wall_force_n"] = max(self.stats["max_wall_force_n"], float(wall_force))
        floor_id = m.geom("floor").id
        floor_normal = d.geom_xmat[floor_id].reshape(3, 3)[:, 2]
        ground_clearance = np.dot(d.qpos[:3] - d.geom_xpos[floor_id], floor_normal)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self.failed = "nonfinite state"
        elif any(w.number for w in d.warning):
            self.failed = "MuJoCo numerical warning"
        elif abs(pitch) > np.deg2rad(65) or abs(roll) > np.deg2rad(45) or ground_clearance < 0.025:
            self.failed = "robot fell"

    def summary(self):
        pitch, roll, _, velocity = self.state()
        return dict(
            policy=self.kind,
            seconds=float(self.data.time),
            failure=self.failed,
            position_m=self.data.qpos[:3].tolist(),
            pitch_deg=float(np.rad2deg(pitch)),
            roll_deg=float(np.rad2deg(roll)),
            velocity_mps=velocity,
            command=self.command.tolist(),
            **self.stats,
        )
