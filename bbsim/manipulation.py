"""Fixed-base arm manipulation with the arm daemon's IK and physical gripper contacts."""

import argparse
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import ASSETS
from .ik import DT, ArmIK
from .policy import onnx_session


class Manipulation:
    kind = "manipulation"

    def __init__(self, rigid_pads=False):
        self.rigid_pads = rigid_pads
        path = ASSETS / "manipulation.xml"
        if rigid_pads:
            xml = ET.parse(path).getroot()
            xml.find("compiler").set("meshdir", str(ASSETS / "meshes"))
            for body in xml.iter("body"):
                for joint in list(body.findall("joint")):
                    if "_pad" in joint.get("name", ""):
                        body.remove(joint)
            equality = xml.find("equality")
            for constraint in list(equality):
                if "_pad" in constraint.get("name", ""):
                    equality.remove(constraint)
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.pad_reference = mujoco.MjData(self.model)
        self.pad_qpos = np.array(
            [
                self.model.jnt_qposadr[i]
                for i in range(self.model.njnt)
                if "_pad" in self.model.joint(i).name
            ],
            dtype=int,
        )
        self.pad_geoms = np.array(
            [
                [
                    i
                    for i in range(self.model.ngeom)
                    if self.model.geom(i).name.startswith(f"{side}_pad")
                ]
                for side in "lr"
            ],
            dtype=int,
        )
        self.joints = np.array(
            [[self.model.joint(f"{side}j{i}").id for i in range(7)] for side in "lr"]
        )
        self.qpos = self.model.jnt_qposadr[self.joints]
        self.actuators = np.array(
            [[self.model.actuator(f"{side}j{i}").id for i in range(7)] for side in "lr"]
        )
        self.grippers = np.array([self.model.actuator(f"{side}_grip").id for side in "lr"])
        self.grip_range = self.model.actuator_ctrlrange[self.grippers]
        self.sites = [self.model.site(f"{side}_eef").id for side in "lr"]
        session = onnx_session("ik")
        self.solvers = [ArmIK(self.model, side, session) for side in "lr"]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.targets = np.array([[0.25, 0.20, 0.60, 180, 0, 0], [0.25, -0.20, 0.60, 180, 0, 0]])
        self.opening = np.ones(2)
        self.command = np.zeros((2, 7))
        self.failed = False
        self.ticks = 0
        for arm, solver in enumerate(self.solvers):
            solver.reset(np.zeros(7))
            quaternion = Rotation.from_euler("xyz", self.targets[arm, 3:], degrees=True).as_quat()
            # Half a second of a held target, ending at t = 0.
            for k in range(30, -1, -1):
                self.command[arm] = solver.solve(self.targets[arm, :3], quaternion, -k * DT)
            self.data.qpos[self.qpos[arm]] = self.command[arm]
            for i in range(2):
                joint = self.model.joint(f"{'lr'[arm]}_grip{i}")
                self.data.qpos[joint.qposadr] = self.grip_range[arm, 1]
        self.solution = self.command.copy()
        self.data.ctrl[self.actuators] = self.command
        self.data.ctrl[self.grippers] = self.grip_range[:, 1]
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, arm, axis, value):
        if not np.isfinite(value):
            return False
        low, high = ((-0.5, 0.65), (-0.65, 0.65), (0.15, 1.3), *[(-180, 180)] * 3)[axis]
        if not low <= value <= high:
            return False
        self.targets[arm, axis] = value
        return True

    def positions(self):
        return self.data.site_xpos[self.sites] - self.data.body("base").xpos

    def errors(self):
        return np.linalg.norm(self.positions() - self.targets[:, :3], axis=1)

    def rotation_errors(self):
        actual = Rotation.from_matrix(self.data.site_xmat[self.sites].reshape(2, 3, 3))
        target = Rotation.from_euler("xyz", self.targets[:, 3:], degrees=True)
        return np.rad2deg((target * actual.inv()).magnitude())

    def pad_deflection(self, arm=None):
        if self.rigid_pads:
            return 0.0
        self.pad_reference.qpos[:] = self.data.qpos
        self.pad_reference.qpos[self.pad_qpos] = 0
        mujoco.mj_kinematics(self.model, self.pad_reference)
        displacement = (
            self.data.geom_xpos[self.pad_geoms] - self.pad_reference.geom_xpos[self.pad_geoms]
        )
        distances = np.linalg.norm(displacement, axis=-1)
        return float((distances if arm is None else distances[arm]).max())

    def step(self):
        if self.failed:
            return
        if self.ticks % 10 == 0:
            for arm, solver in enumerate(self.solvers):
                quaternion = Rotation.from_euler(
                    "xyz", self.targets[arm, 3:], degrees=True
                ).as_quat()
                self.solution[arm] = solver.solve(self.targets[arm, :3], quaternion, self.data.time)
        dt = self.model.opt.timestep
        speed = np.array([0.18, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]) * dt
        self.command += np.clip(self.solution - self.command, -speed, speed)
        self.data.ctrl[self.actuators] = self.command
        grip_target = self.grip_range[:, 0] + self.opening * np.diff(self.grip_range, axis=1)[:, 0]
        self.data.ctrl[self.grippers] += np.clip(
            grip_target - self.data.ctrl[self.grippers], -2 * dt, 2 * dt
        )
        mujoco.mj_step(self.model, self.data)
        self.ticks += 1
        self.failed = not np.isfinite(self.data.qpos).all() or any(
            warning.number for warning in self.data.warning
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--rigid-pads", action="store_true", help="Hold the inserts rigid for comparison"
    )
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    duration = (
        args.duration if args.duration is not None else (5 if args.headless else float("inf"))
    )
    sim = Manipulation(args.rigid_pads)
    if args.headless:
        while sim.data.time < duration and not sim.failed:
            sim.step()
        print(f"{sim.data.time:.1f}s | hand errors (mm): {sim.errors() * 1000}")
    else:
        from .manipulation_ui import ManipulationWindow

        window = ManipulationWindow(sim)
        try:
            window.run(duration, sim.step)
        finally:
            window.close()
    if sim.failed:
        parser.exit(2, "Physics became unstable; reset the scene.\n")


if __name__ == "__main__":
    main()
