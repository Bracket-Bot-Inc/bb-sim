import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from .config import DT
from .simulation import Simulation


def camera(sim):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = sim.data.qpos[:3] + [0, 0, 0.64]
    cam.distance = 2.65
    cam.azimuth = 135
    cam.elevation = -12
    return cam


def screenshot(sim, path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    sim.model.vis.global_.offwidth = 1280
    sim.model.vis.global_.offheight = 960
    with mujoco.Renderer(sim.model, height=960, width=1280) as renderer:
        renderer.update_scene(sim.data, camera=camera(sim))
        Image.fromarray(renderer.render()).save(path)


def finite(value):
    v = float(value)
    if not np.isfinite(v):
        raise argparse.ArgumentTypeError("must be finite")
    return v


def main():
    if sys.argv[1:2] == ["manipulation"]:
        from .manipulation import main as manipulation

        sys.argv[:2] = [f"{sys.argv[0]} manipulation"]
        return manipulation()
    if sys.argv[1:2] == ["replay"]:
        from .replay import main as replay

        sys.argv[:2] = [f"{sys.argv[0]} replay"]
        return replay()
    p = argparse.ArgumentParser(
        description="Run a BracketBot policy locally with chopped CAD geometry.",
        epilog="Run `bbsim manipulation --help` for the fixed-base arm scene and "
        "`bbsim replay --help` to play back recorded episodes.",
    )
    p.add_argument("policy", choices=["arms", "terrain", "lean"])
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--duration",
        type=finite,
        help="Simulation seconds; default: unlimited viewer / 20 headless",
    )
    p.add_argument(
        "--velocity",
        type=finite,
        default=0.0,
        help="Forward command when headless; WASD speed in the UI (m/s)",
    )
    p.add_argument(
        "--yaw",
        type=finite,
        default=0.0,
        help="Yaw command when headless; WASD turn speed in the UI (rad/s)",
    )
    p.add_argument("--lean-angle", type=finite, default=3.0, help="LEAN_V3E angle in degrees, 1–15")
    p.add_argument(
        "--wall-distance", type=finite, default=0.32, help="Lean table slab center X, metres"
    )
    p.add_argument(
        "--arm-motion",
        action="store_true",
        help="Gentle arm reach demo; arm policy only (M in viewer)",
    )
    p.add_argument("--terrain", choices=["flat", "bumps", "slope"], default="flat")
    p.add_argument(
        "--ideal-wheels",
        action="store_true",
        help="Use exact angular velocity instead of Hall estimator",
    )
    p.add_argument(
        "--start-pitch", type=finite, default=0.0, help="Initial pitch perturbation in degrees"
    )
    p.add_argument("--report", type=Path, help="Write rollout summary JSON")
    p.add_argument("--screenshot", type=Path, help="Save final frame as PNG")
    p.add_argument(
        "--trace", type=Path, help="Save 200 Hz state, observation and action arrays as NPZ"
    )
    a = p.parse_args()
    if a.duration is not None and a.duration <= 0:
        p.error("--duration must be positive")
    if not 1 <= a.lean_angle <= 15:
        p.error("--lean-angle must be between 1 and 15 degrees")
    if a.arm_motion and a.policy != "arms":
        p.error("--arm-motion requires the arms policy; terrain/lean use welded arms")
    if abs(a.velocity) > 1 or abs(a.yaw) > 2:
        p.error("command limits: |velocity| <= 1 m/s and |yaw| <= 2 rad/s")
    if abs(a.start_pitch) >= 45:
        p.error("--start-pitch must be inside ±45 degrees")
    if a.wall_distance < 0.10:
        p.error("--wall-distance must be at least .10 m")
    sim = Simulation(
        a.policy,
        velocity=a.velocity,
        yaw=a.yaw,
        lean_degrees=a.lean_angle,
        wall_distance=a.wall_distance,
        arm_motion=a.arm_motion,
        terrain=a.terrain,
        ideal_wheels=a.ideal_wheels,
        start_pitch=a.start_pitch,
    )
    duration = a.duration if a.duration is not None else (20 if a.headless else float("inf"))
    trace = dict(time=[], qpos=[], qvel=[], obs=[], action=[])

    def tick():
        sim.step()
        if a.trace:
            for key, value in [
                ("time", sim.data.time),
                ("qpos", sim.data.qpos.copy()),
                ("qvel", sim.data.qvel.copy()),
                ("obs", sim.last_obs.copy()),
                ("action", sim.last_action.copy()),
            ]:
                trace[key].append(value)

    print(
        f"{a.policy}: 200 Hz | {sim.actor.size} observations | 13.605122 kg | local simulation",
        flush=True,
    )
    if a.headless:
        for _ in range(round(duration / DT)):
            tick()
            if sim.failed:
                break
    else:
        from .drive_ui import run_viewer

        run_viewer(sim, duration, tick)
    report = sim.summary()
    report.update(
        terrain=a.terrain,
        ideal_wheels=a.ideal_wheels,
        arm_motion=sim.arm_motion,
        mujoco_version=mujoco.__version__,
        lean_angle_deg=sim.lean_degrees,
    )
    if a.screenshot:
        screenshot(sim, a.screenshot)
    if a.trace:
        a.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.trace, **{k: np.array(v) for k, v in trace.items()})
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    sys.exit(2 if sim.failed else 0)
