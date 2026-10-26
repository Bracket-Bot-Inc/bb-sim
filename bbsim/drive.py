"""Press-and-hold driving shared by keyboard and visible control buttons."""

import mujoco
import numpy as np

from .config import DIRECTION_KEYS


class DriveControls:
    def __init__(self, sim):
        self.sim = sim
        self.speed = min(0.3, abs(sim.command[0])) or 0.15
        self.turn_speed = min(1.0, abs(sim.command[1])) or 0.6
        self.held = {"keyboard": set(), "mouse": set()}
        self.focused = True
        self.paused = False
        self.lean_contact = False
        self.lean_angle = sim.lean_degrees
        self.table_position = sim.data.mocap_pos.copy()
        self.stop()
        self._apply_mode()

    @property
    def keys(self):
        return self.held["keyboard"] | self.held["mouse"]

    def press(self, key, source="keyboard"):
        if key in DIRECTION_KEYS and self.focused and not self.paused and not self.sim.failed:
            self.held[source].add(key)
        self.update()

    def release(self, key, source="keyboard"):
        self.held[source].discard(key)
        self.update()

    def release_mouse(self):
        self.held["mouse"].clear()
        self.update()

    def stop(self):
        for keys in self.held.values():
            keys.clear()
        self.sim.command[:] = 0

    def focus(self, focused):
        self.focused = bool(focused)
        if not self.focused:
            self.stop()

    def update(self):
        if self.paused or not self.focused or self.sim.failed:
            self.sim.command[:] = 0
            return
        keys = self.keys
        self.sim.command[:] = [
            self.speed * (("W" in keys) - ("S" in keys)),
            self.turn_speed * (("A" in keys) - ("D" in keys)),
        ]

    def _apply_mode(self):
        if self.sim.kind == "lean":
            self.sim.lean_request = self.lean_contact
            self.sim.lean_degrees = self.lean_angle
            self.sim.data.mocap_pos[:] = self.table_position
            if not self.lean_contact:
                self.sim.data.mocap_pos[:, 2] = -5.0
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def reset(self):
        self.stop()
        self.sim.reset()
        self.paused = False
        self._apply_mode()

    def activate(self, action):
        if action == "stop":
            self.stop()
        elif action == "pause":
            self.stop()
            self.paused = not self.paused
        elif action == "reset":
            self.reset()
        elif action == "arms" and self.sim.kind == "arms":
            self.sim.arm_motion = not self.sim.arm_motion
        elif action == "lean" and self.sim.kind == "lean":
            self.lean_contact = not self.lean_contact
            self.reset()
        elif action in ("speed-", "speed+"):
            self.speed = float(
                np.clip(round(self.speed + (0.05 if action.endswith("+") else -0.05), 2), 0.05, 0.3)
            )
        elif action in ("turn-", "turn+"):
            self.turn_speed = float(
                np.clip(
                    round(self.turn_speed + (0.1 if action.endswith("+") else -0.1), 2), 0.1, 1.0
                )
            )
        elif action in ("angle-", "angle+"):
            self.lean_angle = float(
                np.clip(self.lean_angle + (1 if action.endswith("+") else -1), 1, 15)
            )
            self._apply_mode()
        self.update()
