import time

import glfw
import mujoco
import numpy as np

from .drive_ui import Button, DriveWindow

JOG_KEYS = {
    glfw.KEY_UP: (0, 1),
    glfw.KEY_DOWN: (0, -1),
    glfw.KEY_LEFT: (1, 1),
    glfw.KEY_RIGHT: (1, -1),
    glfw.KEY_PAGE_UP: (2, 1),
    glfw.KEY_PAGE_DOWN: (2, -1),
}


class ManipulationControls:
    def __init__(self, sim):
        self.sim = sim
        self.arm = 0
        self.paused = False
        self.focused = True
        self.keys = set()
        self.updated = time.monotonic()

    def stop(self):
        self.keys.clear()

    def focus(self, focused):
        self.focused = focused
        self.stop()

    def update(self):
        now = time.monotonic()
        dt = min(now - self.updated, 0.05)
        self.updated = now
        if self.paused or not self.focused:
            return
        for key in self.keys:
            axis, sign = JOG_KEYS[key]
            self.sim.set_target(self.arm, axis, self.sim.targets[self.arm, axis] + sign * 0.08 * dt)


class ManipulationWindow(DriveWindow):
    PANEL = 360
    INSTRUCTIONS = "Edit XYZ / wrist angles. Arrows: XY; Page Up/Down: Z; Tab: arm; Space: gripper."

    def __init__(self, sim):
        self.editing = None
        self.edit_text = ""
        self.message = ""
        self.hand_view = False
        super().__init__(sim, ManipulationControls(sim), "BracketBot | Fixed-base manipulation")
        glfw.set_window_size_limits(self.window, 1100, 740, glfw.DONT_CARE, glfw.DONT_CARE)
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.1, 145, -18
        glfw.set_char_callback(self.window, self._char)

    def _lookat(self):
        if self.hand_view:
            return self.sim.data.site_xpos[self.sim.sites[self.controls.arm]] + [0, 0, 0.02]
        return [0.1, 0, 0.82]

    def _commit(self):
        if self.editing is None:
            return True
        try:
            accepted = self.sim.set_target(self.controls.arm, self.editing, float(self.edit_text))
        except ValueError:
            accepted = False
        if not accepted:
            self.message = "Invalid number or outside workspace."
            return False
        self.editing = None
        self.message = ""
        return True

    def _char(self, window, codepoint):
        char = chr(codepoint)
        if self.editing is not None and char in "0123456789.-+eE" and len(self.edit_text) < 16:
            self.edit_text += char

    def _key(self, window, key, scancode, action, mods):
        if key in JOG_KEYS and action == glfw.RELEASE:
            self.controls.keys.discard(key)
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if self.editing is not None:
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self._commit()
            elif key == glfw.KEY_ESCAPE:
                self.editing = None
                self.message = ""
            elif key == glfw.KEY_BACKSPACE:
                self.edit_text = self.edit_text[:-1]
            return
        if mods & (glfw.MOD_SUPER | glfw.MOD_CONTROL | glfw.MOD_ALT):
            return
        if key in JOG_KEYS:
            self.controls.keys.add(key)
        elif action == glfw.PRESS:
            if key == glfw.KEY_TAB:
                self.controls.stop()
                self.controls.arm = 1 - self.controls.arm
            elif key == glfw.KEY_SPACE:
                arm = self.controls.arm
                self.sim.opening[arm] = float(self.sim.opening[arm] < 0.5)
            elif key == glfw.KEY_P:
                self.controls.paused = not self.controls.paused
            elif key == glfw.KEY_R:
                self._activate("reset")
            elif key == glfw.KEY_F:
                self.hand_view = not self.hand_view
                self.camera.distance, self.camera.azimuth, self.camera.elevation = (
                    (0.55, 0, -8) if self.hand_view else (2.1, 145, -18)
                )
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def _activate(self, action):
        if not self._commit():
            return
        c, sim = self.controls, self.sim
        if action.startswith("arm:"):
            c.stop()
            c.arm = int(action[-1])
        elif action.startswith("edit:"):
            c.stop()
            self.editing = int(action[-1])
            self.edit_text = ""
        elif action.startswith("jog:"):
            _, axis, sign = action.split(":")
            axis = int(axis)
            sim.set_target(
                c.arm,
                axis,
                sim.targets[c.arm, axis] + float(sign) * (0.01 if axis < 3 else 5),
            )
        elif action == "open":
            sim.opening[c.arm] = 1
        elif action == "close":
            sim.opening[c.arm] = 0
        elif action == "pause":
            c.paused = not c.paused
        elif action == "reset":
            c.stop()
            sim.reset()
            c.paused = False

    def _mouse_button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        if action == glfw.RELEASE:
            self.drag = None
            return
        if x >= self.width - self.PANEL:
            if button == glfw.MOUSE_BUTTON_LEFT:
                for item in self.buttons:
                    if item.contains(x, y):
                        self._activate(item.action)
                        break
        elif self._commit():
            self.drag = button
        self.mouse = (x, y)

    def _decorate_scene(self):
        origin = self.sim.data.body("base").xpos
        for arm, color in enumerate(([0.15, 0.65, 0.9, 0.65], [0.95, 0.48, 0.18, 0.65])):
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.009] * 3,
                origin + self.sim.targets[arm, :3],
                np.eye(3).ravel(),
                color,
            )
            self.scene.ngeom += 1

    def _panel(self):
        c, sim = self.controls, self.sim
        left, x = self.width - self.PANEL, self.width - self.PANEL + 20
        self._fill(left, 0, self.PANEL, self.height, (0.045, 0.065, 0.09, 1))
        self._text(x, 18, "MANIPULATION", big=True)
        self._text(x, 54, "Fixed base  /  arm IK")
        self.buttons = [
            Button("arm:0", "Left arm", x, 87, 154, active=c.arm == 0),
            Button("arm:1", "Right arm", x + 164, 87, 154, active=c.arm == 1),
        ]
        self._text(x, 143, "Hand target (m)  /  click value to type")
        for axis, label in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            y = 176 + axis * 44 + (28 if axis >= 3 else 0)
            if axis == 3:
                self._text(x, y - 26, "Wrist orientation (degrees)")
            self._text(x, y + 8, label)
            text = (
                f"{sim.targets[c.arm, axis]:+.3f}"
                if axis < 3
                else f"{sim.targets[c.arm, axis]:+.1f}"
            )
            if self.editing == axis:
                text = self.edit_text + "|"
            self.buttons.extend(
                [
                    Button(f"edit:{axis}", text, x + 56, y, 140, 36, self.editing == axis),
                    Button(f"jog:{axis}:-1", "-", x + 210, y, 48, 36),
                    Button(f"jog:{axis}:1", "+", x + 268, y, 48, 36),
                ]
            )
        self.buttons.extend(
            [
                Button("open", "Open gripper", x, 480, 154, active=sim.opening[c.arm] > 0.5),
                Button(
                    "close",
                    "Close gripper",
                    x + 164,
                    480,
                    154,
                    active=sim.opening[c.arm] < 0.5,
                ),
                Button("pause", "P  Resume" if c.paused else "P  Pause", x, 625, 154),
                Button("reset", "R  Reset scene", x + 164, 625, 154),
            ]
        )
        error = sim.errors()[c.arm] * 1000
        angle = sim.rotation_errors()[c.arm]
        status = (
            "PHYSICS ERROR - reset"
            if sim.failed
            else (
                "PAUSED"
                if c.paused
                else "TRACKING"
                if error < 10 and angle < 3
                else "MOVING / TARGET NOT REACHED"
            )
        )
        pads = (
            "Rigid inserts"
            if sim.rigid_pads
            else f"Insert flex: {sim.pad_deflection(c.arm) * 1000:.1f} mm"
        )
        self._text(x, 530, f"{status}\nError: {error:.1f} mm / {angle:.1f} deg\n{pads}")
        self._text(x, 592, self.message or "Step: 1 cm / 5 deg. Enter applies input.")
        self._text(x, 682, "X forward / Y left / Z above wheel axle")
        for button in self.buttons:
            self._draw_button(button)
        self._text(18, 16, f"{sim.data.time:.1f} s  |  Fixed base  |  Grab either object")
        self._text(18, self.height - 59, "Arrows: XY   PgUp/PgDn: Z   Tab: arm   Space: grip")
        self._text(
            18,
            self.height - 35,
            "F: hand view   Drag: orbit   Scroll: zoom   P: pause   R: reset   Esc: close",
        )
