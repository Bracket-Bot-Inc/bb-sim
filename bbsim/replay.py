"""Play back recorded episodes from ./datasets on the MuJoCo model."""

import argparse
import io
import json
import re
import textwrap
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import av
import glfw
import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .config import ASSETS
from .drive_ui import Button, DriveWindow

LIFT_RADIUS = 0.0465
IK_SIGN = {"l": np.array([-1, 1, -1, 1, 1, -1, -1]), "r": np.array([-1, -1, -1, -1, 1, -1, 1])}
LIFT_SIGN = {"l": 1, "r": -1}
GRIP_SIGN = {"l": -1, "r": 1}
SIDES = {"l": "left", "r": "right"}
CAMERAS = ("head", "arm_left", "arm_right")
STEREO = {"head"}
GRIPPER = 7
IMAGE_PREFIX = "observation.images."
CHUNK_RE = re.compile(r"ep(\d+)(?:_([0-9a-f]{8}))?(?:_(\d+))?\.npz")


def zoh(times, t):
    return max(int(np.searchsorted(times, t, side="right")) - 1, 0)


class Video:

    def __init__(self):
        self.frame = -1
        self.image = None

    def read(self, frame, size):
        frame = int(np.clip(frame, 0, self.frames - 1))
        if frame == self.frame and self.image is not None and self.image.shape[1::-1] == size:
            return self.image
        image = self._decode(frame, size)
        if image is None:
            return self.image
        self.image = np.asarray(image.convert("RGB").resize(size, Image.BILINEAR))
        self.frame = frame
        return self.image


class MjpegVideo(Video):

    def __init__(self, paths, stereo):
        super().__init__()
        self.paths, self.stereo = paths, stereo
        self.pts = []
        for path in paths:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                self.pts.append([p.pts for p in container.demux(stream) if p.size])
        self.offsets = np.cumsum([0] + [len(p) for p in self.pts])
        self.frames = self.offsets[-1]
        self.container = None
        self.chunk = -1
        self.next = None

    def close(self):
        if self.container:
            self.container.close()
            self.container = None

    def _packet(self, chunk, k):
        if chunk != self.chunk:
            self.close()
            self.container = av.open(str(self.paths[chunk]))
            self.chunk, self.next = chunk, 0
        stream = self.container.streams.video[0]
        if self.next is None or not 0 <= k - self.next < 15:
            self.container.seek(self.pts[chunk][k], stream=stream)
            self.next = None
        target = self.pts[chunk][k]
        for packet in self.container.demux(stream):
            if packet.size and packet.pts >= target:
                self.next = k + 1
                return bytes(packet)
        self.next = None
        return None

    def _decode(self, frame, size):
        chunk = int(np.searchsorted(self.offsets, frame, side="right")) - 1
        data = self._packet(chunk, frame - self.offsets[chunk])
        if data is None:
            return None
        image = Image.open(io.BytesIO(data))
        image.draft("RGB", (size[0] * (2 if self.stereo else 1), size[1]))
        if self.stereo:
            image = image.convert("RGB")
            image = image.crop((0, 0, image.width // 2, image.height))
        return image


class Mp4Video(Video):

    def __init__(self, path, fps):
        super().__init__()
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.fps = fps
        self.frames = self.stream.frames or int(self.stream.duration * self.stream.time_base * fps)
        self.decoded = None
        self.position = None

    def close(self):
        self.container.close()

    def _decode(self, frame, size):
        if self.position is None or not 0 < frame - self.position < 15:
            self.container.seek(int(frame / self.fps / self.stream.time_base), stream=self.stream)
            self.decoded = self.container.decode(self.stream)
        for image in self.decoded:
            self.position = round(float(image.pts * self.stream.time_base) * self.fps)
            if self.position >= frame:
                return image.to_image()
        self.position = None
        return None


class NpzEpisode:
    def __init__(self, run, idx, uid, chunks):
        self.run, self.idx, self.uid, self.chunks = run, idx, uid, chunks
        self.name = f"{run.name} / ep{idx:03d}" + (f"_{uid}" if uid else "")

    def load(self):
        chunks = [np.load(path) for path in self.chunks]
        first = chunks[0]
        self.task = str(first["meta_task"]) if "meta_task" in first else ""

        def channel(name):
            if name not in first:
                return None
            return np.concatenate([chunk[name] for chunk in chunks])

        if "video_head_timestamp_ns" in first:
            self.t0 = int(first["video_head_timestamp_ns"][0])
        else:
            self.t0 = int(min(channel(f"arm_{s}_state")["timestamp"][0] for s in SIDES.values()))

        def seconds(stamps):
            return (stamps.astype("datetime64[ns]").astype(np.int64) - self.t0) / 1e9

        self.arms, end = {}, 0.0
        for source in ("state", "ctrl"):
            for side, name in SIDES.items():
                data = channel(f"arm_{name}_{source}")
                times = seconds(data["timestamp"])
                self.arms[source, side] = (times, data["pos"].astype(float))
                end = max(end, times[-1])
        self.duration = end

        self.cameras, self.videos = {}, {}
        for camera in CAMERAS:
            key = f"video_{camera}_timestamp_ns"
            files = [
                self.run / "video" / f"{camera}_{path.stem}.mkv" for path in self.chunks
            ]
            if key not in first or not all(f.exists() for f in files):
                continue
            self.cameras[camera] = np.concatenate([(c[key] - self.t0) / 1e9 for c in chunks])
            self.videos[camera] = MjpegVideo(files, camera in STEREO)
        return self

    def close(self):
        for video in getattr(self, "videos", {}).values():
            video.close()


def unnormalize(values, calibration):
    low, high = np.asarray(calibration["min"]), np.asarray(calibration["max"])
    frac = (values + 100) / 200
    frac[:, GRIPPER] = values[:, GRIPPER] / 100
    return low + frac * (high - low)


class LeRobotEpisode:

    def __init__(self, root, info, meta):
        self.root, self.info, self.meta = root, info, meta
        self.idx = meta["episode_index"]
        self.name = f"{root.name} / episode {self.idx}"

    def _path(self, template, **keys):
        chunk = self.idx // self.info["chunks_size"]
        return self.root / template.format(episode_chunk=chunk, episode_index=self.idx, **keys)

    def _calibration(self):
        path = self.root / "meta" / "calibration.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["episode_index"] == self.idx:
                    return row
        return None

    def load(self):
        table = pq.read_table(self._path(self.info["data_path"]))
        times = table["timestamp"].to_numpy().astype(float)
        self.task = ", ".join(self.meta.get("tasks", []))
        self.duration = float(times[-1])

        def column(name):
            values = table[name].combine_chunks().flatten().to_numpy()
            return values.reshape(len(table), -1).astype(float)

        self.arms = {}
        calibration = None
        for source, key in (("state", "observation.state"), ("ctrl", "action")):
            raw = f"raw.{key}" in table.column_names
            if not raw and calibration is None:
                calibration = self._calibration()
                if calibration is None:
                    raise ValueError(
                        f"{self.name}: no raw.{key} column and no meta/calibration.jsonl entry"
                    )
            values = column(f"raw.{key}" if raw else key)
            for i, (side, name) in enumerate(SIDES.items()):
                arm = values[:, 8 * i : 8 * i + 8]
                if not raw:
                    arm = unnormalize(arm, calibration[f"arm_{name}"])
                self.arms[source, side] = (times, arm)

        self.cameras, self.videos = {}, {}
        for key, feature in self.info["features"].items():
            path = self._path(self.info.get("video_path") or "", video_key=key)
            if feature["dtype"] != "video" or not key.startswith(IMAGE_PREFIX) or not path.exists():
                continue
            camera = key.removeprefix(IMAGE_PREFIX)
            self.cameras[camera] = times
            self.videos[camera] = Mp4Video(path, self.info["fps"])
        return self

    def close(self):
        for video in getattr(self, "videos", {}).values():
            video.close()


def find_lerobot(root):
    episodes = []
    for info_path in sorted(root.rglob("meta/info.json")):
        info = json.loads(info_path.read_text())
        version = info.get("codebase_version", "")
        if not version:
            continue
        dataset = info_path.parent.parent
        if not version.startswith("v2."):
            print(f"Skipping {dataset}: LeRobot {version} is not supported (need v2.1)")
            continue
        lines = (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
        episodes += [LeRobotEpisode(dataset, info, json.loads(line)) for line in lines if line]
    return episodes


def find_episodes(root):
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("episodes/ep*.npz"))
    groups = {}
    for path in paths:
        match = CHUNK_RE.fullmatch(path.name)
        if not match:
            continue
        idx, uid, chunk = match.groups()
        key = (path.parent.parent, int(idx), uid or "")
        groups.setdefault(key, []).append((int(chunk or 0), path))
    if root.is_file():
        (run, idx, uid), _ = next(iter(groups.items()))
        return [e for e in find_episodes(run) if (e.idx, e.uid) == (idx, uid)]
    raw = [
        NpzEpisode(run, idx, uid, [path for _, path in sorted(chunks)])
        for (run, idx, uid), chunks in sorted(groups.items(), key=lambda g: (str(g[0][0]), g[0][1:]))
    ]
    return find_lerobot(root) + raw


class Replay:

    kind = "replay"

    def __init__(self, episodes, index=0):
        xml = ET.parse(ASSETS / "manipulation.xml").getroot()
        xml.find("compiler").set("meshdir", str(ASSETS / "meshes"))
        world = xml.find("worldbody")
        for child in list(world):
            props = child.find("freejoint") is not None or child.get("name", "").startswith("table")
            if props:
                world.remove(child)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.failed = False
        self.joints = {
            side: self.model.jnt_qposadr[[self.model.joint(f"{side}j{i}").id for i in range(7)]]
            for side in SIDES
        }
        self.limits = {
            side: self.model.jnt_range[[self.model.joint(f"{side}j{i}").id for i in range(7)]]
            for side in SIDES
        }
        self.grips = {
            side: [self.model.joint(f"{side}_grip{i}") for i in range(2)] for side in SIDES
        }
        self.episodes = episodes
        self.source = "state"
        self.episode = None
        self.playing = True
        self.open(index)

    def open(self, index):
        if self.episode:
            self.episode.close()
        self.index = index % len(self.episodes)
        start = time.monotonic()
        self.episode = self.episodes[self.index].load()
        print(
            f"[{self.index + 1}/{len(self.episodes)}] {self.episode.name}  "
            f"{self.episode.duration:.1f}s  {self.episode.task!r}  "
            f"(loaded in {time.monotonic() - start:.1f}s)",
            flush=True,
        )
        self.t = 0.0
        self.seek(0.0)

    def seek(self, t):
        self.t = float(np.clip(t, 0, self.episode.duration))
        for side in SIDES:
            times, pos = self.episode.arms[self.source, side]
            turns = pos[zoh(times, self.t)]
            q = turns[:7] * 2 * np.pi
            q[0] *= LIFT_SIGN[side] * LIFT_RADIUS
            q *= IK_SIGN[side]
            limits = self.limits[side]
            self.data.qpos[self.joints[side]] = np.clip(q, limits[:, 0], limits[:, 1])
            grip = turns[7] * 2 * np.pi * GRIP_SIGN[side]
            for joint in self.grips[side]:
                self.data.qpos[joint.qposadr] = np.clip(grip, *joint.range)
        self.data.time = self.t
        mujoco.mj_forward(self.model, self.data)

    def stop(self):
        self.playing = False

    def advance(self, dt):
        if self.playing:
            self.seek(self.t + dt)
            if self.t >= self.episode.duration:
                self.playing = False

    def close(self):
        if self.episode:
            self.episode.close()


class ReplayWindow(DriveWindow):
    PANEL = 360
    INSTRUCTIONS = (
        "Space: play/pause   Left/Right: -/+ 1 s (Shift: 0.1 s)   "
        "N/B: next/prev episode   C: state/ctrl   Home: restart"
    )

    def __init__(self, replay):
        self.scrubbing = False
        super().__init__(replay, replay, "BracketBot | Episode replay")
        glfw.set_window_size_limits(self.window, 1100, 760, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.set_window_size(self.window, 1280, 800)
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.6, 150, -15

    def _lookat(self):
        return [0.15, 0, 0.95]

    def _focus(self, window, focused):
        self.drag = None

    def _key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        r = self.sim
        step = 0.1 if mods & glfw.MOD_SHIFT else 1.0
        if key == glfw.KEY_RIGHT:
            r.seek(r.t + step)
        elif key == glfw.KEY_LEFT:
            r.seek(r.t - step)
        elif action != glfw.PRESS:
            return
        elif key == glfw.KEY_SPACE:
            self._activate("play")
        elif key == glfw.KEY_N:
            self._activate("next")
        elif key == glfw.KEY_B:
            self._activate("prev")
        elif key == glfw.KEY_C:
            self._activate("source")
        elif key == glfw.KEY_HOME:
            r.seek(0)
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def _activate(self, action):
        r = self.sim
        if action == "play":
            if r.t >= r.episode.duration:
                r.seek(0)
            r.playing = not r.playing
        elif action in ("next", "prev"):
            r.open(r.index + (1 if action == "next" else -1))
            r.playing = True
        elif action == "source":
            r.source = "ctrl" if r.source == "state" else "state"
            r.seek(r.t)

    def _bar(self):
        left = self.width - self.PANEL + 20
        return left, 520, self.PANEL - 40, 14

    def _scrub(self, x):
        left, _, width, _ = self._bar()
        self.sim.seek((x - left) / width * self.sim.episode.duration)

    def _mouse_button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        if action == glfw.RELEASE:
            self.drag = None
            self.scrubbing = False
            return
        self.mouse = (x, y)
        if x < self.width - self.PANEL:
            self.drag = button
            return
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        bx, by, bw, bh = self._bar()
        if bx - 4 <= x <= bx + bw + 4 and by - 8 <= y <= by + bh + 8:
            self.scrubbing = True
            self._scrub(x)
            return
        for item in self.buttons:
            if item.contains(x, y):
                self._activate(item.action)
                break

    def _cursor(self, window, x, y):
        if self.scrubbing:
            self.mouse = (x, y)
            self._scrub(x)
            return
        super()._cursor(window, x, y)

    def _image(self, camera, x, y, width, height):
        self._fill(x, y, width, height, (0.02, 0.03, 0.04, 1))
        episode = self.sim.episode
        if camera in episode.videos:
            rect = self._rect(x, y, width, height)
            frame = zoh(episode.cameras[camera], self.sim.t)
            pixels = episode.videos[camera].read(frame, (rect.width, rect.height))
            if pixels is not None:
                mujoco.mjr_drawPixels(
                    np.ascontiguousarray(pixels[::-1]).ravel(), None, rect, self.context
                )
        else:
            self._text(x + 8, y + 8, "no video")
        self._text(x + 6, y + height - 24, camera)

    def _panel(self):
        r, episode = self.sim, self.sim.episode
        left, x = self.width - self.PANEL, self.width - self.PANEL + 20
        w = self.PANEL - 40
        self._fill(left, 0, self.PANEL, self.height, (0.045, 0.065, 0.09, 1))
        self._fill(left, 0, 2, self.height, (0.13, 0.20, 0.27, 1))
        self._text(x, 14, "EPISODE REPLAY", big=True)
        self._text(x, 48, f"{r.index + 1}/{len(r.episodes)}  {episode.name}"[:44])
        task = episode.task or "(no task)"
        self._text(x, 70, "\n".join(textwrap.wrap(task, 40, max_lines=2, placeholder="...")))
        self._image("head", x, 116, w, w * 3 // 4)
        half = (w - 8) // 2
        self._image("arm_left", x, 116 + w * 3 // 4 + 8, half, half * 3 // 4)
        self._image("arm_right", x + half + 8, 116 + w * 3 // 4 + 8, half, half * 3 // 4)

        bx, by, bw, bh = self._bar()
        self._fill(bx, by, bw, bh, (0.10, 0.15, 0.21, 1))
        done = r.t / max(episode.duration, 1e-9)
        self._fill(bx, by, bw * done, bh, (0.12, 0.55, 0.47, 1))
        self._fill(bx + bw * done - 2, by - 4, 4, bh + 8, (0.88, 0.93, 0.97, 1))
        self._text(x, by + 22, f"{r.t:.2f} / {episode.duration:.2f} s")

        self.buttons = [
            Button("prev", "B  Prev", x, 576, 98),
            Button("play", "Pause" if r.playing else "Play", x + 106, 576, 108, active=r.playing),
            Button("next", "N  Next", x + 222, 576, 98),
            Button("source", f"C  Joints: {r.source}", x, 624, w, active=r.source == "ctrl"),
        ]
        for button in self.buttons:
            self._draw_button(button)
        self._text(x, 674, "state = measured, ctrl = commanded")
        self._text(18, 16, f"{episode.name}  |  {r.t:.2f} s  |  {r.source}")
        self._text(
            18,
            self.height - 35,
            "Space: play   Left/Right: seek   N/B: episode   Drag: orbit   Esc: close",
        )

    def run(self):
        print(self.INSTRUCTIONS, flush=True)
        last = time.monotonic()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            now = time.monotonic()
            self.sim.advance(min(now - last, 0.1))
            last = now
            self.render()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - now)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("datasets"),
        help="Dataset root, run directory, or one episode .npz (default: ./datasets)",
    )
    parser.add_argument("--episode", type=int, default=1, help="Start at this entry of --list")
    parser.add_argument("--list", action="store_true", help="List episodes and exit")
    args = parser.parse_args()

    if not args.path.exists():
        parser.exit(1, f"{args.path} does not exist\n")
    episodes = find_episodes(args.path)
    if not episodes:
        parser.exit(1, f"No episodes found under {args.path}\n")
    if args.list:
        for i, episode in enumerate(episodes, 1):
            print(f"{i:4}  {episode.name}")
        return
    if not 1 <= args.episode <= len(episodes):
        parser.error(f"--episode must be between 1 and {len(episodes)}")

    replay = Replay(episodes, args.episode - 1)
    try:
        window = ReplayWindow(replay)
        try:
            window.run()
        finally:
            window.close()
    finally:
        replay.close()


if __name__ == "__main__":
    main()
