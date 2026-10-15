"""Shared paths and policy constants."""

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ARM_JOINTS = [f"{side}j{i}" for side in "lr" for i in range(7)]
WHEELS = ["left", "right"]
POLICY_INPUTS = {"balance": 77, "lean": 19}
POLICIES = {"arms": "balance", "terrain": "balance", "lean": "lean"}
DIRECTION_KEYS = ("W", "A", "S", "D")
DT = 0.005
