"""Roadmap 1.3a: the arm geometry measured from the model matches the numbers the IK is built on."""

import math
from pathlib import Path

import pytest

from sim.arm_geometry import measure_arm_geometry
from sim.scene_state import load_scene

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")

# Measured from MuJoCo Menagerie's SO-101 on 2026-10-24. If these tests fail after
# updating Menagerie, the model changed: re-check the IK before trusting it.
EXPECTED_LENGTHS_M = {
    "pan_x": 0.0388,
    "shoulder_r": 0.0304,
    "shoulder_z": 0.1166,
    "L1": 0.1160,
    "L2": 0.1350,
    "L3": 0.1597,
}
EXPECTED_ANGLES_DEG = {"a1": 76.0, "a2": 2.2, "a3": 4.3}


@pytest.fixture(scope="module")
def geometry():
    model, _ = load_scene()
    return measure_arm_geometry(model)


@pytest.mark.parametrize("name, expected", EXPECTED_LENGTHS_M.items())
def test_lengths_match_the_model(geometry, name, expected):
    assert geometry[name] == pytest.approx(expected, abs=0.0005)  # within 0.5 mm


@pytest.mark.parametrize("name, expected_deg", EXPECTED_ANGLES_DEG.items())
def test_zero_pose_angles_match_the_model(geometry, name, expected_deg):
    assert math.degrees(geometry[name]) == pytest.approx(expected_deg, abs=0.5)  # within 0.5 degree


def test_joint_axes_point_the_way_the_ik_assumes():
    _, data = load_scene()
    assert data.joint("shoulder_pan").xaxis == pytest.approx([0, 0, -1], abs=1e-3)
    for joint in ["shoulder_lift", "elbow_flex", "wrist_flex"]:
        assert data.joint(joint).xaxis == pytest.approx([0, 1, 0], abs=1e-3)