"""Roadmap 1.3c: move_to() really moves the gripper there in the physics simulation."""

from pathlib import Path

import numpy as np
import pytest

from sim.ik import ARM_JOINTS
from sim.scene_state import get_scene_state, load_scene
from sim.skills import move_to

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")

ABOVE = {
    "above red cube": (0.25, -0.06, 0.05),
    "above green cube": (0.25, 0.0, 0.05),
    "above blue cube": (0.25, 0.06, 0.05),
    "above left bin": (0.18, 0.18, 0.06),
    "above right bin": (0.18, -0.18, 0.06),
}


def tip_error_mm(data, target):
    return np.linalg.norm(data.site("gripperframe").xpos - np.array(target)) * 1000


def cube_positions(data):
    return {name: np.array(info["position"]) for name, info in get_scene_state(data).items()}


@pytest.mark.parametrize("name", ABOVE)
def test_move_to_arrives_without_touching_the_cubes(name):
    model, data = load_scene()
    before = cube_positions(data)

    result = move_to(model, data, *ABOVE[name])

    assert result is None, result
    assert tip_error_mm(data, ABOVE[name]) < 5.0
    after = cube_positions(data)
    for cube in before:
        assert np.linalg.norm(after[cube] - before[cube]) < 0.001, f"{cube} was bumped"


def test_several_moves_in_a_row():
    model, data = load_scene()
    for name in ["above green cube", "above left bin", "above red cube", "above right bin"]:
        assert move_to(model, data, *ABOVE[name]) is None
        assert tip_error_mm(data, ABOVE[name]) < 5.0


def test_tilts_when_straight_down_is_blocked_by_the_wrist_limit():
    model, data = load_scene()
    target = (0.25, 0.0, 0.095)  # 8 cm above the green cube: straight down passes the wrist limit
    assert move_to(model, data, *target) is None
    assert tip_error_mm(data, target) < 5.0


def test_unreachable_target_returns_a_reason_and_does_not_move():
    model, data = load_scene()
    before = [data.joint(name).qpos[0] for name in ARM_JOINTS]

    result = move_to(model, data, 0.60, 0.0, 0.02)

    assert isinstance(result, str) and result.startswith("unreachable")
    after = [data.joint(name).qpos[0] for name in ARM_JOINTS]
    assert after == pytest.approx(before, abs=1e-6)