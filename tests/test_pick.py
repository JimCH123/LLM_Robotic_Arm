"""Roadmap 1.3e: pick() lifts the requested cube, and explains why when it cannot."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

import sim.skills
from sim.scene_state import CUBES, get_scene_state, load_scene
from sim.skills import is_holding, pick

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")


def positions(data):
    return {name: np.array(info["position"]) for name, info in get_scene_state(data).items()}


@pytest.mark.parametrize("cube", CUBES)
def test_pick_lifts_the_cube_and_leaves_the_others(cube):
    model, data = load_scene()
    before = positions(data)

    result = pick(model, data, cube)

    assert result is None, result
    assert is_holding(data)
    after = positions(data)
    assert after[cube][2] > 0.06, f"{cube} is only at z = {after[cube][2]}"
    for other in CUBES:
        if other != cube:
            assert np.linalg.norm(after[other] - before[other]) < 0.002, f"{other} was bumped"


def test_unknown_cube_is_refused():
    model, data = load_scene()
    result = pick(model, data, "purple_cube")
    assert isinstance(result, str) and result.startswith("unknown cube")


def test_cannot_pick_while_already_holding_a_cube():
    model, data = load_scene()
    assert pick(model, data, "green_cube") is None

    result = pick(model, data, "red_cube")

    assert isinstance(result, str) and result.startswith("already holding")


def test_reports_a_miss_when_the_cube_is_not_where_the_robot_thinks(monkeypatch):
    # Simulate a perception error: the robot "sees" the green cube 7 cm further away than it is.
    real_get_scene_state = sim.skills.get_scene_state

    def wrong_scene_state(data):
        state = real_get_scene_state(data)
        state["green_cube"]["position"] = [0.32, 0.0, 0.015]
        return state

    monkeypatch.setattr(sim.skills, "get_scene_state", wrong_scene_state)
    model, data = load_scene()

    result = pick(model, data, "green_cube")

    assert isinstance(result, str) and result.startswith("missed")
    assert not is_holding(data)


def test_passes_on_the_move_to_reason_when_the_cube_is_out_of_reach():
    model, data = load_scene()
    data.joint("green_cube").qpos[:3] = [0.55, 0.0, 0.015]  # beyond the arm's reach
    mujoco.mj_forward(model, data)

    result = pick(model, data, "green_cube")

    assert isinstance(result, str) and result.startswith("unreachable")
    assert not is_holding(data)