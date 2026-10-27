"""Roadmap 1.2: get_scene_state() reports where each cube is and which bin it is in."""

from pathlib import Path

import mujoco
import pytest

from sim.scene_state import CUBES, get_scene_state, load_scene

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")


def test_cubes_start_on_the_table_outside_the_bins():
    _, data = load_scene()
    state = get_scene_state(data)

    assert set(state) == set(CUBES)
    assert state["green_cube"]["position"] == [0.25, 0.0, 0.015]
    for cube in CUBES:
        assert state[cube]["in_bin"] is None


def test_cube_moved_into_a_bin_is_reported_in_that_bin():
    model, data = load_scene()

    # Teleport the red cube into the left bin: a free joint's qpos is [x, y, z, qw, qx, qy, qz].
    data.joint("red_cube").qpos[:3] = [0.18, 0.18, 0.02]
    mujoco.mj_forward(model, data)

    state = get_scene_state(data)
    assert state["red_cube"]["in_bin"] == "left_bin"
    assert state["blue_cube"]["in_bin"] is None