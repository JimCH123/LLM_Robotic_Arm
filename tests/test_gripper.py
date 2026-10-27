"""Roadmap 1.3d: the gripper opens, closes, and can tell whether it is holding something."""

from pathlib import Path

import numpy as np
import pytest

from sim.scene_state import get_scene_state, load_scene
from sim.skills import close_gripper, is_holding, move_to, open_gripper

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")

FIXED_TIPS = ["fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3"]
MOVING_TIPS = ["moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"]
GREEN_CUBE = (0.25, 0.0, 0.015)


def fingertip_gap_mm(model, data):
    """Smallest distance between the two fingers' tip spheres (surface to surface)."""
    gaps = []
    for a in FIXED_TIPS:
        for b in MOVING_TIPS:
            centers = np.linalg.norm(data.geom(a).xpos - data.geom(b).xpos)
            gaps.append(centers - model.geom(a).size[0] - model.geom(b).size[0])
    return min(gaps) * 1000


def gripper_around_green_cube(model, data):
    """Open, then lower the open gripper around the green cube (not closed yet)."""
    open_gripper(model, data)
    assert move_to(model, data, GREEN_CUBE[0], GREEN_CUBE[1], 0.06) is None
    assert move_to(model, data, *GREEN_CUBE, duration=1.0) is None


def test_open_gripper_spreads_the_fingers_wider_than_a_cube():
    model, data = load_scene()
    close_gripper(model, data)
    open_gripper(model, data)
    assert fingertip_gap_mm(model, data) > 40  # the cube is 30 mm wide


def test_closing_on_nothing_shuts_the_fingers_and_holds_nothing():
    model, data = load_scene()
    open_gripper(model, data)
    close_gripper(model, data)
    assert fingertip_gap_mm(model, data) < 5
    assert is_holding(data) is False


def test_closing_around_a_cube_holds_it():
    model, data = load_scene()
    gripper_around_green_cube(model, data)
    assert is_holding(data) is False  # still open

    close_gripper(model, data)

    assert is_holding(data) is True
    assert 25 < fingertip_gap_mm(model, data) < 35  # stopped by the 30 mm cube


def test_opening_again_releases_the_cube():
    model, data = load_scene()
    gripper_around_green_cube(model, data)
    close_gripper(model, data)
    open_gripper(model, data)

    assert is_holding(data) is False
    green = get_scene_state(data)["green_cube"]["position"]
    assert green[2] == pytest.approx(0.015, abs=0.003)  # back on the table