"""Roadmap 1.3f: place() puts the held cube into a bin, and explains why when it cannot."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

import sim.skills
from sim.scene_state import BINS, CUBES, get_scene_state, load_scene
from sim.skills import held_cube, is_holding, pick, place

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")


@pytest.mark.parametrize("bin_name", BINS)
@pytest.mark.parametrize("cube", CUBES)
def test_place_puts_cube_in_bin(cube, bin_name):
    model, data = load_scene()
    assert pick(model, data, cube) is None
    assert held_cube(data) == cube
    others = {c: data.body(c).xpos.copy() for c in CUBES if c != cube}

    assert place(model, data, bin_name) is None
    assert get_scene_state(data)[cube]["in_bin"] == bin_name
    assert not is_holding(data)
    for c, start in others.items():
        assert np.linalg.norm(data.body(c).xpos - start) < 0.002  # neighbours untouched


def test_held_cube_is_none_when_empty():
    model, data = load_scene()
    assert held_cube(data) is None


def test_unknown_bin():
    model, data = load_scene()
    assert pick(model, data, "red_cube") is None
    result = place(model, data, "purple_bin")
    assert result.startswith("unknown bin")
    assert is_holding(data)  # nothing happened, still holding the cube


def test_not_holding():
    model, data = load_scene()
    result = place(model, data, "left_bin")
    assert result.startswith("not holding anything")


def test_unreachable_bin_keeps_the_cube():
    model, data = load_scene()
    assert pick(model, data, "green_cube") is None
    model.body("left_bin").pos = [0.55, 0.0, 0.0]  # move the bin out of reach
    mujoco.mj_forward(model, data)
    result = place(model, data, "left_bin")
    assert result.startswith("unreachable")
    assert is_holding(data)  # the cube was not dropped on the way


def test_missed_bin_is_reported(monkeypatch):
    model, data = load_scene()
    assert pick(model, data, "blue_cube") is None
    # Pretend the ground truth says the cube landed outside the bin.
    real = sim.skills.get_scene_state

    def fake_state(d):
        state = real(d)
        state["blue_cube"] = {"position": [0.25, 0.1, 0.015], "in_bin": None}
        return state

    monkeypatch.setattr(sim.skills, "get_scene_state", fake_state)
    result = place(model, data, "right_bin")
    assert result.startswith("missed the bin")