"""Roadmap 1.1: the tabletop scene loads and the cubes rest still on the table."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "sim" / "scene.xml"
SO101 = REPO_ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
CUBES = ["red_cube", "green_cube", "blue_cube"]

pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    return model, data


def test_scene_has_arm_cubes_and_bins():
    model, _ = load_scene()
    assert model.nu == 6  # the arm's 6 motors
    for name in CUBES + ["left_bin", "right_bin"]:
        model.body(name)  # raises KeyError if the body is missing


def test_cubes_stay_still_for_two_seconds():
    model, data = load_scene()
    mujoco.mj_forward(model, data)
    start = {name: data.body(name).xpos.copy() for name in CUBES}
    # print(data.body("red_cube"))

    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    for name in CUBES:
        moved = np.linalg.norm(data.body(name).xpos - start[name])
        assert moved < 0.005, f"{name} moved {moved:.4f} m"