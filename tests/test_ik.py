"""Roadmap 1.3b: the IK's joint angles really put the gripper where we asked.

The judge is MuJoCo's own forward kinematics: set the IK's angles on the arm,
compute where the gripper tip ends up, and compare with the target.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.ik import ARM_JOINTS, solve_ik
from sim.scene_state import load_scene

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")

TARGETS = {
    "red cube": (0.25, -0.06, 0.015),
    "green cube": (0.25, 0.0, 0.015),
    "blue cube": (0.25, 0.06, 0.015),
    "above left bin": (0.18, 0.18, 0.06),
    "above right bin": (0.18, -0.18, 0.06),
    "between cubes and left bin": (0.22, 0.10, 0.04),
}


def forward_kinematics(model, angles):
    """Return (gripper tip position, gripper pointing direction) for the given arm angles."""
    data = mujoco.MjData(model)
    for name, angle in zip(ARM_JOINTS, angles):
        data.joint(name).qpos = angle
    mujoco.mj_kinematics(model, data)
    tip = data.site("gripperframe")
    pointing = tip.xmat.reshape(3, 3)[:, 0]  # the site's x axis points out of the gripper
    return tip.xpos.copy(), pointing.copy()


def tilt_from_straight_down_deg(pointing):
    return math.degrees(math.acos(np.clip(-pointing[2], -1.0, 1.0)))


@pytest.fixture(scope="module")
def model():
    return load_scene()[0]


@pytest.mark.parametrize("name", TARGETS)
def test_ik_puts_the_gripper_on_target(model, name):
    target = np.array(TARGETS[name])
    angles = solve_ik(model, *target)
    assert angles is not None, f"no solution for {name}"

    tip, pointing = forward_kinematics(model, angles)
    error_mm = np.linalg.norm(tip - target) * 1000
    assert error_mm < 2.0, f"{name}: gripper is {error_mm:.1f} mm from the target"
    assert tilt_from_straight_down_deg(pointing) < 1.0


def test_tilt_reaches_a_point_the_wrist_limit_blocks_straight_down(model):
    above_green = (0.25, 0.0, 0.095)  # 8 cm above the green cube
    assert solve_ik(model, *above_green, psi=0.0) is None  # wrist_flex would pass its limit

    angles = solve_ik(model, *above_green, psi=math.radians(15))
    assert angles is not None
    tip, pointing = forward_kinematics(model, angles)
    assert np.linalg.norm(tip - np.array(above_green)) * 1000 < 2.0
    assert tilt_from_straight_down_deg(pointing) == pytest.approx(15.0, abs=1.0)


def test_out_of_reach_returns_none(model):
    assert solve_ik(model, 0.60, 0.0, 0.02) is None