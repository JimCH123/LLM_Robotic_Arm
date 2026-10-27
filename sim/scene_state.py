"""Roadmap 1.2: read the true state of the tabletop scene from the simulator."""

from pathlib import Path

import mujoco

SCENE_PATH = Path(__file__).resolve().parent / "scene.xml"
CUBES = ["red_cube", "green_cube", "blue_cube"]
BINS = ["left_bin", "right_bin"]
BIN_INNER_HALF_WIDTH = 0.034  # inside of a bin: 0.04 half-size minus the 0.006 wall
BIN_WALL_HEIGHT = 0.03


def load_scene():
    """Load the scene and return (model, data) with positions computed."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def which_bin(data, position):
    """Return the name of the bin that contains `position`, or None."""
    x, y, z = position
    for bin_name in BINS:
        bx, by, _ = data.body(bin_name).xpos
        inside_x = abs(x - bx) < BIN_INNER_HALF_WIDTH
        inside_y = abs(y - by) < BIN_INNER_HALF_WIDTH
        below_rim = z < BIN_WALL_HEIGHT
        if inside_x and inside_y and below_rim:
            return bin_name
    return None


def get_scene_state(data):
    """Return each cube's position (meters, rounded to 1 mm) and which bin it is in."""
    state = {}
    for cube in CUBES:
        position = data.body(cube).xpos
        state[cube] = {
            "position": [round(float(v), 3) for v in position],
            "in_bin": which_bin(data, position),
        }
    return state