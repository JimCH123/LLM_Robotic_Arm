"""Roadmap 1.3: robot skills built on the IK. Each skill runs the physics simulation.

A skill returns None when it succeeds, or a short reason string when it fails.
Later the LLM agent reads that string to decide what to do next.
"""

import math

import mujoco
import numpy as np

from sim.ik import ARM_JOINTS, solve_ik
from sim.scene_state import BINS, CUBES, get_scene_state

TILTS_TO_TRY_DEG = [0, 15, 30, 45, 60]  # straight down first; tilt only if the wrist limit requires it
ARRIVAL_TOLERANCE_M = 0.005  # the gripper must end within 5 mm of the target
SETTLE_TIME_S = 0.3  # extra time after the motion so the arm comes to rest

# Measured in the simulator: a larger gripper command opens the fingers wider.
#   -0.17 -> 2.8 mm between the fingertips (fully closed)    0.3 -> 37 mm    1.0 -> 86 mm
# Holding a 3 cm cube, the fingers stop with the gripper joint at about +0.18;
# closing on nothing, the joint goes all the way to about -0.17.
GRIPPER_OPEN = 1.0  # about 86 mm: much wider than a 3 cm cube, so aiming is forgiving
GRIPPER_CLOSED = -0.17  # the gripper's lower limit
GRIPPER_TIME_S = 0.6  # time to let the fingers finish moving
HOLDING_THRESHOLD = 0.0  # gripper joint above this after closing = something is between the fingers


# Functions called after every physics step, e.g. to update a viewer window or record a video.
STEP_CALLBACKS = []


def step(model, data):
    """Advance the physics by one timestep, then let every callback see the new state."""
    mujoco.mj_step(model, data)
    for callback in STEP_CALLBACKS:
        callback(model, data)


def run_for(model, data, seconds):
    """Advance the physics for `seconds` without changing any motor targets."""
    for _ in range(int(seconds / model.opt.timestep)):
        step(model, data)


# ---------------------------------------------------------------- Roadmap 1.3c: move_to
def move_to(model, data, x, y, z, duration=1.5):
    """Move the gripper tip to (x, y, z) smoothly over `duration` seconds.

    Returns None on success, or a reason string on failure.
    """
    # Step 1 - find joint angles: try each tilt until solve_ik returns a solution.
    angles = None
    for tilt in TILTS_TO_TRY_DEG:
        angles = solve_ik(model, x, y, z, psi=math.radians(tilt))
        if angles is not None:
            break  # found a solution, stop trying
    if angles is None:
        return f"unreachable: no joint angles for ({x}, {y}, {z})"

    # Step 2 - where the motors are aiming now (start) and where they should aim (goal).
    start = np.array([data.actuator(name).ctrl[0] for name in ARM_JOINTS])
    goal = np.array(angles)

    # Step 3 - move smoothly: each simulation step, aim the motors a little closer to the goal.
    steps = max(1, int(duration / model.opt.timestep))
    for i in range(steps):
        s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / steps)  # 0 -> 1, slow at both ends
        target = start + s * (goal - start)
        for name, value in zip(ARM_JOINTS, target):
            data.actuator(name).ctrl = value
        step(model, data)          

    # Step 4 - let the arm settle: keep simulating without changing the motor targets.
    run_for(model, data, SETTLE_TIME_S)

    # Step 5 - check where the gripper really is (ground truth), not where we asked it to go.
    error = np.linalg.norm(data.site("gripperframe").xpos - np.array([x, y, z]))
    if error > ARRIVAL_TOLERANCE_M:
        return f"did not arrive: gripper is {error * 1000:.1f} mm from the target"
    return None


# ---------------------------------------------------------------- Roadmap 1.3d: gripper
def open_gripper(model, data):
    """Open the fingers wide. Returns None."""
    data.actuator("gripper").ctrl = GRIPPER_OPEN
    run_for(model, data, GRIPPER_TIME_S)
    return None


def close_gripper(model, data):
    """Close the fingers until they meet a cube or each other. Returns None.

    Use is_holding() afterwards to know whether something was caught.
    """
    data.actuator("gripper").ctrl = GRIPPER_CLOSED
    run_for(model, data, GRIPPER_TIME_S)
    return None


def is_holding(data):
    """True if the gripper was told to close but the fingers were stopped by something."""
    # An open gripper also has a large joint angle, so the angle alone is not enough:
    # the gripper must also have been told to close.
    told_to_close = data.actuator("gripper").ctrl[0] < HOLDING_THRESHOLD
    fingers_blocked = data.joint("gripper").qpos[0] > HOLDING_THRESHOLD
    return bool(told_to_close and fingers_blocked)


# ---------------------------------------------------------------- Roadmap 1.3e: pick
APPROACH_HEIGHT_M = 0.045  # first stop: this far above the cube's center
LIFT_HEIGHT_M = 0.08  # gripper height after lifting
LIFTED_MIN_Z_M = 0.04  # the cube's center must be at least this high to count as lifted


def pick(model, data, cube):
    """Pick up `cube` (e.g. "red_cube") and lift it. Returns None, or a reason string."""
    # Step 1 - check the request against the ground truth.
    state = get_scene_state(data)
    if cube not in state:
        return f"unknown cube: {cube} (known cubes: {', '.join(state)})"
    if is_holding(data):
        return "already holding something: place it before picking another cube"

    # Step 2 - where is the cube?
    x, y, z = state[cube]["position"]

    # Step 3 - open, go above the cube, then straight down to the cube's center.
    open_gripper(model, data)
    result = move_to(model, data, x, y, z + APPROACH_HEIGHT_M)
    if result is not None:
        return result
    result = move_to(model, data, x, y, z, duration=1.0)
    if result is not None:
        return result

    # Step 4 - close, and check that something is between the fingers.
    close_gripper(model, data)
    if not is_holding(data):
        return f"missed: the gripper closed on nothing at ({x}, {y}, {z})"

    # Step 5 - lift straight up.
    result = move_to(model, data, x, y, LIFT_HEIGHT_M)
    if result is not None:
        return result

    # Step 6 - check with the ground truth that the cube really came up with the gripper.
    cube_z = get_scene_state(data)[cube]["position"][2]
    if not is_holding(data) or cube_z < LIFTED_MIN_Z_M:
        return f"dropped: {cube} did not come up with the gripper (its center is at z = {cube_z})"
    return None


# ---------------------------------------------------------------- Roadmap 1.3f: place
RELEASE_HEIGHT_M = 0.04  # open the fingers here: the cube's bottom is just below the bin's 3 cm rim


def held_cube(data):
    """Name of the cube between the fingers, or None if the gripper holds nothing."""
    if not is_holding(data):
        return None
    tip = data.site("gripperframe").xpos
    return min(CUBES, key=lambda cube: np.linalg.norm(data.body(cube).xpos - tip))


def place(model, data, bin_name):
    """Put the held cube into `bin_name` (e.g. "left_bin"). Returns None, or a reason string."""
    # Step 1 - check the request: the bin must exist and the gripper must hold a cube.
    if bin_name not in BINS:
        return f"unknown bin: {bin_name} (known bins: {', '.join(BINS)})"
    cube = held_cube(data)
    if cube is None:
        return "not holding anything: pick a cube before placing"

    # Step 2 - where is the bin?
    x, y, _ = data.body(bin_name).xpos

    # Step 3 - carry the cube above the bin, then lower it into the bin.
    result = move_to(model, data, x, y, LIFT_HEIGHT_M)
    if result is not None:
        return result
    result = move_to(model, data, x, y, RELEASE_HEIGHT_M, duration=1.0)
    if result is not None:
        return result

    # Step 4 - let go, and move back up so the gripper is out of the way.
    open_gripper(model, data)
    result = move_to(model, data, x, y, LIFT_HEIGHT_M)
    if result is not None:
        return result

    # Step 5 - check with the ground truth that the cube really ended in the bin.
    cube_state = get_scene_state(data)[cube]
    if cube_state["in_bin"] != bin_name:
        return (f"missed the bin: {cube} ended at {cube_state['position']}, "
                f"in {cube_state['in_bin'] or 'no bin'}")
    return None