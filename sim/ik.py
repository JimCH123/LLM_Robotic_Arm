"""Roadmap 1.3b: analytic inverse kinematics for the SO-101.

Adapted from my 5-joint IK (originally MATLAB). Same method:
base angle -> wrist point -> two-link law of cosines -> wrist angle from the
gripper orientation. Changed for the SO-101: pan-axis and shoulder offsets,
zero-pose angle offsets, and MuJoCo's joint signs.

All angles are in radians. The arm moves in a vertical plane: "r" is the
distance forward from the shoulder_pan axis, "z" is the height.
"""

import math
from sim.arm_geometry import measure_arm_geometry

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def solve_ik(model, x, y, z, psi=0.0, roll=0.0):
    """Return the 5 joint angles (radians) that put the gripper tip at (x, y, z).

    psi:  gripper tilt from pointing straight down (0 = straight down).
    roll: wrist_roll angle, passed straight through.
    Returns None if the point is out of reach or a joint would pass its limit.
    """
    g = measure_arm_geometry(model)
    # print(g)
    
    L1, L2, L3 = g["L1"], g["L2"], g["L3"]
    
    # Step 1 - base angle, measured around the pan axis (not the origin).
    #   MATLAB: theta1 = atan2(ye, xe)
    theta1 = math.atan2(y, x - g["pan_x"])
    r = math.hypot(x - g["pan_x"], y)

    # Step 2 - wrist point: step back from the tip along the gripper by L3.
    #   MATLAB: xe_new = xe - sin(psi)*L3*cos(theta1); ze_new = ze + cos(psi)*L3
    #   psi - pi/2 is the gripper direction; a3 is the L3 link's zero-pose offset.
    phi3 = psi - math.pi / 2 + g["a3"]
    wr = r - L3 * math.cos(phi3)
    wz = z - L3 * math.sin(phi3)

    # Step 3 - move the origin to the shoulder_lift axis.
    #   MATLAB used (ze_new - d1); the SO-101 shoulder is also shoulder_r forward.
    dr = wr - g["shoulder_r"]
    dz = wz - g["shoulder_z"]

    # Step 4 - law of cosines for the elbow.
    #   MATLAB clamped |D| > 1 to 1, which hides unreachable targets. Return None instead.
    cos_elbow = (dr**2 + dz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(cos_elbow) > 1:
        return None  # the wrist point is too far from (or too close to) the shoulder
    elbow = -math.acos(cos_elbow)  # elbow-up solution, like theta3 = -theta3 in MATLAB
    
    # Step 5 - absolute angle of each link above horizontal.
    #   MATLAB: theta2 = beta + alpha
    phi1 = math.atan2(dz, dr) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
    phi2 = phi1 + elbow

    # Step 6 - convert to MuJoCo joint angles. Subtract each link's zero-pose angle
    #   (a1..a3) and flip signs: in MuJoCo positive pitch tilts the arm down, and the
    #   pan axis points along -z.
    #   MATLAB: theta4 = psi - theta2 - theta3
    q_pan = -theta1
    q_lift = g["a1"] - phi1
    q_elbow = g["a2"] - q_lift - phi2
    q_wrist = g["a3"] - q_lift - q_elbow - phi3
    q_roll = roll
    angles = [q_pan, q_lift, q_elbow, q_wrist, q_roll]
    

    # Step 7 - joint limits: an angle the motor cannot reach is no solution.
    for name, angle in zip(ARM_JOINTS, angles):
        low, high = model.joint(name).range
        if not low <= angle <= high:
            return None

    return angles
    
    

if __name__ == "__main__":
    import mujoco
    from sim.scene_state import load_scene

    model, data = load_scene()
    # Check the IK with the gripper position at the zero pose。
    solve_ik(model, 0.3913614999999999, -0.0009794652485835562, 0.24634517669302677, psi = math.pi/2)