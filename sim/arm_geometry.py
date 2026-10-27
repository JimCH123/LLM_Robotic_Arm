"""Roadmap 1.3a: measure the SO-101 arm's geometry from the MuJoCo model.

The IK (roadmap 1.3b) needs these numbers. Measuring them in code, instead of
typing them in by hand, keeps the IK correct if the Menagerie model changes.

Units: meters and radians. The arm works in a vertical plane:
"r" is the distance forward from the shoulder_pan axis, "z" is the height.
"""

import math

import mujoco

GRIPPER_SITE = "gripperframe"  # the point between the gripper fingers

def distance(start, end):
    """Straight-line distance between two (x, z) points."""
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    return math.hypot(dx, dz), dx, dz

def measure_arm_geometry(model):
    """Return the arm's link lengths and zero-pose angles, measured with every joint at 0."""
    data = mujoco.MjData(model)  # a fresh MjData: every joint angle starts at 0
    mujoco.mj_kinematics(model, data)  # compute positions for that pose

    # TODO 1: read the (x, z) position of these joint axes:
    #         shoulder_pan, shoulder_lift, elbow_flex, wrist_flex
    #         hint: data.joint("elbow_flex").xanchor gives [x, y, z]
    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    # print(data.joint("shoulder_pan"))
    points = {}
    for name in JOINTS:
        x,_, z = data.joint(name).xanchor
        points[name] = (x, z)
    

    
    # TODO 2: read the (x, z) position of the gripper tip
    #         hint: data.site(GRIPPER_SITE).xpos gives [x, y, z]
    x,_, z = data.site(GRIPPER_SITE).xpos
    points["tip"] = (x, z)
    
    # TODO 3: link lengths
    #         L1: shoulder_lift -> elbow_flex
    #         L2: elbow_flex -> wrist_flex
    #         L3: wrist_flex -> gripper tip
    #         hint: math.hypot(dx, dz)
    L1, _, _ = distance(points["shoulder_lift"], points["elbow_flex"])  # upper arm
    L2, _, _ = distance(points["elbow_flex"], points["wrist_flex"])     # forearm
    L3, _, _ = distance(points["wrist_flex"], points["tip"])  

    # TODO 4: zero-pose angle of each link, measured up from horizontal
    #         a1 for L1, a2 for L2, a3 for L3
    #         hint: math.atan2(dz, dx)
    _, dx, dz= distance(points["shoulder_lift"], points["elbow_flex"])
    a1 = math.atan2(dz, dx)
    _, dx, dz = distance(points["elbow_flex"], points["wrist_flex"]) 
    a2 = math.atan2(dz, dx)
    _, dx, dz = distance(points["wrist_flex"], points["tip"])  
    a3 = math.atan2(dz, dx) 

    return {
        "pan_x": points["shoulder_pan"][0],       # x of the shoulder_pan axis
        "shoulder_r": points["shoulder_lift"][0] - points["shoulder_pan"][0],     # shoulder_lift axis: distance forward of the pan axis
        "shoulder_z": points["shoulder_lift"][1],  # shoulder_lift axis: height
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    } 
    


if __name__ == "__main__":
    from sim.scene_state import load_scene
    model, _ = load_scene()
    measure_arm_geometry(model)

