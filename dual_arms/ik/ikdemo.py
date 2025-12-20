import mujoco as mj
import numpy as np

# MINK imports
from mink.configuration import Configuration
from mink.lie.se3 import SE3
from mink.tasks.frame_task import FrameTask
from mink.solve_ik import solve_ik
from mink.limits.configuration_limit import ConfigurationLimit
from importlib.resources import files


XML_PATH = str(files("dual_arms.aloha").joinpath("aloha.xml"))
# ---- 0. Paths / constants ----
dt = 0.02                # control timestep (50 Hz)

# ---- 1. Load MuJoCo model + data ----
model = mj.MjModel.from_xml_path(XML_PATH)
config = Configuration(model)          # <-- no second positional arg
data   = config.data    
config.update_from_keyframe("neutral_pose")

# End-effector for RIGHT arm (from your XML: <site name="right/gripper" .../>)
EE_SITE_NAME = "right/gripper"
EE_SITE_TYPE = "site"

# Ghost visualization site we added in the XML
GHOST_SITE_NAME = "cartesian_target"
ghost_site_id = mj.mj_name2id(model, mj.mjtObj.
                              mjOBJ_SITE, GHOST_SITE_NAME)

# ---- 2. Define a FrameTask for the right gripper ----
# This says: "Move the site 'right/gripper' to a target pose in the world"
ee_task = FrameTask(
    frame_name=EE_SITE_NAME,
    frame_type=EE_SITE_TYPE,
    position_cost=1.0,   # care about position
    orientation_cost=0.0 # ignore orientation for now
)

# Basic joint position limits for safety
limits = [ConfigurationLimit(model)]


def clamp_workspace(xyz: np.ndarray) -> np.ndarray:
    """
    Clamp the target into a safe-ish box so we don't ask for something crazy.
    Tune these numbers to your setup.
    """
    x, y, z = xyz
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    z = np.clip(z, -1.0, 1.0)
    return np.array([x, y, z])


def step_towards_xyz(target_xyz: np.ndarray):
    """
    One IK + physics step that nudges the right gripper toward target_xyz.
    """

    # 1) Make sure the target is within some workspace
    target_xyz = clamp_workspace(target_xyz)

    # 2) Move the ghost site there (visual only)
    data.site_xpos[ghost_site_id] = target_xyz

    # 3) Set the FrameTask target to that position (no orientation)
    target_pose = SE3.from_translation(target_xyz)
    ee_task.set_target(target_pose)

    # 4) Update kinematics
    config.update()

    # 5) Solve differential IK: get joint-space velocity v
    v = solve_ik(
        configuration=config,
        tasks=[ee_task],
        dt=dt,
        solver="daqp",   # use whichever solver MINK supports in your env
        damping=1e-4,      # a bit of Levenberg-Marquardt damping
        limits=limits,
        constraints=None,
    )

    v *= 0.01

    # 6) Integrate that velocity for dt to get new joint angles
    config.integrate_inplace(v, dt)
    data.qpos[:] = config.q

    # 7) Step MuJoCo physics once
    mj.mj_step(model, data)

target_xyz = np.array([0.2, -0.25, 0.1], dtype=float)

def key_callback(keycode: int):
    """
    Adjust target_xyz with keyboard keys.

    w/s -> x forward/back
    a/d -> y left/right
    r/f -> z up/down
    """
    global target_xyz

    step = 0.1  # how much to move per key press

    try:
        ch = chr(keycode)
    except ValueError:
        # non-printable key
        print("non-printable key:", keycode)
        return

    ch = ch.lower()  # handle caps / uppercase
    print("key pressed:", keycode, repr(ch), "current target:", target_xyz)

    if ch == '1':
        target_xyz[0] += step
    elif ch == 's':
        target_xyz[0] -= step
    elif ch == 'a':
        target_xyz[1] += step
    elif ch == 'd':
        target_xyz[1] -= step
    elif ch == 'r':
        target_xyz[2] += step
    elif ch == 'f':
        target_xyz[2] -= step

    print("updated target:", target_xyz)


if __name__ == "__main__":
    from mujoco import viewer

    with viewer.launch_passive(model, data, key_callback=key_callback) as v:
        while v.is_running():
            with v.lock():               # 🔒 safe to modify data/model
                step_towards_xyz(target_xyz)
            v.sync()         
