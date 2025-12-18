import time
import numpy as np
import mujoco
import mujoco.viewer as viewer

MODEL_PATH = "aloha/scene.xml"  # adjust if your path is different

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ----- joint indices from your inspect_joints.py output -----
LEFT_ARM_QPOS_IDX = [0, 1, 2, 3, 4, 5]      # waist → wrist_rotate
RIGHT_ARM_QPOS_IDX = [8, 9, 10, 11, 12, 13] # waist → wrist_rotate

# save home pose so we oscillate around it
left_home = data.qpos[LEFT_ARM_QPOS_IDX].copy()
right_home = data.qpos[RIGHT_ARM_QPOS_IDX].copy()

t = 0.0
dt = 0.01  # 100 Hz sim, but we’ll sleep to keep things light

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        t += dt

        amp = 0.3   # radians of swing (about 17 degrees)
        freq = 0.4  # Hz – slow wave

        # left arm: each joint has slightly different phase
        for k, idx in enumerate(LEFT_ARM_QPOS_IDX):
            data.qpos[idx] = left_home[k] + amp * np.sin(2 * np.pi * freq * t + 0.5 * k)

        # right arm
        for k, idx in enumerate(RIGHT_ARM_QPOS_IDX):
            data.qpos[idx] = right_home[k] + amp * np.sin(2 * np.pi * freq * t + 0.5 * k)

        # tell MuJoCo we changed qpos directly
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        v.sync()
        time.sleep(dt)  # throttle CPU
