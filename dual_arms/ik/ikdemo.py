from importlib.resources import files
import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = str(files("dual_arms.aloha").joinpath("aloha.xml"))

# Load model
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

# --- KEYFRAME RESET ---
key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
mujoco.mj_resetDataKeyframe(m, d, key_id)
mujoco.mj_forward(m, d)

# --- SETUP GHOST ---
body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
mocap_id = m.body_mocapid[body_id]
effector_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')
d.mocap_pos[mocap_id] = d.site_xpos[effector_id]

# --- CRITICAL FIX: ACTUATOR MAPPING ---
# Create a map: actuator_index -> joint_index
# This ensures we send the correct velocity to the correct motor.
actuator_to_joint = []
for i in range(m.nu): # For every actuator
    # trnid gives the ID of the object this actuator controls.
    # The first column [i, 0] is the joint ID.
    joint_id = m.actuator_trnid[i, 0]
    actuator_to_joint.append(joint_id)

print(f"Mapped {len(actuator_to_joint)} actuators to their joints.")

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 1. READ TARGET
        # Use the fixed mocap lookup
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
        mocap_id = m.body_mocapid[body_id]
        target_pos = d.mocap_pos[mocap_id]

        # 2. READ END-EFFECTOR
        effector_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')
        current_pos = d.site_xpos[effector_id]

        # 3. CALCULATE ERROR
        error = target_pos - current_pos
        
        # Scale down error if it's too big (Prevents explosion if target is far)
        if np.linalg.norm(error) > 0.05:
            error = error / np.linalg.norm(error) * 0.05

        # 4. SOLVE IK
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, effector_id)

        # Since we only want to move the RIGHT arm, we should strictly speaking
        # zero out the Jacobian columns for the left arm to prevent drift,
        # but usually they are already 0 because the left arm doesn't move the right hand.
        
        J = jacp.reshape((3, m.nv))
        
        # Damped Least Squares
        dq = J.T @ np.linalg.inv(J @ J.T + np.eye(3) * 1e-4) @ error * 5.0

        # 5. APPLY CONTROL (THE FIX)
        # Iterate over all actuators and apply the velocity from the CORRECT joint
        for i in range(m.nu):
            joint_id = actuator_to_joint[i]
            
            # We are using Position Control Actuators (likely), so:
            # ctrl = current_joint_angle + desired_velocity * dt
            d.ctrl[i] = d.qpos[joint_id] + dq[joint_id] * 0.5

        mujoco.mj_step(m, d)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)