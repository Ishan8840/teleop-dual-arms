import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from scipy.spatial.transform import Rotation as R
from importlib.resources import files

# Import your class
from .rotation import HandOrientationTuner

XML_PATH = str(files("dual_arms.aloha").joinpath("scene.xml"))

def euler_to_quat(roll, pitch, yaw):
    # Convert RPY to MuJoCo Quaternion [w, x, y, z]
    r = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])

def main():
    # 1. Initialize Robot
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Go to neutral pose
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    # 2. Setup Ghost (Mocap)
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
    mocap_id = m.body_mocapid[body_id]
    effector_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')

    # LOCK POSITION: Save the initial position of the hand
    fixed_position = d.site_xpos[effector_id].copy()
    d.mocap_pos[mocap_id] = fixed_position
    
    # Initialize Mocap orientation to match current gripper orientation
    # We must convert the site's matrix to a quaternion first
    init_mat = d.site_xmat[effector_id].reshape(9)
    init_quat = np.zeros(4)
    mujoco.mju_mat2Quat(init_quat, init_mat)
    d.mocap_quat[mocap_id] = init_quat

    # Map Actuators for easy access
    actuator_to_joint = [m.actuator_trnid[i, 0] for i in range(m.nu)]

    # 3. Initialize Vision
    tracker = HandOrientationTuner()

    print("--- Rotation Test Mode ---")
    print("The hand position is LOCKED.")
    print("Only orientation (Roll/Pitch/Yaw) will change.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. VISION UPDATE ---
            rotation, frame = tracker.get_orientation()
            
            if rotation is not None:
                # 1. Update ONLY the rotation of the ghost
                roll, pitch, yaw = rotation
                quat = euler_to_quat(roll, pitch, yaw)
                d.mocap_quat[mocap_id] = quat
            
            # 2. Force the position to remain locked
            d.mocap_pos[mocap_id] = fixed_position

            # Show Camera
            if frame is not None:
                cv2.imshow("Rotation Test", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            # --- B. ROBOT CONTROL (6-DOF IK) ---
            
            # 1. Position Error (Should be zero, keeping it locked)
            current_pos = d.site_xpos[effector_id]
            err_pos = d.mocap_pos[mocap_id] - current_pos

            # 2. Rotation Error (Corrected)
            target_quat = d.mocap_quat[mocap_id]
            
            # FIX: Get current orientation from Rotation Matrix (site_xmat)
            current_mat = d.site_xmat[effector_id].reshape(9)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat)
            
            # Compute quaternion difference
            q_err = np.zeros(4)
            mujoco.mju_mulQuat(q_err, target_quat, current_quat * np.array([1, -1, -1, -1]))
            
            err_rot = np.zeros(3)
            mujoco.mju_quat2Vel(err_rot, q_err, 1.0) 

            # Combine Errors
            error = np.hstack([err_pos, err_rot])

            # 3. Solve IK
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, jacr, effector_id)
            J = np.vstack([jacp, jacr]) # Stack to 6xN

            dq = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6)) @ error * 5.0

            # 4. Apply
            for i in range(m.nu):
                if "gripper" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): continue
                jid = actuator_to_joint[i]
                d.ctrl[i] = d.qpos[jid] + dq[jid]

            # --- C. STEP ---
            for _ in range(5): 
                mujoco.mj_step(m, d)
            viewer.sync()

    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()