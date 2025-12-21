import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from importlib.resources import files
from .vision import HandTracker


XML_PATH = str(files("dual_arms.aloha").joinpath("aloha.xml"))

def main():
    # 1. Initialize Robot
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Reset Pose
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    # Setup Mocap/Ghost
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
    mocap_id = m.body_mocapid[body_id]
    effector_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')
    
    # Snap ghost to initial hand position
    d.mocap_pos[mocap_id] = d.site_xpos[effector_id]

    # Map Actuators
    actuator_to_joint = [m.actuator_trnid[i, 0] for i in range(m.nu)]

    # 2. Initialize Vision
    tracker = HandTracker()

    print("Starting Teleop... Press ESC in the Camera window to quit.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. GET INPUT FROM VISION ---
            target_pos, frame = tracker.get_target()

            # If we see a hand, update the ghost
            if target_pos is not None:
                d.mocap_pos[mocap_id] = target_pos

            # Show Camera Feed
            if frame is not None:
                cv2.imshow("Teleop Eye", frame)
                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    break

            # --- B. ROBOT CONTROL (IK) ---
            
            # 1. Calculate Error
            current_target = d.mocap_pos[mocap_id]
            current_hand = d.site_xpos[effector_id]
            error = current_target - current_hand

            # Safety clamp for error
            if np.linalg.norm(error) > 0.05:
                error = error / np.linalg.norm(error) * 0.05

            # 2. Jacobian IK
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, jacr, effector_id)
            
            J = jacp.reshape((3, m.nv))
            dq = J.T @ np.linalg.inv(J @ J.T + np.eye(3) * 1e-4) @ error * 3.0

            # 3. Apply to Motors
            for i in range(m.nu):
                jid = actuator_to_joint[i]
                d.ctrl[i] = d.qpos[jid] + dq[jid]

            # --- C. STEP PHYSICS ---
            for _ in range(5): 
                mujoco.mj_step(m, d)

            viewer.sync()

    # Cleanup
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()