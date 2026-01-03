import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from importlib.resources import files


XML_PATH = str(files("dual_arms.aloha").joinpath("scene.xml"))

def main():
    # 1. Initialize Robot
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    H, W = 240, 426   # per-camera size (keeps it fast)
    renderer = mujoco.Renderer(m, height=H, width=W)

    # Cameras to show
    sim_cams = ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"]

    available = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)}
    sim_cams = [c for c in sim_cams if c in available]
    print("Sim cameras:", sim_cams)

    def render_sim_cam(cam_name):
        cam_id = m.camera(cam_name).id
        renderer.update_scene(d, camera=cam_id)
        img = renderer.render()
        return img[:, :, ::-1]

    # Reset Pose
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    # Setup Mocap/Ghost
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
    mocap_id = m.body_mocapid[body_id]
    effector_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')

    gripper_name = "right/gripper"
    gripper_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_name)
    
    gripper_range = m.actuator_ctrlrange[gripper_id]
    
    # Snap ghost to initial hand position
    d.mocap_pos[mocap_id] = d.site_xpos[effector_id]

    # Map Actuators
    actuator_to_joint = [m.actuator_trnid[i, 0] for i in range(m.nu)]


    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()


            for _ in range(5): 
                mujoco.mj_step(m, d)

            if len(sim_cams) > 0:
                frames = [render_sim_cam(c) for c in sim_cams]

                if len(frames) == 1:
                    grid = frames[0]
                elif len(frames) == 2:
                    grid = np.hstack(frames)
                elif len(frames) == 3:
                    blank = np.zeros_like(frames[0])
                    grid = np.vstack([np.hstack(frames[:2]), np.hstack([frames[2], blank])])
                else:
                    grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])

                cv2.imshow("MuJoCo Cameras", grid)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            viewer.sync()

if __name__ == "__main__":
    main()