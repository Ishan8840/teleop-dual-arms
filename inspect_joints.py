import mujoco
model = mujoco.MjModel.from_xml_path("aloha/scene.xml")  # scene with your Cartesian actuators
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
