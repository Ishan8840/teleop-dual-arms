import mujoco

MODEL_PATH = "aloha/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("=== JOINTS ===")
for i in range(model.njnt):
    j = model.joint(i)
    print(i, j.name, "qposadr:", j.qposadr[0])
