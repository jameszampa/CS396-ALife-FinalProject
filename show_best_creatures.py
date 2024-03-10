import os
import json
import time
import mujoco
import mujoco.viewer


def simulate(file_path, motor_strength_dict):
    m = mujoco.MjModel.from_xml_path(file_path)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set camera parameters
        # These parameters can be adjusted to change the camera angle and perspective
        viewer.cam.azimuth = 180  # Azimuthal angle (in degrees)
        viewer.cam.elevation = -20  # Elevation angle (in degrees)
        viewer.cam.distance = 15.0  # Distance from the camera to the target
        viewer.cam.lookat[0] = 0.0  # X-coordinate of the target position
        viewer.cam.lookat[1] = 0.0  # Y-coordinate of the target position
        viewer.cam.lookat[2] = 0.0  # Z-coordinate of the target position

        for i in range(10000):
            for m_name in motor_strength_dict:
                # print(m_name)
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m_name)
                if motor_strength_dict[m_name] > 0:
                    d.ctrl[i] = 1000
                else:
                    d.ctrl[i] = -1000
            
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(1/1000)

        viewer.close()

if __name__ == "__main__":
    # Load the best creatures
    best_creatures = []
    for file_name in os.listdir("best_creatures"):
        if file_name.endswith(".xml") and file_name.startswith("gen"):
            gen_num = int(file_name.split("_")[0].replace("gen", ""))
            best_creatures.append((f"best_creatures{os.sep}{file_name}", gen_num))
    best_creatures.sort(key=lambda x: x[1])
    best_creatures = [x[0] for x in best_creatures]

    # load motor strength
    motor_strength_dict = []
    for file_name in os.listdir("best_creatures"):

        if file_name.endswith(".json"):
            with open(f"best_creatures{os.sep}{file_name}") as f:
                motor_strength_dict.append(json.load(f))

    # Simulate the best creatures
    for creature, motor_dict in zip(best_creatures, motor_strength_dict):
        print(creature)
        simulate(creature, motor_dict)