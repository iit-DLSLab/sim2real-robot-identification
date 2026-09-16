import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../mujoco/")
sys.path.append(dir_path+"/../")

import mujoco
import mujoco.viewer
import config

import numpy as np
import copy
import time
import torch

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # Create the environment -----------------------------------------------------------
    mjModel = mujoco.MjModel.from_xml_path(dir_path + "/../robot_model/" + config.robot + "/scene_flat.xml")
    mjData = mujoco.MjData(mjModel)

    viewer = mujoco.viewer.launch_passive(
            mjModel,
            mjData,
            show_left_ui=False,
            show_right_ui=False,
    )

    freezed_base_position = np.array([0, 0, 0.4])
    freezed_base_orientation = np.array([1, 0, 0, 0])
    freezed_base_linear_velocity = np.array([0, 0, 0.])
    freezed_base_angular_velocity = np.array([0, 0, 0])
    

    # Load datasets for calibration
    datasets_path = dir_path + "/" + config.robot + "/trajectory_1.pt"
    try:
        data = torch.load(datasets_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(datasets_path, map_location="cpu")
    dataset_actual_joint_pos = data["dof_pos"]
    dataset_time = data["time"]

    timestep = 0

    while True:
        
        print(f"Running timestep: {timestep}")
        
        if(timestep >= dataset_actual_joint_pos.shape[0] - 1):
            print("End of dataset reached, resetting to the beginning.")
            break
        
        joint_pos = dataset_actual_joint_pos[timestep]
        
        mjData.qpos[0:3] = copy.deepcopy(freezed_base_position)
        mjData.qpos[3:7] = copy.deepcopy(freezed_base_orientation)
        mjData.qvel[0:3] = copy.deepcopy(freezed_base_linear_velocity)
        mjData.qvel[3:6] = copy.deepcopy(freezed_base_angular_velocity)
        mjData.qpos[7:19] = copy.deepcopy(joint_pos)

        if(timestep > 0):
            mjModel.opt.timestep = dataset_time[timestep] - dataset_time[timestep-1]
        else:
            mjModel.opt.timestep = dataset_time[timestep]


        mujoco.mj_forward(mjModel, mjData) 
        timestep += 1

        viewer.sync()
        #time.sleep(float(dataset_time[timestep]))
