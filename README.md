  <div style="display: flex; justify-content: space-around;">
    <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
    <img src="https://img.shields.io/badge/Mujoco%20-v3.13.0-blue" alt="Mujoco v3.13.0" style="margin-bottom: 1px;">
    <div style="display: flex; justify-content: space-around;">
      <img src="./gifs/z1_real.gif" alt="Z1Real" width="49%">
      <img src="./gifs/go2_real.gif" alt="Z1Real" width="49%">
    </div>
  </div>


## Overwiew

A system identification routine for IsaacLab and Mujoco, to estimate **motor parameters**, **inertia**, **mass**, and **center of mass**. It provides scripts for data collection on the real robot (the robot should be in the air with the base fixed if a quadruped, or on a table if a manipulator). 
This repo interfaces directly with [Pace](https://github.com/leggedrobotics/pace-sim2real) and the new [sys-id functionality](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md) of Mujoco.

The models (usd and xml) identified so far can be found in the folder **robot_model**. 

## Supported Robots

<!-- model-dates:start -->
| Robot | last modified XML | last modified USD | Identified |
|:--|:--:|:--:|:--:|
| [a2](./robot_model/a2) | 2026-09-19 | 2026-09-19 | ✓ |
| [aliengo](./robot_model/aliengo) | 2026-05-13 | 2026-05-13 | ✓ |
| [go2](./robot_model/go2) | 2026-08-04 | 2026-05-13 | ✓ |
| [hyqreal2](./robot_model/hyqreal2) | 2026-04-09 | 2026-03-24 | X |
| [pegasus](./robot_model/pegasus) | 2026-09-18 | 2026-09-20 |  |
| [piper_l](./robot_model/piper_l) | 2026-09-20 | 2026-09-20 | ✓ |
| [z1](./robot_model/z1) | 2026-09-19 | 2026-09-20 | ✓ |
<!-- model-dates:end -->

## Installation
1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform) or [pixi](https://pixi.prefix.dev/latest/installation/) - the latter for full reproducibility.

2. create an environment using the file in the folder [deploy/installation](./deploy/installation):

```bash
# for conda
conda env create -f mamba_environment.yaml
conda activate sim2real_robot_identification_env

# for pixi
pixi shell --manifest-path deploy/installation/pixi.toml 
```

4. install IsaacLab if you need it

## Run a collection
This repo works best with [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for communicating with unitree go2, b2, a2, and z1 robots, and with [piper-ros2-dls2](https://github.com/iit-DLSLab/piper-ros2-dls2) for the agilex piper arm.

1. Choose the robot and the gains in the  [config file](config.py) or add yours in the [robot_model](./robot_model) folder.

2. In the xml of your robot, add two keyframe (sys_id_1, sys_id_2) to define the start and end point of the chirp trajectory (see [here](./robot_model/go2/go2.xml) for an example)

3. Runs one of the following files
```bash
python3 run_collection_quadruped_ros2.py
python3 run_collection_manipulator_ros2.py
```


4. Inside the script
```bash
startCollection
trajectory
```

5. Close the script
```bash
ctrl+c
```

6. Visualize your runned trajectory
```bash
python3 datasets/replay_dataset_quadruped.py
python3 datasets/replay_dataset_manipulator.py
python3 datasets/plot_joint_trajectories.py
```

## Run a calibration in IsaacLab

Add a new [task](./sysid_isaaclab/tasks) for a new robot. Then

```bash
python3 sysid_isaaclab/my_fit.py --headless
```

## Run a calibration in Mujoco

```bash
python3 sysid_mujoco/my_fit.py 
```

## How to contribute

PRs are very welcome (search for **TODO** in the issue, or add what you like)!


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi) and [Lorenzo Amatucci](https://github.com/lorenzo96-cmd).
