## Overwiew

A joint calibration routine for IsaacLab and Mujoco, to estimate leg motors parameters. It provides scripts for data collection on the real robot (the robot should be in air with the base fixed). 
This repo interfaces directly with [Pace](https://github.com/leggedrobotics/pace-sim2real) and the new sys-id functionality of Mujoco.


Work in progress for supporting arms and mujoco identification, PRs are very welcome!


## Run a collection
This repo works the best with [unitree_ros2_dls](https://github.com/iit-DLSLab/unitree_ros2_dls) for communicating with unitree go2 and b2 robots. Soon, will support different arms, such as unitree z1, and agilex piper arms using - the last using [piper_ros2_dls2](https://github.com/iit-DLSLab/piper_ros2_dls2).

```bash
python3 run_collection_quadruped_ros2.py
```

## Run a calibration in IsaacLab

```bash
python3 sysid_isaaclab/my_fit.py --headless
```


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).
