# Overwiew

A simple joint calibration routine for IsaacLab, to estimate friction parameters and PD gains. It provides scripts for data collection on the real robot (the robot should be in air with the base fixed). Random sampling provides the best parameters fitting the saved trajectories.

Work in progress, PRs are very welcome!

## Run a collection
This repo works the best with [unitree_ros2_dls](https://github.com/iit-DLSLab/unitree_ros2_dls), for unitree go2 and b2 robots. Soon, will support different arms, such as unitree z1, and agilex piper arms [piper_ros2_dls2](https://github.com/iit-DLSLab/piper_ros2_dls2) 


## Run a calibration in IsaacLab

```bash
python3 calibrate_isaaclab.py --task=Locomotion-Aliengo-Flat --num_envs=8192  --headless
```


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).