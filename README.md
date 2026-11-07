  <div style="display: flex; justify-content: space-around;">
    <img src="https://img.shields.io/badge/IsaacLab%20-v3.0.0-green" alt="IsaacLab v3.0.0" style="margin-bottom: 1px;">
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
| [a2](./robot_model/a2) | 2026-11-07 | 2026-09-30 | ✓ |
| [aliengo](./robot_model/aliengo) | 2026-11-07 | 2026-09-30 | ✓ |
| [b2](./robot_model/b2) | 2026-11-07 | 2026-09-30 |  |
| [go2](./robot_model/go2) | 2026-11-07 | 2026-09-30 | ✓ |
| [hyqreal2](./robot_model/hyqreal2) | 2026-11-07 | 2026-09-30 | X |
| [pegasus](./robot_model/pegasus) | 2026-11-07 | 2026-11-04 |  |
| [piper_l](./robot_model/piper_l) | 2026-11-07 | 2026-09-30 | ✓ |
| [z1](./robot_model/z1) | 2026-11-07 | 2026-09-30 | ✓ |
<!-- model-dates:end -->

## Installation
1. Install [pixi](https://pixi.prefix.dev/latest/installation/) and clone this repository with its submodule (PACE):

```bash
git clone --recursive https://github.com/iit-DLSLab/sim2real-robot-identification.git
# or, in an existing clone
git submodule update --init
```

2. Activate the environment you need from anywhere inside the repo (it uses [pixi.toml](./pixi.toml) in the repo root). The first run creates it:

```bash
# data collection with ROS2 lyrical, and the calibration in Mujoco (default environment)
pixi shell   # same as: pixi shell -e lyrical

# data collection with ROS2 humble, and the calibration in Mujoco
pixi shell -e humble

# calibration in IsaacLab (Isaac Lab 3.0 from PyPI, no Isaac Sim, and PACE from the submodule)
pixi shell -e isaaclab
```

The `isaaclab` environment is separate because Isaac Lab 3.0 requires MuJoCo 3.12, while the calibration in Mujoco uses MuJoCo 3.13. It requires Ubuntu 22.04 or newer (glibc >= 2.35).

## Run a collection
This repo works best with [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for communicating with unitree go2, b2, a2, aliengo and z1 robots, and with [piper-ros2-dls2](https://github.com/iit-DLSLab/piper-ros2-dls2) for the agilex piper arm.

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
chirp        # or: excitation
```
`chirp` moves all the joints together back and forth between the two keyframes. `excitation` ([excitation_trajectory.py](excitation_trajectory.py)) stays in the same joint box, but gives every joint its own chirp plus a slow posture sweep, which makes the parameters easier to identify; it asks for a seed (use different seeds for the fit and the validation trajectories) and a speed scale in (0, 1] (default 0.5, the same motion slowed down; increase it once the motion looks safe), prints the joint ranges and peak velocities, and waits for confirmation. It is saved as `excitation_N.pt`.

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

Add a new [task](./sysid_isaaclab/tasks) for a new robot. Then, in `pixi shell -e isaaclab`

```bash
python3 sysid_isaaclab/my_fit.py                   # CMA-ES (default): one best combination
python3 sysid_isaaclab/my_fit.py --optimizer ot    # a set of plausible combinations
```

With `--optimizer ot`, a diversity-regularized version of the optimal-transport optimizer of [arXiv:2605.02147](https://arxiv.org/abs/2605.02147) ([ot_optimizer.py](./sysid_isaaclab/ot_optimizer.py)) returns `num_particles` parameter combinations $\Theta$ solving

$$\min_{\Theta}\ \frac{1}{N}\sum_i \frac{L(\theta_i)}{L_{best}} - \lambda\, D(\Theta) \quad \text{s.t.} \quad L(\theta_i) \le (1 + \text{loss\_tolerance})\, L_{best}$$

where $D$ is a log-det (DPP) diversity. They are saved in `plausible_params.pt` in the log folder, and the per-joint range of every parameter over the set is printed at the end (a wide range means that parameter is not identifiable from the data). The defaults are in `OTSinkhornOptimizerCfg`; to change them for a robot, add a field to its `PaceCfg`, e.g. `ot: OTSinkhornOptimizerCfg = OTSinkhornOptimizerCfg(num_particles=64, loss_tolerance=0.05)`.

With both optimizers, the joint parameters are searched linearly within their bounds, as in PACE, and:
- `--identify-link-mass`, `--identify-center-of-mass` (mass + CoM) and `--identify-inertia-tensor` (mass + CoM + inertia) also identify the inertial parameters of the bodies, with the same options of the MuJoCo fit (`--inertial-bodies`, `--tie-quadruped-inertias`, bounds). They are corrections of the nominal USD values: mass scale, CoM offset in the body frame (mirrored between legs when tied) and scales of the principal second moments of the mass distribution, so every value within the bounds is a physically consistent body. The identified values are saved in `body_params.pt` (CMA-ES) or under `inertial` in `plausible_params.pt` (OT). On Newton, Isaac Lab warns that runtime CoM changes may be unstable: prefer `physics=physx` with `--identify-center-of-mass`.

## Run a calibration in Mujoco

In `pixi shell -e humble` (or `lyrical`):

```bash
python3 sysid_mujoco/my_fit.py 
```

## How to contribute

PRs are very welcome (search for **TODO** in the issue, or add what you like)!


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi) and [Lorenzo Amatucci](https://github.com/lorenzo96-cmd).
