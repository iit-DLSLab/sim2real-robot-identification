# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import config
task_name = "IsaacLab-Pace-" + config.robot

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=task_name, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments; leftover Hydra-style tokens (e.g. `physics=physx`) select presets
args_cli, unknown_args = parser.parse_known_args()
preset_overrides = [arg for arg in unknown_args if not arg.startswith("-")]
if len(preset_overrides) != len(unknown_args):
    parser.error(f"unrecognized arguments: {' '.join(arg for arg in unknown_args if arg.startswith('-'))}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from tasks import register_my_tasks

from pace_sim2real.utils import project_root
from pace_sim2real import CMAESOptimizer


def load_torch_data_compat(data_file):
    """Load torch data across PyTorch/NumPy compatibility differences."""
    try:
        return torch.load(data_file, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(data_file, map_location="cpu")
    except ModuleNotFoundError as exc:
        # Some pickles reference numpy._core (NumPy 2.x internal path).
        if exc.name == "numpy._core":
            sys.modules["numpy._core"] = np.core
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
            try:
                return torch.load(data_file, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(data_file, map_location="cpu")
        raise


class BackendAgnosticCMAESOptimizer(CMAESOptimizer):
    """CMA-ES optimizer that writes joint params through APIs shared by PhysX and Newton."""

    def update_simulator(self, articulation, joint_ids, initial_position):
        env_ids = torch.arange(len(self.sim_params[:, self.armature_idx]), device=joint_ids.device)
        armature = self.sim_params[:, self.armature_idx]
        viscous_friction = self.sim_params[:, self.damping_idx]
        friction = self.sim_params[:, self.friction_idx]
        articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
        # Newton exposes a single dry-friction value, while PhysX backends also
        # provide a separate dynamic-friction coefficient (kept equal to the static one).
        if hasattr(articulation, "write_joint_dynamic_friction_coefficient_to_sim_index"):
            # static, dynamic and viscous are written to PhysX in a single call, so no ordering issue
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=friction,
                joint_dynamic_friction_coeff=friction,
                joint_viscous_friction_coeff=viscous_friction,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=friction,
                joint_viscous_friction_coeff=viscous_friction,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        articulation.write_joint_position_to_sim_index(
            position=initial_position + self.sim_params[:, self.bias_idx], joint_ids=joint_ids
        )
        articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial_position), joint_ids=joint_ids)
        for drive_type in articulation.actuators.keys():
            drive_indices = articulation.actuators[drive_type].joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            articulation.actuators[drive_type].update_encoder_bias(self.sim_params[:, self.bias_idx][:, drive_joint_idx])
            articulation.actuators[drive_type].update_time_lags(self.sim_params[:, self.delay_idx].to(torch.int))
            articulation.actuators[drive_type].reset(env_ids)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=preset_overrides
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    # Dataset files are trusted and may contain non-tensor Python objects.
    data = load_torch_data_compat(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    opt = BackendAgnosticCMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
    )

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            opt.tell(env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
    # close optimizer
    opt.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
