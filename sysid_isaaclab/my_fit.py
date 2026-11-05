# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

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
parser.add_argument(
    "--optimizer",
    type=str,
    default="cmaes",
    choices=["cmaes", "ot"],
    help="cmaes: single best combination (PACE). ot: set of plausible combinations (diversity-regularized optimal transport).",
)
parser.add_argument(
    "--dataset",
    nargs="+",
    default=None,
    metavar="PATH",
    help=(
        "Datasets (.pt files or directories of them) fitted together, each replayed by its own group of "
        "environments. By default, the data_dir of the task."
    ),
)
# optional inertial params, with the same options of sysid_mujoco/my_fit.py
parser.add_argument("--identify-link-mass", action="store_true", help="Also identify the mass of the selected bodies.")
parser.add_argument(
    "--identify-center-of-mass", action="store_true", help="Identify mass and center of mass of the selected bodies."
)
parser.add_argument(
    "--identify-inertia-tensor",
    action="store_true",
    help="Identify mass, center of mass and physically consistent inertia of the selected bodies.",
)
parser.add_argument(
    "--inertial-bodies",
    nargs="+",
    default=None,
    metavar="BODY",
    help="Bodies (regex) of the inertial params. By default, all bodies with mass except the fixed root.",
)
parser.add_argument(
    "--tie-quadruped-inertias",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Share the inertial params of corresponding FL, FR, RL and RR bodies, mirroring the CoM offsets (default: enabled).",
)
parser.add_argument(
    "--link-mass-scale-bounds",
    nargs=2,
    type=float,
    default=(0.9, 1.1),
    metavar=("LOWER", "UPPER"),
    help="Mass bounds as multipliers of the nominal mass (log scale).",
)
parser.add_argument(
    "--center-of-mass-offset-bounds",
    nargs=2,
    type=float,
    default=(-0.03, 0.03),
    metavar=("LOWER", "UPPER"),
    help="Additive bounds in metres around every nominal CoM component, in the body frame.",
)
parser.add_argument(
    "--inertia-tensor-scale-bounds",
    nargs=2,
    type=float,
    default=(0.9, 1.1),
    metavar=("LOWER", "UPPER"),
    help="Bounds of the multipliers of the principal second moments of the mass distribution (log scale).",
)
# append launcher cli args
add_launcher_args(parser)
# parse the arguments; leftover Hydra-style tokens (e.g. `physics=physx`) select presets
args_cli, unknown_args = parser.parse_known_args()
# presets written after a list option are parsed as items of the list: move them back
for list_option in ("inertial_bodies", "dataset"):
    values = getattr(args_cli, list_option)
    if values:
        unknown_args += [value for value in values if "=" in value]
        setattr(args_cli, list_option, [value for value in values if "=" not in value])
preset_overrides = [arg for arg in unknown_args if not arg.startswith("-")]
if len(preset_overrides) != len(unknown_args):
    parser.error(f"unrecognized arguments: {' '.join(arg for arg in unknown_args if arg.startswith('-'))}")

"""Rest everything follows."""

import cmaes
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
from ot_optimizer import OTSinkhornOptimizer, OTSinkhornOptimizerCfg
from inertial_params import BodyParams


def params_to_sim_params(params, bounds, log_scale=None):
    """Map normalized params in [-1, 1] to the bounds, log-uniformly where ``log_scale`` is set."""
    unit = (params + 1.0) / 2.0  # change range from 0 to 1
    sim_params = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])  # range from lower to upper bound
    if log_scale is None or not bool(log_scale.any()):
        return sim_params
    log_low = bounds[:, 0].clamp_min(torch.finfo(bounds.dtype).tiny).log()
    log_high = bounds[:, 1].clamp_min(torch.finfo(bounds.dtype).tiny).log()
    return torch.where(log_scale, torch.exp(log_low + unit * (log_high - log_low)), sim_params)


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


def expand_data_files(entries, base_dir):
    """Dataset files from paths relative to base_dir: a .pt file, or a directory whose .pt files are all used."""
    if isinstance(entries, (str, Path)):
        entries = [entries]
    data_files = []
    for entry in entries:
        path = base_dir / entry  # an absolute entry stays absolute
        data_files += sorted(path.glob("*.pt")) if path.is_dir() else [path]
    missing = [str(path) for path in data_files if not path.is_file()]
    if missing or not data_files:
        raise FileNotFoundError(f"Datasets not found: {missing or entries}")
    return data_files


def load_trajectories(data_files, device):
    """Stack the recorded trajectories, padded to the longest one by holding their last sample.

    Returns the first dataset (logged by the optimizers), the target and measured joint positions (K, T, J) and
    the mask of the recorded steps (K, T), which excludes the padding from the score.
    """
    # Dataset files are trusted and may contain non-tensor Python objects.
    datasets = [load_torch_data_compat(data_file) for data_file in data_files]
    # the actuator gains are set by the task, so they must be the same in all recordings
    for gain in ("kp", "kd"):
        values = [torch.as_tensor(data[gain]) for data in datasets if gain in data]
        if any(not torch.equal(value, values[0]) for value in values[1:]):
            raise ValueError(f"The datasets were recorded with different `{gain}`, but they share the actuator gains.")
    num_steps = max(data["time"].shape[0] for data in datasets)

    def padded(key):
        trajectories = []
        for data in datasets:
            values = torch.as_tensor(data[key], dtype=torch.float32)
            trajectories.append(torch.cat([values, values[-1:].expand(num_steps - values.shape[0], -1)]))
        return torch.stack(trajectories).to(device)

    recorded_steps = torch.stack([torch.arange(num_steps) < data["time"].shape[0] for data in datasets]).to(device)
    return datasets[0], padded("des_dof_pos"), padded("dof_pos"), recorded_steps


def update_run_config(log_dir, **entries):
    """Add entries to the config.pt saved by the optimizer in its log directory."""
    config_path = os.path.join(log_dir, "config.pt")
    run_config = torch.load(config_path, weights_only=False)
    run_config.update(entries)
    torch.save(run_config, config_path)


class BackendAgnosticSimulatorMixin:
    """Writes the optimizer params through APIs shared by PhysX and Newton.

    The joint params are mapped linearly to the bounds as in PACE. They may be followed by the inertial params of
    some bodies, whose scale factors are mapped in log scale.

    Every candidate replays all the trajectories in parallel: env k * num_candidates + i replays trajectory k with
    the params of candidate i.
    """

    def __init__(self, *args, recorded_steps, log_scale=None, body_params=None, **kwargs):
        # set before the optimizer init, which samples the first population through _params_to_sim_params
        self.log_scale = log_scale
        self.body_params = body_params
        self.recorded_steps = recorded_steps
        self.num_trajectories = recorded_steps.shape[0]
        # optimizers divide the summed score by the number of steps of the longest trajectory: rescale it to the
        # mean over the recorded steps of all trajectories, as PACE's mean over a single trajectory
        self.score_scale = recorded_steps.shape[1] / recorded_steps.sum().item()
        super().__init__(*args, **kwargs)

    def tell(self, sim_dof_pos, real_dof_pos):
        num_candidates = self.sim_params.shape[0]
        recorded = self.recorded_steps[:, self.scores_counter].repeat_interleave(num_candidates)
        bias = self.sim_params[:, self.bias_idx].repeat(self.num_trajectories, 1)
        error = torch.sum(torch.square(sim_dof_pos - real_dof_pos - bias), dim=1) * recorded
        self.scores += error.view(self.num_trajectories, num_candidates).sum(dim=0) * self.score_scale
        # the best trajectory saved by the optimizers is the one of the first dataset
        if self.scores_counter < self.sim_dof_pos_buffer.shape[1]:
            self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos[:num_candidates]
        self.scores_counter += 1

    def _params_to_sim_params(self, params):
        return params_to_sim_params(params, self.bounds, self.log_scale)

    def _print_body_params(self, sim_params):
        if self.body_params is None or sim_params.shape[0] == 0:
            return
        print("Inertial params:")
        for line in self.body_params.describe(sim_params[:, self.body_params.slice]):
            print(line)

    def update_simulator(self, articulation, joint_ids, initial_position):
        # the params of every candidate, once per trajectory group
        sim_params = self.sim_params.repeat(self.num_trajectories, 1)
        if self.body_params is not None:
            self.body_params.write(articulation, sim_params[:, self.body_params.slice])
        env_ids = torch.arange(len(sim_params[:, self.armature_idx]), device=joint_ids.device)
        armature = sim_params[:, self.armature_idx]
        viscous_friction = sim_params[:, self.damping_idx]
        friction = sim_params[:, self.friction_idx]
        articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
        # Newton exposes a single dry-friction value, while PhysX backends also provide a separate
        # dynamic-friction coefficient (equal to the static one unless identified separately).
        if hasattr(articulation, "write_joint_dynamic_friction_coefficient_to_sim_index"):
            # static, dynamic and viscous are written to PhysX in a single call, so no ordering issue
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=friction,
                joint_dynamic_friction_coeff=self.dynamic_friction().repeat(self.num_trajectories, 1),
                joint_viscous_friction_coeff=viscous_friction,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            if self.separate_dynamic_friction:
                raise ValueError("Newton has a single dry-friction value: set separate_dynamic_friction to False.")
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=friction,
                joint_viscous_friction_coeff=viscous_friction,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        articulation.write_joint_position_to_sim_index(
            position=initial_position + sim_params[:, self.bias_idx], joint_ids=joint_ids
        )
        articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial_position), joint_ids=joint_ids)
        for drive_type in articulation.actuators.keys():
            drive_indices = articulation.actuators[drive_type].joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            articulation.actuators[drive_type].update_encoder_bias(sim_params[:, self.bias_idx][:, drive_joint_idx])
            articulation.actuators[drive_type].update_time_lags(sim_params[:, self.delay_idx].to(torch.int))
            articulation.actuators[drive_type].reset(env_ids)


class BackendAgnosticCMAESOptimizer(BackendAgnosticSimulatorMixin, CMAESOptimizer):
    """CMA-ES optimizer that writes joint params through APIs shared by PhysX and Newton."""

    def __init__(self, bounds, *args, log_scale=None, body_params=None, sigma=0.5, **kwargs):
        # PACE >= 0.2 accepts only its own params: build it on those, then append the inertial params
        num_pace_params = bounds.shape[0] if body_params is None else body_params.slice.start
        pace_log_scale = None if log_scale is None else log_scale[:num_pace_params]
        super().__init__(
            bounds[:num_pace_params], *args, log_scale=pace_log_scale, body_params=body_params, sigma=sigma, **kwargs
        )
        if num_pace_params < bounds.shape[0]:
            self._append_params(bounds, log_scale, sigma)

    def _append_params(self, bounds, log_scale, sigma):
        """Rebuild the CMA-ES state and the param buffers on all params, as CMAESOptimizer.__init__ does."""
        population_size = self.optimizer.population_size
        self.bounds = bounds
        self.log_scale = log_scale
        bounds_normalized = torch.ones_like(bounds)
        bounds_normalized[:, 0] *= -1
        mean_normalized = torch.zeros_like(bounds[:, 0])
        self.optimizer = cmaes.CMA(mean=mean_normalized.cpu().numpy(), sigma=sigma, bounds=bounds_normalized.cpu().numpy(), seed=0, population_size=population_size)
        self.params = torch.zeros((population_size, bounds.shape[0]), device=self.device)
        self.sim_params = torch.zeros_like(self.params)
        if self.save_optimization_process:
            self.sim_params_buffer = torch.zeros((self.max_iteration, population_size, bounds.shape[0]), device=self.device)
        # config.pt was saved with the PACE bounds only
        update_run_config(self.writer.log_dir, bounds=bounds)
        self._reset_population()

    def _print_iteration(self):
        super()._print_iteration()
        self._print_body_params(self.sim_params[torch.argmin(self.scores)].unsqueeze(0))

    def save_checkpoint(self, mean, iteration, finished=False):
        super().save_checkpoint(mean, iteration, finished)
        if self.body_params is not None:
            torch.save(self.body_params.to_dict(mean[None, self.body_params.slice]), os.path.join(self.writer.log_dir, "body_params.pt"))


class BackendAgnosticOTOptimizer(BackendAgnosticSimulatorMixin, OTSinkhornOptimizer):
    """Diversity-regularized OT optimizer that writes joint params through APIs shared by PhysX and Newton."""

    def _plausible_dict(self):
        plausible = super()._plausible_dict()
        if self.body_params is not None:
            plausible["inertial"] = self.body_params.to_dict(plausible["sim_params"][:, self.body_params.slice])
        return plausible

    def _print_plausible_set(self):
        super()._print_plausible_set()
        self._print_body_params(self.get_plausible_sim_params())


def main(env_cfg):
    """Zero actions agent with Isaac Lab environment."""

    # every candidate replays all the trajectories, each in its own group of envs
    if args_cli.dataset:
        data_files = expand_data_files(args_cli.dataset, Path.cwd())
    else:
        data_files = expand_data_files(env_cfg.sim2real.data_dir, project_root() / "data")
    data, target_dof_pos, measured_dof_pos, recorded_steps = load_trajectories(data_files, env_cfg.sim.device)
    num_trajectories = len(data_files)
    num_candidates = args_cli.num_envs // num_trajectories
    if num_candidates < 2:
        raise ValueError(f"{args_cli.num_envs} envs are too few for {num_trajectories} trajectories.")
    env_cfg.scene.num_envs = num_candidates * num_trajectories
    print(f"[INFO]: {num_trajectories} trajectories x {num_candidates} candidates = {env_cfg.scene.num_envs} envs:")
    for data_file, num_steps in zip(data_files, recorded_steps.sum(dim=1).tolist()):
        print(f"  - {data_file} ({num_steps} steps)")

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
    log_scale = None

    body_params = None
    if args_cli.identify_link_mass or args_cli.identify_center_of_mass or args_cli.identify_inertia_tensor:
        body_params = BodyParams(
            articulation,
            offset=bounds_params.shape[0],
            is_newton="newton" in env.unwrapped.sim.physics_manager.__name__.lower(),
            body_names=args_cli.inertial_bodies,
            identify_mass=args_cli.identify_link_mass,
            identify_com=args_cli.identify_center_of_mass,
            identify_inertia=args_cli.identify_inertia_tensor,
            tie_quadruped=args_cli.tie_quadruped_inertias,
            mass_scale_bounds=tuple(args_cli.link_mass_scale_bounds),
            com_offset_bounds=tuple(args_cli.center_of_mass_offset_bounds),
            inertia_scale_bounds=tuple(args_cli.inertia_tensor_scale_bounds),
        )
        print("[INFO]: Inertial params:", body_params.names)
        joint_log_scale = torch.zeros(bounds_params.shape[0], dtype=torch.bool)
        log_scale = torch.cat([joint_log_scale, body_params.log_scale]).to(bounds_params.device)
        bounds_params = torch.cat([bounds_params, body_params.bounds.to(bounds_params.device)])

    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    initial_dof_pos = measured_dof_pos[:, 0].repeat_interleave(num_candidates, dim=0)

    time_steps = target_dof_pos.shape[1]
    sim_dt = env.unwrapped.sim.cfg.dt

    if args_cli.optimizer == "cmaes":
        opt = BackendAgnosticCMAESOptimizer(
            bounds=bounds_params,
            population_size=num_candidates,
            log_dir=log_dir,
            joint_order=joint_order,
            max_iteration=env_cfg.sim2real.cmaes.max_iteration,
            data=data,
            device=env.unwrapped.device,
            epsilon=env_cfg.sim2real.cmaes.epsilon,
            sigma=env_cfg.sim2real.cmaes.sigma,
            save_interval=env_cfg.sim2real.cmaes.save_interval,
            save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
            separate_dynamic_friction=env_cfg.sim2real.separate_dynamic_friction,
            recorded_steps=recorded_steps,
            log_scale=log_scale,
            body_params=body_params,
        )
    else:
        # a task can tune the optimizer by adding an `ot: OTSinkhornOptimizerCfg` field to its PaceCfg
        opt = BackendAgnosticOTOptimizer(
            bounds=bounds_params,
            batch_size=num_candidates,
            log_dir=log_dir,
            joint_order=joint_order,
            data=data,
            device=env.unwrapped.device,
            cfg=getattr(env_cfg.sim2real, "ot", None) or OTSinkhornOptimizerCfg(),
            separate_dynamic_friction=env_cfg.sim2real.separate_dynamic_friction,
            recorded_steps=recorded_steps,
            log_scale=log_scale,
            body_params=body_params,
        )
    # the optimizers log the first dataset only
    update_run_config(opt.writer.log_dir, data_files=[str(data_file) for data_file in data_files])

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # each group of envs tracks its own trajectory
            opt.tell(articulation.data.joint_pos.torch[:, sim_joint_ids], measured_dof_pos[:, counter].repeat_interleave(num_candidates, dim=0))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[:, counter].repeat_interleave(num_candidates, dim=0)
            # apply actions
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_steps * sim_dt:.1f} seconds ({counter / time_steps * 100:.1f} %)")
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
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=preset_overrides
    )
    # start only the runtime the task needs (Kit is launched only for Isaac Sim PhysX / Kit visualizers)
    with launch_simulation(env_cfg, args_cli):
        main(env_cfg)
