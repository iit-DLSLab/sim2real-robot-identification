# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import os
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from .physics_cfg import PacePhysicsCfg
from .pace_actuator import PaceDCMotorCfg


ISAAC_ASSET_DIR = os.path.abspath(os.path.dirname(__file__))
TASKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASKS_DIR.parents[1]


PIPER_ARM_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["joint[1-3]"],
    saturation_effort=32.0,
    effort_limit=32.0,
    velocity_limit=5.0,
    stiffness={
        "joint1": 150.0,
        "joint2": 150.0,
        "joint3": 150.0,
    },
    damping={
        "joint1": 10.0,
        "joint2": 10.0,
        "joint3": 10.0,
    },
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=2,
)

PIPER_WRIST_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["joint[4-6]"],
    saturation_effort=8.0,
    effort_limit=8.0,
    velocity_limit=5.0,
    stiffness={
        "joint4": 10.0,
        "joint5": 10.0,
        "joint6": 10.0,
    },
    damping={
        "joint4": 0.8,
        "joint5": 0.8,
        "joint6": 0.8,
    },
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=2,
)

PIPER_GRIPPER_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["joint7"],
    #saturation_effort=0.5,
    #effort_limit=0.5,
    #velocity_limit=0.5,
    saturation_effort=2.0,
    effort_limit=2.0,
    velocity_limit=2.0,
    stiffness={"joint7": 10},
    damping={"joint7": 0.8},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=2,
)


PIPER_L_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/../../robot_model/piper_l/generated_usd/piper_l.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": PIPER_ARM_ACTUATOR_CFG,
        "wrist": PIPER_WRIST_ACTUATOR_CFG,
        "gripper": PIPER_GRIPPER_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)


@configclass
class PiperLPaceCfg(PaceCfg):
    """PACE configuration for the AgileX Piper robot."""

    robot_name: str = "piper_l_sim"
    data_dir: str = str(REPO_ROOT / "datasets" / "piper_l" / "traj_0.pt")
    bounds_params: torch.Tensor = torch.zeros((29, 2))
    joint_order: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    def __post_init__(self):
        num_joints = len(self.joint_order)

        # Armature [kg m²].
        self.bounds_params[:num_joints, 0] = 1e-5
        self.bounds_params[:num_joints, 1] = 1.0

        # Viscous friction [N m s/rad] (or [N s/m] for the gripper).
        self.bounds_params[num_joints : 2 * num_joints, 1] = 1.0

        # Static/dynamic friction [N m] (or [N] for the gripper).
        self.bounds_params[2 * num_joints : 3 * num_joints, 1] = 1.0

        # Encoder bias: radians for joint1-joint6, metres for joint7.
        self.bounds_params[3 * num_joints : 4 * num_joints, 0] = -0.0
        self.bounds_params[3 * num_joints : 4 * num_joints, 1] = 0.0
        self.bounds_params[4 * num_joints - 1, 0] = -0.0
        self.bounds_params[4 * num_joints - 1, 1] = 0.0

        # Global actuation delay [simulation steps].
        self.bounds_params[4 * num_joints, 1] = 2.0


@configclass
class PiperLPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene configuration for Piper in the PACE environment."""

    robot: ArticulationCfg = PIPER_L_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


@configclass
class PiperLPaceEnvCfg(PaceSim2realEnvCfg):
    """Isaac Lab PACE environment configuration for Piper."""

    scene: PiperLPaceSceneCfg = PiperLPaceSceneCfg()
    sim2real: PaceCfg = PiperLPaceCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = 0.005  # 200 Hz simulation.
        self.sim.physics = PacePhysicsCfg()  # physics backend presets, same as basic-locomotion-isaaclab
        self.sim.use_newton_actuators = False  # PaceDCMotorCfg is not a Newton-native actuator
        self.decimation = 1  # 200 Hz control.
