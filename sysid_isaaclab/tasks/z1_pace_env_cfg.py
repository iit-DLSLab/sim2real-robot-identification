# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

import os
import toml
ISAAC_ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

Z1_13456_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".joint1", ".*joint3", ".*joint4", ".*joint5", ".*joint6"],
    saturation_effort=30.,
    effort_limit=30.0,
    velocity_limit=3.14,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)

Z1_2_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".joint2"],
    saturation_effort=30.,
    effort_limit=60.0,
    velocity_limit=3.14,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)

Z1_GRIPPER_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".jointGripper"],
    saturation_effort=5.,
    effort_limit=5.0,
    velocity_limit=3.14,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)

Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/../../robot_model/z1/z1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".joint*": 0.,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={"joint1": Z1_13456_ACTUATOR_CFG,
               "joint2": Z1_2_ACTUATOR_CFG,
               "joint3": Z1_13456_ACTUATOR_CFG,
               "joint4": Z1_13456_ACTUATOR_CFG,
               "joint5": Z1_13456_ACTUATOR_CFG,
               "joint6": Z1_13456_ACTUATOR_CFG,
               "jointGripper": Z1_GRIPPER_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)



@configclass
class Z1PaceCfg(PaceCfg):
    """Pace configuration for z1 robot."""
    robot_name: str = "z1_sim"
    data_dir: str = f"{ISAAC_ASSET_DIR}/../../datasets/z1/traj_0.pt"  # located in pace_sim2real/data/z1_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((35, 2))  # 7 + 7 + 7 + 7 + 1 = 35 parameters to optimize
    joint_order: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "jointGripper",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:7, 0] = 1e-5
        self.bounds_params[:7, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[7:14, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[14:21, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[21:28, 0] = -0.1
        self.bounds_params[21:28, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[28, 1] = 5.0  # delay between 0.0 - 5.0 [sim steps]


@configclass
class Z1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Z1 robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class Z1PaceEnvCfg(PaceSim2realEnvCfg):

    scene: Z1PaceSceneCfg = Z1PaceSceneCfg()
    sim2real: PaceCfg = Z1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.005  # 200Hz simulation
        self.decimation = 1  # 200Hz control
