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

ALIENGO_HIP_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint"],
    saturation_effort=35.278,
    effort_limit=35.278,
    velocity_limit=20.0,
    stiffness={".*": 25.0},  # P gain in Nm/rad
    damping={".*": 0.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)


ALIENGO_THIGH_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_thigh_joint"],
    saturation_effort=35.278,
    effort_limit=35.278,
    velocity_limit=20.0,
    stiffness={".*": 25.0},  # P gain in Nm/rad
    damping={".*": 0.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)

ALIENGO_CALF_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    saturation_effort=44.4,
    effort_limit=44.4,
    velocity_limit=15.89,
    stiffness={".*": 25.0},  # P gain in Nm/rad
    damping={".*": 0.5},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 0.0},  # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=5,  # max delay in simulation steps
)

ALIENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/../../robot_model/aliengo/generated_usd/aliengo.usda",
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
            ".*L_hip_joint": 0.,
            ".*R_hip_joint": 0.,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={"hip": ALIENGO_HIP_ACTUATOR_CFG, "thigh": ALIENGO_THIGH_ACTUATOR_CFG, "calf": ALIENGO_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)



@configclass
class AliengoPaceCfg(PaceCfg):
    """Pace configuration for Aliengo robot."""
    robot_name: str = "aliengo_sim"
    data_dir: str = f"{ISAAC_ASSET_DIR}/../../datasets/aliengo/traj_0.pt"  # located in pace_sim2real/data/aliengo_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 1.0  # friction between 0.0 - 1.0
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 5.0  # delay between 0.0 - 5.0 [sim steps]


@configclass
class AliengoPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Aliengo robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class AliengoPaceEnvCfg(PaceSim2realEnvCfg):

    scene: AliengoPaceSceneCfg = AliengoPaceSceneCfg()
    sim2real: PaceCfg = AliengoPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.005  # 200Hz simulation
        self.decimation = 1  # 200Hz control
