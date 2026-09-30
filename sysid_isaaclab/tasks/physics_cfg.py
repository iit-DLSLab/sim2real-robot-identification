from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.physics import PhysxAutoCfg


# Same physics presets as basic-locomotion-isaaclab, so the identified parameters
# match the backend used for training. Select one with e.g. `physics=physx`.
@configclass
class PacePhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=2**23)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=2**23)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp
