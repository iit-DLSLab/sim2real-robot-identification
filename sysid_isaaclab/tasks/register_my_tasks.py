import gymnasium as gym

gym.register(
    id="IsaacLab-Pace-go2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.go2_pace_env_cfg:Go2PaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-a2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.aliengo_pace_env_cfg:AliengoPaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-pegasus",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.pegasus_pace_env_cfg:PegasusPaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-z1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.z1_pace_env_cfg:Z1PaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-piper_l",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.piper_l_pace_env_cfg:PiperLPaceEnvCfg",
    },
)
