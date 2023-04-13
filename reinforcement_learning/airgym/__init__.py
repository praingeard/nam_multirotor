from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
     id="airsim-drone-multi-v0", entry_point="airgym.envs:AirSimDroneMultiEnv",
)

register(
    id="airsim-drone-leader-v0", entry_point="airgym.envs:AirSimLeaderEnv",
)

register(
    id="airsim-drone-leader-v1", entry_point="airgym.envs:AirSimLeaderGoalEnv",
)

register(
    id="airsim-drone-leader-v2", entry_point="airgym.envs:AirSimLeaderGoalHerEnv",
)

register(
    id="airsim-drone-leader-v3", entry_point="airgym.envs:AirSimLeader2DEnv",
)

register(
    id="airsim-drone-leader-v4", entry_point="airgym.envs:AirSimLeader2DHerEnv",
)