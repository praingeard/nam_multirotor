from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-drone-multi", entry_point="airgym.envs:AirSimDroneMultiEnv",
)

register(
    id="airsim-drone-leader", entry_point="airgym.envs:AirSimLeaderEnv",
)