import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import SAC
from sb3_contrib import QRDQN 
from stable_baselines3.common.evaluation import evaluate_policy

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-leader-v3",
                ip_address="127.0.0.1",
                step_length=0.5,
                image_shape=(84,84,1),
            )
        )
    ]
)

model = SAC.load("policiesandmodels/dqn_airsim_leader_policy_new_2_sac1obs")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic = True)
    obs, rewards, dones, info = env.step(action)