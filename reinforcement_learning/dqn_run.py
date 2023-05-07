import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import DQN, SAC
from sb3_contrib import QRDQN 
from stable_baselines3.common.evaluation import evaluate_policy

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-leader-v3",
                ip_address="127.0.0.1",
                step_length=1.0,
                image_shape=(84,84,1),
            )
        )
    ]
)

env = VecTransposeImage(env)

model = SAC.load("bestmodel_sacmpc1681791552.26457/best_model")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic = True)
    obs, rewards, dones, info = env.step(action)