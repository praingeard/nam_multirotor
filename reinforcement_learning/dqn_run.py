import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-leader-v0",
                ip_address="127.0.0.1",
                step_length=0.5,
                image_shape=(36864,),
            )
        )
    ]
)

model = DQN.load("best_model")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)