import gym
import sys

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import DQN, SAC
from sb3_contrib import QRDQN 
from stable_baselines3.common.evaluation import evaluate_policy

if len(sys.argv) > 1:
    dname = sys.argv[1]
    print("Argument received:", dname)
else:
    print("No argument provided.")
    dname = "Leader"

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-leader-v3",
                drone_name = dname,
                ip_address="127.0.0.1",
                step_length=1.0,
                image_shape=(84,84,1),
            )
        )
    ]
)

env = VecTransposeImage(env)

if dname == "Drone1":
    model = SAC.load("bestmodel_multisac1686190248.110544/best_model")
elif dname == "Drone2":
    model = SAC.load("bestmodel_multisac1686190248.140853/best_model")
elif dname == "Drone3":
    model = SAC.load("bestmodel_multisac1686190248.1538694/best_model")
elif dname == "Leader":
    model = SAC.load("bestmodel_multisac1686190248.0790057/best_model")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic = False)
    obs, rewards, dones, info = env.step(action)