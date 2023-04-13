import setup_path
import gym
import airgym
import time
import torch
import torch.cuda
import tensorflow as tf

from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass 


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    
force_cudnn_initialization()

# Create a DummyVecEnv for main airsim gym env
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

#check_env(env)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00001,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=100,
    buffer_size=500000,
    max_grad_norm=10,
    gamma=0.95,
    tensorboard_log="./tb_logs/",
)

# model = QRDQN(
#     "CnnPolicy",
#     env,
#     learning_rate=0.005,
#     verbose=1,
#     tensorboard_log="./tb_logs/",
# )

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
    deterministic=True
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=150000,
    tb_log_name="dqn_airsim_leader_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_leader_policy")
