import setup_path
import gym
import airgym
import time
import torch
import torch.cuda
import tensorflow as tf

from stable_baselines3 import DQN, HerReplayBuffer, SAC
#from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack

# try:
#     tf_gpus = tf.config.list_physical_devices('GPU')
#     for gpu in tf_gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# except:
#     pass 


# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    
# force_cudnn_initialization()

# Create a DummyVecEnv for main airsim gym env
# env =  gym.make(
#                 "airgym:airsim-drone-leader-v3",
#                 ip_address="127.0.0.1",
#                 step_length=1.0,
#                 image_shape=(84,84,1),
#             )


goal_selection_strategy = 'future'

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

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)



# Initialize RL algorithm type and parameters
model = SAC(
    "CnnPolicy",
    env,
    tensorboard_log="./tb_logs_test/",
    verbose=1,
)

# model = DQN(
#     "CnnPolicy",
#     env,
#     learning_rate=0.005,
#     verbose=1,
#     target_update_interval=500,
#     learning_starts=6000,
#     buffer_size=1000000,
#     tensorboard_log="./tb_logs_new/",
# )

# model = QRDQN(
#     "CnnPolicy",
#     env,
#     learning_rate=0.005,
#     verbose=1,
    # batch_size=32,
    # train_freq=4,
#     tensorboard_log="./tb_logs/",
# )



#Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs_model/" + str(time.time()),
  name_prefix="rl_model_mpc" + str(time.time()),
  save_replay_buffer=True,
  save_vecnormalize=True,
)

# Create an evaluation callback with the same env, called every 10000 iterations
eval_callback = EvalCallback(
    env,
    n_eval_episodes=10,
    best_model_save_path="bestmodel_sacmpc_contrchange" + str(time.time()),
    log_path="bestmodel_sacmpc_contrchange" + str(time.time()),
    eval_freq=500,
    deterministic=False
)
callbacks = CallbackList([eval_callback])

# Train for a certain number of timesteps
model.learn(
    total_timesteps=100000,
    tb_log_name="dqn_airsim_leader_run_new_mpc_contrchange" + str(time.time()),
    callback = callbacks,
    progress_bar=True
)

# Save policy weights
model.save("dqn_airsim_leader_policy_new_mpc_contrchange")
