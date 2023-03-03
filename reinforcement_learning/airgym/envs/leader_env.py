import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimLeaderEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super(AirSimLeaderEnv, self).__init__(image_shape)
        self.step_length = step_length
        self.step_num = 0
        self.time = 0
        self.image_shape = image_shape
        self.vehicle_name = "Leader"
        self.drone_names = ["Drone1", "Drone2", "Drone3", "Leader"]

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )
        self.info = {}
        self.truncated = False

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        #also need to start MPC
        self.drone.reset()
        self.drone.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.drone.armDisarm(True, vehicle_name=self.vehicle_name)

        # Set home position and velocity
        # self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        self.drone.moveToPositionAsync(0, 0, -3, 10,vehicle_name=self.vehicle_name).join()
        self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name=self.vehicle_name).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request], vehicle_name=self.vehicle_name)
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState(vehicle_name=self.vehicle_name)

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = False
        for drone_name in self.drone_names:
            collision_current = self.drone.simGetCollisionInfo(vehicle_name=drone_name).has_collided
            if collision_current == True:
                 collision = True
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
            vehicle_name=self.vehicle_name
        ).join()

    def _compute_reward(self):
        thresh_dist = 7
        thresh_time = 15
        self.time = self.time + 1
        beta = 1

        z = -3
        pts = [
            np.array([10, 0, -3]),
            np.array([25, 0, -3]),
            np.array([50, 0, -3]),
            # np.array([193.5974, -55.0786, -46.32256]),
            # np.array([369.2474, 35.32137, -62.5725]),
            # np.array([541.3474, 143.6714, -32.07256]),
        ]

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            reward = -1000
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )
                print(dist)

            if dist > thresh_dist:
                reward = -10
            else:
                
                reward_time = -self.time*15
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
                reward = reward_dist + reward_speed
                print(reward_dist, reward_speed)
        done = False
        if reward <= -10 or self.time >=10:
            done = True

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.truncated, self.info

    def reset(self):
        self.time = 0
        self._setup_flight()
        return  self._get_obs(), self.info

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
