import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneMultiEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super(AirSimDroneMultiEnv, self).__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position_drone1": np.zeros(3),
            "position_drone2": np.zeros(3),
            "position_drone3": np.zeros(3),
            "collision": False,
            "prev_position_drone1": np.zeros(3),
            "prev_position_drone2": np.zeros(3),
            "prev_position_drone3": np.zeros(3)
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
        self.drone.reset()
        self.drone.enableApiControl(True, "Drone1")
        self.drone.enableApiControl(True, "Drone2")
        self.drone.enableApiControl(True, "Drone3") 
        self.drone.armDisarm(True, "Drone1")
        self.drone.armDisarm(True, "Drone2")
        self.drone.armDisarm(True, "Drone3")

        # Set home position and velocity
        # self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        self.drone.moveToPositionAsync(0, 0, -2, 10, vehicle_name="Drone1").join()
        self.drone.moveToPositionAsync(0, 2, -2, 10, vehicle_name="Drone2").join()
        self.drone.moveToPositionAsync(0, -2, -2, 10, vehicle_name="Drone3").join()
        self.drone.moveByVelocityAsync(10, 0, 0, 1, vehicle_name="Drone1").join()
        self.drone.moveByVelocityAsync(10, 0, 0, 1, vehicle_name="Drone2").join()
        self.drone.moveByVelocityAsync(10, 0, 0, 1, vehicle_name="Drone3").join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses_drone1 = self.drone.simGetImages([self.image_request], vehicle_name= "Drone1")
        responses_drone2 = self.drone.simGetImages([self.image_request], vehicle_name= "Drone2")
        responses_drone3 = self.drone.simGetImages([self.image_request], vehicle_name= "Drone3")
        image1 = self.transform_obs(responses_drone1)
        image2 = self.transform_obs(responses_drone2)
        image3 = self.transform_obs(responses_drone3)
        self.drone_state_drone1 = self.drone.getMultirotorState(vehicle_name = "Drone1")
        self.drone_state_drone2 = self.drone.getMultirotorState(vehicle_name = "Drone2")
        self.drone_state_drone3 = self.drone.getMultirotorState(vehicle_name = "Drone3")

        self.state["prev_position_drone1"] = self.state["position_drone1"]
        self.state["prev_position_drone2"] = self.state["position_drone2"]
        self.state["prev_position_drone3"] = self.state["position_drone3"]
        self.state["position_drone1"] = self.drone_state_drone1.kinematics_estimated.position
        self.state["position_drone2"] = self.drone_state_drone2.kinematics_estimated.position
        self.state["position_drone3"] = self.drone_state_drone3.kinematics_estimated.position
        self.state["velocity_drone1"] = self.drone_state_drone1.kinematics_estimated.linear_velocity
        self.state["velocity_drone2"] = self.drone_state_drone2.kinematics_estimated.linear_velocity
        self.state["velocity_drone3"] = self.drone_state_drone3.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image1 + image2 + image3

    def _do_action(self, action, drone_name):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState(vehicle_name = drone_name).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    def _compute_reward(self, drone_name):
        thresh_dist = 7
        beta = 1

        z = -10

        position = "position_" + drone_name  
        velocity = "velocity_" + drone_name

        if drone_name == "drone1":
            pts = [
                np.array([50, 0, -2])
            ]
        if drone_name == "drone2":
            pts = [
                np.array([50, 2, -2])
            ]
        if drone_name == "drone3":
            pts = [
                np.array([50, -2, -2])
            ]

        quad_pt = np.array(
            list(
                (
                    self.state[position].x_val,
                    self.state[position].y_val,
                    self.state[position].z_val,
                )
            )
        )

        if self.state["collision"]:
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )

            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state[velocity].x_val,
                            self.state[velocity].y_val,
                            self.state[velocity].z_val,
                        ]
                    )
                    - 0.5
                )
                reward = reward_dist + reward_speed

        done = False
        if reward <= -10:
            done = True

        return reward, done

    def step(self, action):
        self._do_action(action, "Drone1")
        self._do_action(action, "Drone2")
        self._do_action(action, "Drone3")
        obs = self._get_obs()
        reward1, done1 = self._compute_reward("drone1")
        reward2, done2 = self._compute_reward("drone2")
        reward3, done3 = self._compute_reward("drone3")

        reward = reward1 + reward2 + reward3
        done = done1 and done2 and done3

        return obs, reward, done, self.truncated, self.info

    def reset(self):
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
