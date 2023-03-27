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

        self.state = []
        self.drone_names = ["Drone1", "Drone2", "Drone3", "Leader"]
        for drone_id in range(len(self.drone_names)):
            self.state.append({
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        })
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, pixels_as_float = True, compress = False
        )

        self.info_n = {'0': [], '1': [], '2': [], '3': []}
        self.truncated_n = [False, False, False, False]

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        for drone in self.drone_names:
            self.drone.enableApiControl(True, vehicle_name=drone)
            self.drone.armDisarm(True, vehicle_name=drone)

        # Set home position and velocity
        f1 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone1")
        f2 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone2")
        f3 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone3")
        f4 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Leader")
        f1.join()
        f2.join()
        f3.join()
        f4.join()
        f1 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone1")
        f2 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone2")
        f3 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone3")
        f4 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Leader")
        f1.join()
        f2.join()
        f3.join()
        f4.join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape(-1)

    def _get_obs(self):
        obs_n  = []
        for drone_id in range(len(self.drone_names)):
            drone = self.drone_names[drone_id]
            responses_drone = self.drone.simGetImages([self.image_request], vehicle_name= drone)
            image = self.transform_obs(responses_drone)
            self.drone_state = self.drone.getMultirotorState(vehicle_name = drone)
            self.state[drone_id]["prev_position"] = self.state[drone_id]["position"]
            self.state[drone_id]["position"] = self.drone_state.kinematics_estimated.position
            self.state[drone_id]["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
            collision = self.drone.simGetCollisionInfo(vehicle_name = drone).has_collided
            self.state[drone_id]["collision"] = collision
            obs_n.append(image)
        return obs_n

    def _do_action(self, action_n):
        quad_offset_n = self.interpret_action(action_n)
        f_list = []
        for drone_id in range(len(self.drone_names)):
            drone_name = self.drone_names[drone_id]
            quad_vel = self.drone.getMultirotorState(vehicle_name = drone_name).kinematics_estimated.linear_velocity
            f_list.append(self.drone.moveByVelocityAsync(
                quad_vel.x_val + quad_offset_n[drone_id][0],
                quad_vel.y_val + quad_offset_n[drone_id][1],
                quad_vel.z_val + quad_offset_n[drone_id][2],
                5,
                vehicle_name = drone_name
            ))
        for f_value in f_list:
            f_value.join()

    def _compute_reward(self):
        thresh_dist = 7
        beta = 1

        z = -10

        position = "position"
        velocity = "velocity"

        pts_n = [0,0,0,0]
        quad_pt_n = [0,0,0,0]
        rewards = [0,0,0,0]
        done = [False, False, False, False]
        for drone_id in range(len(self.drone_names)):
            if drone_id == 0:
                pts_n[drone_id] = [np.array([48, -2, -5]), np.array([49, -2, -5]),np.array([50, -2, -5])]
            if drone_id == 1:
                pts_n[drone_id] = [np.array([50, 0, -5]), np.array([51, 0, -5]),np.array([52, 0, -5])]
            if drone_id == 2:
                pts_n[drone_id] = [np.array([48, 2, -5]), np.array([49, 2, -5]),np.array([50, 2, -5])]
            if drone_id == 3:
                pts_n[drone_id] = [np.array([52, 0, -5]), np.array([53, 0, -5]),np.array([54, 0, -5])]

            quad_pt_n[drone_id] = np.array(
                list(
                    (
                        self.state[drone_id][position].x_val,
                        self.state[drone_id][position].y_val,
                        self.state[drone_id][position].z_val,
                    )
                )
            )

            if self.state[drone_id]["collision"]:
                rewards[drone_id] = -100
            else:
                dist = 10000000
                for i in range(0, len(pts_n[drone_id]) - 1):
                    dist = min(
                        dist,
                        np.linalg.norm(np.cross((quad_pt_n[drone_id] - pts_n[drone_id][i]), (quad_pt_n[drone_id] - pts_n[drone_id][i + 1])))
                        / np.linalg.norm(pts_n[drone_id][i] - pts_n[drone_id][i + 1]),
                    )

                if dist > thresh_dist:
                    rewards[drone_id] = -10
                else:
                    reward_dist = math.exp(-beta * dist) - 0.5
                    reward_speed = (
                        np.linalg.norm(
                            [
                                self.state[drone_id][velocity].x_val,
                                self.state[drone_id][velocity].y_val,
                                self.state[drone_id][velocity].z_val,
                            ]
                        )
                        - 0.5
                    )
                    rewards[drone_id] = reward_dist + reward_speed
            done[drone_id] = False
            if rewards[drone_id] <= -10:
                done[drone_id] = True

        return rewards, done

    def step(self, action_n):
        self._do_action(action_n)
        obs_n = self._get_obs()
        reward_n, done_n = self._compute_reward()

        return obs_n, reward_n, done_n, self.info_n

    def reset(self):
        self._setup_flight()
        return  self._get_obs()

    def interpret_action(self, action_n):
        quad_offset_n = []
        print(action_n[0][0])
        for drone in range(len(self.drone_names)):
            action = np.argmax(action_n[drone])
            if action == 0:
                quad_offset_n.append((self.step_length, 0, 0))
            elif action == 1:
                quad_offset_n.append((0, self.step_length, 0))
            elif action == 2:
                quad_offset_n.append((0, 0, self.step_length))
            elif action == 3:
                quad_offset_n.append((-self.step_length, 0, 0))
            elif action == 4:
                quad_offset_n.append((0, -self.step_length, 0))
            elif action == 5:
                quad_offset_n.append((0, 0, -self.step_length))
            else:
                quad_offset_n.append((0, 0, 0))

        return quad_offset_n
