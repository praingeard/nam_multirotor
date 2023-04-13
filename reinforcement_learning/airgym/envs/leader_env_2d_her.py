import airsim
import numpy as np
import math

from gym import spaces
from airgym.envs.airsim_env import AirSimGoalEnv
from PIL import Image

import signal
import os
import datetime
from typing import Dict, Any
import math

STOP_SIGNAL = signal.SIGABRT
START_SIGNAL = signal.SIGFPE

class AirSimLeader2DHerEnv(AirSimGoalEnv):
    
    sim_target = "Goal_4" 
    camera_name = '0'
    image_type = airsim.ImageType.Scene
    
    def __init__(self, ip_address, step_length, image_shape):
        super(AirSimLeader2DHerEnv, self).__init__(image_shape)
        self.step_length = step_length
        self.time = 0
        self.image_shape = image_shape
        self.dist = 75
        self.vehicle_name = "Leader"
        self.drone_names = ["Leader"]
        self.drone = airsim.MultirotorClient(ip=ip_address)
        #self.drone_names = ["Drone1", "Drone2", "Drone3", "Leader"]
        
        # Set detection radius in [cm]
        self.drone.simSetDetectionFilterRadius(AirSimLeader2DHerEnv.camera_name,  AirSimLeader2DHerEnv.image_type, 100 * 200)
        # Add desired object name to detect in wild card/regex format, useful to change drone pose
        self.drone.simAddDetectionFilterMeshName( AirSimLeader2DHerEnv.camera_name,  AirSimLeader2DHerEnv.image_type,  AirSimLeader2DHerEnv.sim_target)
        print(self.drone.simGetDetections(AirSimLeader2DHerEnv.camera_name,  AirSimLeader2DHerEnv.image_type))

        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "prev_position": np.zeros(2),
        }


        self._action_space = spaces.Discrete(4)
        self._obs_space = spaces.Dict({"observation": spaces.Box(0, 255, shape=image_shape, dtype=np.uint8), 
                                      "desired_goal": spaces.Discrete(75), 
                                      "achieved_goal": spaces.Discrete(75)}, seed=0)
        self.reward_range = (-2000,2000)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )
        
        #info init
        self.image_show_size = (144, 256)
        self.frame_num = 0    
        self.action_num = 0
        #init empty image of right size
        self.last_image = Image.fromarray(np.reshape(np.zeros(36864), self.image_show_size))
        self.truncated = False
        self.achieved_goal = 75
        self.desired_goal = 20

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        #also need to start MPC
        self.drone.reset()
        for drone in self.drone_names:
            self.drone.enableApiControl(True, vehicle_name=drone)
            self.drone.armDisarm(True, vehicle_name=drone)
            
        self.drone.moveToPositionAsync(0, 0, -3, 10).join()

        # Set home position and velocity
        self.drone.moveByVelocityAsync(1, 0, 0, 0.5, vehicle_name="Leader").join()
        
        
    def _get_info(self) -> Dict[str, Any]:
        info = {
            "frame_num": self.frame_num,
            "last_image": self.last_image,
            "action_number": self.action_num,
        }

        return info

    def transform_obs(self, responses):
        self.frame_num += 1
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        if img1d.size != 36864:
            img1d = np.zeros(36864)
            print("Error in the image gotten")
        img2d = np.reshape(img1d, self.image_show_size)
        self.last_image = Image.fromarray(img2d)
        im_final = np.array(self.last_image.resize((84, 84)).convert("L"))
        arr = im_final.reshape([84, 84, 1])
        return arr

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request], vehicle_name=self.vehicle_name)
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState(vehicle_name=self.vehicle_name)
        self.state["prev_position"] = self.state["position"]
        if all(v == 0 for v in self.state["prev_position"]):
            self.state["prev_position"] = np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
        self.state["position"] =np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
        collision = self.drone.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        self.state["collision"] = collision
        obs= {"observation" : image,
              "desired_goal" : self.desired_goal,
              "achieved_goal" : self.achieved_goal}
        return obs

    def _do_action(self, action):
        self.action_num += 1
        quad_offset = self.interpret_action(action)
        # quad_vel = self.drone.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
        # self.drone.moveByVelocityAsync(
        #     quad_vel.x_val + quad_offset[0],
        #     quad_vel.y_val + quad_offset[1],
        #     quad_vel.z_val + quad_offset[2],
        #     1,
        #     vehicle_name=self.vehicle_name
        # ).join()
        quad_pos = self.drone.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        self.drone.moveToPositionAsync(
            quad_pos.x_val + quad_offset[0],
            quad_pos.y_val + quad_offset[1],
            -3,
            1,
            vehicle_name=self.vehicle_name
        ).join()

    def _compute_reward(self):
        dist = 75
        self.time = self.time + 1
        done = False
        collision = False
        goal_reached = False

        pt = [float(i) for i in self.drone.simGetObjectPose(AirSimLeader2DHerEnv.sim_target).position][:2]

        quad_pt = self.state["position"]/10
        old_quad_pt = self.state["prev_position"]/10
        print(quad_pt,old_quad_pt)
        dist = np.sqrt((quad_pt[0]-pt[0])**2 + (quad_pt[1]-pt[1])**2)
        if self.state["collision"]:
            reward = -1000
            collision = True
        else:
            old_dist = np.sqrt((old_quad_pt[0]-pt[0])**2 + (old_quad_pt[1]-pt[1])**2)
            quad_dist = np.abs(dist - old_dist)
            if dist < 20:
                reward = 1000
                goal_reached = True
            elif dist<=self.dist:
                reward = (5+(5*quad_dist)) + (pt[0]-dist) 
            else:
                reward =-(5+ 5*quad_dist) + (pt[0]-dist)
        self.dist = dist
        self.achieved_goal = dist
        if goal_reached or collision:
            print("env reset")
            done = True
            self.reset()
        reward = int(reward)
        print(reward, self.achieved_goal)


        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        info = self._get_info
        reward, done = self._compute_reward()
        info = {}

        return obs, reward, done, info 

    def reset(self):
        self.time = 0
        self._setup_flight()
        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "prev_position": np.zeros(2),
        }
        image = self._get_obs()
        self.dist = 75
        return  image

    def interpret_action(self, action):
        offset = self.step_length
        if action == 0:
            quad_offset = (offset, 0, 0)
        elif action == 1:
            quad_offset = (0, offset, 0)
        elif action == 2:
            quad_offset = (0, -offset, 0)
        elif action == 3:
            quad_offset = (-offset, 0, 0)

        return quad_offset
    
    #observation space already set up by the Airgym env class
    @property
    def action_space(self) -> spaces.Discrete:
        """
        Return Gym's action space.
        """
        return self._action_space

    @property
    def observation_space(self) -> spaces.Dict:
        """
        Return Gym's action space.
        """
        return self._obs_space

