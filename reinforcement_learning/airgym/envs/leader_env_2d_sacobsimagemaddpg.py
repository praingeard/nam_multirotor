import airsim
import numpy as np
import math

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

import signal
import os
import datetime
from typing import Dict, Any
import math

STOP_SIGNAL = signal.SIGABRT
START_SIGNAL = signal.SIGFPE

class AirSimLeader2DEnv(AirSimEnv):
    
    sim_target = "Goal_4" 
    camera_name = '0'
    image_type = airsim.ImageType.Scene
    
    def __init__(self, ip_address, step_length, image_shape):
        super(AirSimLeader2DEnv, self).__init__(image_shape)
        self.n = 4
        self.step_length = step_length
        self.time = 0
        self.image_shape = image_shape
        self.dist = 75
        self.vehicle_name = "Leader"
        #self.drone_names = ["Leader"]
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone_names = ["Drone1", "Drone2", "Drone3", "Leader"]
        
        # Set detection radius in [cm]
        self.drone.simSetDetectionFilterRadius(AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type, 100 * 200)
        # Add desired object name to detect in wild card/regex format, useful to change drone pose
        self.drone.simAddDetectionFilterMeshName( AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type,  AirSimLeader2DEnv.sim_target)
        print(self.drone.simGetDetections(AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type))

        self.state =[]
        for drone in self.drone_names:
            self.state.append({
            "position": np.zeros(2),
            "collision": False,
            "prev_position": np.zeros(2),
        })


        self.observation_space = [spaces.Box(low=-np.inf, high = +np.inf, shape = (7056,), dtype=np.float32) for i in range(self.n)]
        #self._action_space = [spaces.Box(-1., 1.,shape=(2,), dtype=np.float32) for i in range(self.n)]
        self._action_space = [spaces.Discrete(3) for i in range(self.n)]
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
        self.dones_n = [False, False, False, False]
        self.rewards_n = [0., 0., 0., 0.] 

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        #also need to start MPC
        self.drone.reset()
        setups = []
        for drone in self.drone_names:
            self.drone_state = self.drone.getMultirotorState(vehicle_name=drone)
            self.drone.enableApiControl(True, vehicle_name=drone)
            self.drone.armDisarm(True, vehicle_name=drone)
            setups.append(self.drone.moveToPositionAsync(
                self.drone_state.kinematics_estimated.position.x_val, 
                self.drone_state.kinematics_estimated.position.y_val, 
                -3, 10, vehicle_name=drone))
        for setup in setups:
            setup.join()
        
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
        im_final = np.array(self.last_image.resize((84,84)).convert("L"))
        #arr = im_final.reshape([7056, 1])
        return im_final.reshape(-1)

    def _get_obs(self):
        image_arr = []
        for drone_id in range(len(self.drone_names)):
            vehicle_name = self.drone_names[drone_id]
            responses = self.drone.simGetImages([self.image_request], vehicle_name=vehicle_name)
            image = self.transform_obs(responses)
            image_arr.append(image)
            self.drone_state = self.drone.getMultirotorState(vehicle_name=vehicle_name)
            self.state[drone_id]["prev_position"] = self.state[drone_id]["position"]
            if all(v == 0 for v in self.state[drone_id]["prev_position"]):
                self.state[drone_id]["prev_position"] = np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
            self.state[drone_id]["position"] =np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
            collision = self.drone.simGetCollisionInfo(vehicle_name=vehicle_name).has_collided
            self.state[drone_id]["collision"] = collision
        return image_arr

    def _do_action(self, action):
        actions = []
        for drone_id in range(len(self.drone_names)): 
            self.action_num += 1
            drone = self.drone_names[drone_id]
            quad_offset = self.interpret_action(action[drone_id])
            quad_pos = self.drone.getMultirotorState(vehicle_name=drone).kinematics_estimated.position
            print(quad_pos.x_val + quad_offset[0], quad_pos.y_val + quad_offset[1])
            if (self.state[drone_id]["collision"] == False):
                actions.append(self.drone.moveToPositionAsync(
                    quad_pos.x_val + quad_offset[0],
                    quad_pos.y_val + quad_offset[1],
                    -3,
                    1,
                    vehicle_name=drone
                ))
        for act in actions:
            act.join()

    def _compute_reward(self):
        rewards = []
        dones = []
        dists = []
        for drone_id in range(len(self.drone_names)):
            dist = 75
            self.time = self.time + 1
            done = False
            collision = False
            goal_reached = False

            pt = [float(i) for i in self.drone.simGetObjectPose(AirSimLeader2DEnv.sim_target).position][:2]

            quad_pt = self.state[drone_id]["position"]/10
            old_quad_pt = self.state[drone_id]["prev_position"]/10
            print(quad_pt,old_quad_pt)
            dist = np.sqrt((quad_pt[0]-pt[0])**2 + (quad_pt[1]-pt[1])**2)
            old_dist = np.sqrt((old_quad_pt[0]-pt[0])**2 + (old_quad_pt[1]-pt[1])**2)
            if abs(old_dist - dist) <= 0.05 and self.time > 5:
                reward = -250
                collision = True
            elif self.state[drone_id]["collision"] and self.time > 1:
                reward = -250
                collision = True
            else:
                old_dist = np.sqrt((old_quad_pt[0]-pt[0])**2 + (old_quad_pt[1]-pt[1])**2)
                quad_dist = np.abs(dist - old_dist)
                if dist < 20:
                    reward = 250
                    goal_reached = True
                elif dist<=self.dist:
                    reward = ((5+(5*quad_dist)) + (pt[0]-dist) )/4
                else:
                    reward =-((5+ 5*quad_dist) + (pt[0]-dist))/4
            print(collision, goal_reached)
            self.dist = dist
            dists.append(dist)
            if goal_reached or collision:
                print("env reset")
                done = True
            if self.dones_n[drone_id]:
                reward = 0.
                done = True
            self.rewards_n[drone_id] = reward
            self.dones_n[drone_id] = done
            
        print(self.rewards_n, self.dones_n)

        return self.rewards_n, self.dones_n

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        info = self._get_info
        reward, done = self._compute_reward()
        info = {'0': [], '1': [], '2': [], '3': []}

        return obs, reward, done, info 

    def reset(self):
        self.dones_n = [False, False, False, False]
        self.rewards_n = [0., 0., 0., 0.] 
        self.time = 0
        self._setup_flight()
        self.state =[]
        for drone in self.drone_names:
            self.state.append({
            "position": np.zeros(2),
            "collision": False,
            "prev_position": np.zeros(2),
        })
        image = self._get_obs()
        self.dist = 75
        return  image

    def interpret_action(self, action):
        #quad_offset = (action[0], action[1], 0)
        action_current = np.argmax(action)
        offset = self.step_length
        if action_current == 0:
            quad_offset = (offset, 0, 0)
        elif action_current == 1:
            quad_offset = (0, offset, 0)
        elif action_current == 2:
            quad_offset = (0, -offset, 0)
        elif action_current == 3:
            quad_offset = (-offset, 0, 0)

        return quad_offset
    
    #observation space already set up by the Airgym env class
    @property
    def action_space(self) -> spaces.Discrete:
        """
        Return Gym's action space.
        """
        return self._action_space
