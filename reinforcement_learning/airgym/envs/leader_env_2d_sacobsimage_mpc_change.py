import airsim
import numpy as np
import math

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

import configparser

import signal
import os
import datetime
from typing import Dict, Any
import math
import matplotlib.pyplot as plt

STOP_SIGNAL = signal.SIGABRT
START_SIGNAL = signal.SIGFPE

def get_app_file_path(file):
    """Return the absolute path of the app's files. They should be in the same folder as this py file."""
    folder,_ = os.path.split(__file__)
    file_path = os.path.join(folder,file)
    return file_path

class AirSimLeader2DEnv(AirSimEnv):
    
    sim_target = "Goal_4" 
    camera_name = '0'
    image_type = airsim.ImageType.Scene
    
    def __init__(self, ip_address, step_length, image_shape):
        super(AirSimLeader2DEnv, self).__init__(image_shape)
        self.step_length = step_length
        self.time = 0
        self.image_shape = image_shape
        self.dist = 75
        self.vehicle_name = "Leader"
        self.drone_names = ["Leader"]
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone_names = ["Drone1", "Drone2", "Drone3", "Leader"]
        self.delta_list = [0.0]
        self.t_list = [0]
        self.success = 0
        self.failure = 0
        self.success_rate = 0 
        
        # Set detection radius in [cm]
        self.drone.simSetDetectionFilterRadius(AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type, 100 * 200)
        # Add desired object name to detect in wild card/regex format, useful to change drone pose
        self.drone.simAddDetectionFilterMeshName( AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type,  AirSimLeader2DEnv.sim_target)
        print(self.drone.simGetDetections(AirSimLeader2DEnv.camera_name,  AirSimLeader2DEnv.image_type))

        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "prev_position": np.zeros(2),
        }


        self._obs_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self._action_space = spaces.Box(np.array([-1., -1., 0.]), np.array([1., 1., 1.]),shape=(3,), dtype=np.float32)
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
        self.config = configparser.ConfigParser()
        self.config.read(get_app_file_path('config.ini'))

        V = 1.2
        delta = 0.0
        controller_rate = 30


        self.config['DEFAULT']['V'] = str(V)
        self.config['DEFAULT']['delta'] = str(delta)
        self.config['DEFAULT']['controller_rate'] = str(controller_rate)

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

    def __del__(self):
        self.drone.reset()

    def get_mpc_pid(self):
        # Define the directory where the log files are stored
        log_directory = "./log_processes"
        # Define the expected naming convention for the log files
        file_name_pattern = "%Y-%m-%d_%H-%M-%S_*_log.txt"
        # Get a list of all the log files in the log directory
        log_files = [f for f in os.listdir(log_directory) if os.path.isfile(os.path.join(log_directory, f))]
        # Filter out any files that do not match the expected naming convention
        log_files = [f for f in log_files if datetime.datetime.strptime(f.split("_")[0], "%Y-%m-%d")]
        # Sort the remaining files by their modification time, with the most recent file first
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_directory, x)), reverse=True)
        # Get the path to the most recent log file, if one exists
        if log_files:
            most_recent_log_file = os.path.join(log_directory, log_files[0])
        else:
            most_recent_log_file = None
        print(f"Most recent log file: {most_recent_log_file}")
        with open(most_recent_log_file, "r") as log_file:
            log_contents = log_file.read()
            pid2 = int(log_contents.split("\n")[1].split(":")[1])
        print(f"PID of mpc_process: {pid2}")
        return pid2
    
    def send_start_signal(self, mpc_pid):
        os.kill(mpc_pid, START_SIGNAL)
        print("started mpc control")

    def send_stop_signal(self, mpc_pid):
        os.kill(mpc_pid, STOP_SIGNAL)
        print("stopped mpc control")

    def _setup_flight(self):
        #also need to start MPC
        self.drone.reset()
        self.delta_list = [0.0]
        self.t_list = [0]
        for drone in self.drone_names:
            self.drone.enableApiControl(True, vehicle_name=drone)
            self.drone.armDisarm(True, vehicle_name=drone)

            
        #self.drone.moveToPositionAsync(0, 0, -3, 10, vehicle_name="Leader").join()
        # self.drone.moveToPositionAsync(-2, -2, -3, 10, vehicle_name="Drone1").join()
        # self.drone.moveToPositionAsync(-2, 0, -3, 10, vehicle_name="Drone2").join()
        # self.drone.moveToPositionAsync(-2, 2, -3, 10, vehicle_name="Drone3").join()


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
        images = []
        self.frame_num += 1
        for response_drone in responses:
            img1d = np.array(response_drone[0].image_data_float, dtype=float)
            img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
            if img1d.size != 36864:
                img1d = np.zeros(36864)
                print("Error in the image gotten")
            img2d = np.reshape(img1d, self.image_show_size)
            image_depth = Image.fromarray(img2d)
            images.append(image_depth)
        images_vect = images
        widths, heights = zip(*(i.size for i in images_vect))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('L', (total_width, max_height))
        

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        self.last_image = new_im
        im_final = np.array(self.last_image.resize((84, 84)).convert("L"))
        arr = im_final.reshape([84, 84, 1])
        return arr

    def _get_obs(self):
        responses = []
        for namer in self.drone_names:
            responses.append(self.drone.simGetImages([self.image_request], vehicle_name=self.vehicle_name))
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState(vehicle_name=self.vehicle_name)
        self.state["prev_position"] = self.state["position"]
        if all(v == 0 for v in self.state["prev_position"]):
            self.state["prev_position"] = np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
        self.state["position"] =np.array([int(10*self.drone_state.kinematics_estimated.position.x_val), int(10*self.drone_state.kinematics_estimated.position.y_val)])
        collision = self.drone.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        collision = False
        for drone_name in self.drone_names:
           collision_current = self.drone.simGetCollisionInfo(vehicle_name=drone_name).has_collided
           if collision_current == True:
                collision = True
        self.state["collision"] = collision
        return image

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
        
    def get_energy(self):
        energy = 0
        try:
            self.config.read('config.ini')
            energy = float(self.config['DEFAULT']['energy'])
            energy_checked = True
            self.config['DEFAULT']['energy_checked'] = str(energy_checked)

            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
  
        except KeyError as e:
            print(e)
        
        return energy, energy_checked

    def _compute_reward(self):
        dist = 75
        self.time = self.time + 1
        self.t_list.append(self.time)
        done = False
        collision = False
        goal_reached = False
        try:
            self.config.read('config.ini')
            delta = float(self.config['DEFAULT']['delta'])
  
        except KeyError as e:
            print("error delta")

        pt = [float(i) for i in self.drone.simGetObjectPose(AirSimLeader2DEnv.sim_target).position][:2]

        quad_pt = self.state["position"]/10
        old_quad_pt = self.state["prev_position"]/10
        print(quad_pt,old_quad_pt)
        dist = np.sqrt((quad_pt[0]-pt[0])**2 + (quad_pt[1]-pt[1])**2)
        if self.state["collision"]:
            reward = -1000
            collision = True
            self.failure = self.failure + 1
        else:
            energy, energy_checked = self.get_energy()
            print(energy)
            old_dist = np.sqrt((old_quad_pt[0]-pt[0])**2 + (old_quad_pt[1]-pt[1])**2)
            quad_dist = np.abs(dist - old_dist)
            if self.time > 30:
                collision = True
                reward = -1000
                self.failure = self.failure + 1
            elif dist < 15:
                reward = 1000
                self.success = self.success + 1
                goal_reached = True
                plt.plot(self.t_list,self.delta_list)
                print(self.delta_list)
                plt.xlabel('t (step)')
                plt.ylabel('delta')
                #plt.savefig('deltaevol2')
            elif dist<=self.dist:
                reward = (5+(5*quad_dist)) + (pt[0]-dist) #- delta*0.1
            else:
                reward =-(5+ 5*quad_dist) + (pt[0]-dist) #-delta*0.1
            # elif dist<=self.dist:
            #     reward = (5+(5*quad_dist)) + (pt[0]-dist) - delta*0.1 - energy*0.05
            # else:
            #     reward =-(5+ 5*quad_dist) + (pt[0]-dist) -delta*0.1 - energy*0.05
        self.dist = dist
        if goal_reached or collision:
            print("env reset")
            done = True
            self.success_rate = self.success/(self.success + self.failure)
            trials = self.success + self.failure
            print(self.success_rate, trials)
            self.reset()
        reward = int(reward)
        print(reward, dist)
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
        V = 1.2
        delta = 0.0


        self.config['DEFAULT']['V'] = str(V)
        self.config['DEFAULT']['delta'] = str(delta)

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return  image

    def interpret_action(self, action):
        quad_offset = (action[0], action[1], 0)
        delta = action[2]
        self.delta_list.append(delta)
        self.config['DEFAULT']['delta'] = str(delta)
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        # offset = self.step_length
        # if action == 0:
        #     quad_offset = (offset, 0, 0)
        # elif action == 1:
        #     quad_offset = (0, offset, 0)
        # elif action == 2:
        #     quad_offset = (0, -offset, 0)
        # elif action == 3:
        #     quad_offset = (-offset, 0, 0)

        return quad_offset
    
    #observation space already set up by the Airgym env class
    @property
    def action_space(self) -> spaces.Discrete:
        """
        Return Gym's action space.
        """
        return self._action_space
    
    @property
    def observation_space(self) -> spaces.Box:
        """
        Return Gym's action space.
        """
        return self._obs_space
