import airsim
import numpy as np
import math

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

import signal
import os
import datetime

STOP_SIGNAL = signal.SIGABRT
START_SIGNAL = signal.SIGFPE

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

        self.mpc_pid = self.get_mpc_pid()


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
        for drone in self.drone_names:
            self.drone.enableApiControl(True, vehicle_name=drone)
            self.drone.armDisarm(True, vehicle_name=drone)

        # Set home position and velocity
        # self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        # f1 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 1, vehicle_name="Drone1")
        # f2 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 1, vehicle_name="Drone2")
        # f3 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 1, vehicle_name="Drone3")
        f4 = self.drone.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 1, vehicle_name="Leader")
        # f1.join()
        # f2.join()
        # f3.join()
        f4.join()
        # f1 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone1")
        # f2 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone2")
        # f3 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Drone3")
        f4 = self.drone.moveByVelocityAsync(10, 0, 0, 0.5, vehicle_name="Leader")
        # f1.join()
        # f2.join()
        # f3.join()
        f4.join()
        self.send_start_signal(self.mpc_pid)
        


    def transform_obs(self, responses):
        print('reshaping image')
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))
        arr = im_final.reshape([84, 84, 1])
        return arr

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
    
    
    def thrust_to_pwm(self,thrust):
        max_thrust = 4.179446268
        air_density = 1.293
        standard_air_density = 1.225
        air_density_ratio = air_density / standard_air_density
        pwm = np.zeros(4)
        for i in range(len(thrust)):
            pwm[i] = max(0.0, min(1.0,thrust[i] / (air_density_ratio * max_thrust)))
        return pwm

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
    
    # def _do_action(self, action):
    #     base_pwm = 0.594
    #     rotor_states = self.drone.getRotorStates(vehicle_name = self.vehicle_name)
    #     thrusts = [0,0,0,0]
    #     for i in range(4):
    #         thrusts[i]=rotor_states.rotors[i]["thrust"]
    #     pwm_rotors = self.thrust_to_pwm(thrusts)
    #     quad_offset = self.interpret_action(action)
    #     pwm1_offset = quad_offset[0] + quad_offset[1] + quad_offset[2]
    #     pwm2_offset = quad_offset[0] - quad_offset[1] - quad_offset[2]
    #     pwm3_offset = quad_offset[0] - quad_offset[1] + quad_offset[2]
    #     pwm4_offset = quad_offset[0] + quad_offset[1] - quad_offset[2]
    #     self.drone.moveByMotorPWMsAsync(pwm1_offset + pwm_rotors[0], 
    #                                     pwm2_offset + pwm_rotors[1], 
    #                                     pwm3_offset + pwm_rotors[2], 
    #                                     pwm4_offset + pwm_rotors[3], 
    #                                     0.5, 
    #                                     vehicle_name=self.vehicle_name)

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

        return obs, reward, done, self.info

    def reset(self):
        self.time = 0
        self.send_stop_signal(self.mpc_pid)
        self._setup_flight()
        image = self._get_obs()
        return  image

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
