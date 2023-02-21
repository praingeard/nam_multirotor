import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile

#initial setup 
drone_num = 3
drone_names = ["Drone1", "Drone2", "Drone3"]
g = 9.81
timestep_max = 1000
timestep_rate = 0.1
L_matrix = [[3, -1, -1, -1],[-1, 2, -1, 0],[-1, -1, 2, 0],[-1, 0, 0, -1]]
#state in x,y is [x,y, u, v, -g*theta, g*phi, -gq, gp]
drone_init_state_xy = [[2, 0, 0, 0, 0, 0, 0, 0],[0, 2, 0, 0, 0, 0, 0, 0],[0, -2, 0, 0, 0, 0, 0, 0],[4, 0, 0, 0, 0, 0, 0, 0]]
#state in z is z,w
drone_init_state_z = [[-2, 0],[-2, 0],[-2, 0],[-2, 0]]
relative_state_xy = []
relative_state_z = []

#compute relative state between each drone and the leader 
for drone_id in range(drone_num):
	for pose_id in range(8):
		relative_state_xy[drone_id][pose_id] = drone_init_state_xy[-1][pose_id] - drone_init_state_xy[drone_id][pose_id]
	for pose_id in range(2):
		relative_state_z[drone_id][pose_id] = drone_init_state_z[-1][pose_id] - drone_init_state_z[drone_id][pose_id]	

#get initial inertia (maybe need to get it dynamically)
max_thrust = 4.179446268
mass =  1.0
num_rotor = 4
assembly_weight = 0.055
arm_length = 0.2275
body_box_x = 0.180
body_box_y = 0.11
body_box_z = 0.040
box_mass = mass - num_rotor * assembly_weight
Ixx = box_mass/12 * (body_box_y^2 + body_box_z^2) + num_rotor*(1/np.sqrt(2)*arm_length)^2
Iyy = box_mass/12 * (body_box_x^2 + body_box_z^2) + num_rotor*(1/np.sqrt(2)*arm_length)^2

#setup gains 
beta_gains = [1, 5, 20, 3]
gamma_gains = [2.9, 3.9]
states = []

#setup useful functions 
def get_states(drone_id):
	drone_name = drone_names[drone_id]
	state = client.getMultirotorState(vehicle_name=drone_name)
	rix = [state.kinematics_estimated.position.x_val, state.kinematics_estimated.linear_velocity.x_val, 
	-g*state.kinematics_estimated.orientation.x_val, -g*state.kinematics_estimated.angular_velocity.x_val]
	riy = [state.kinematics_estimated.position.y_val, state.kinematics_estimated.linear_velocity.y_val, 
	g*state.kinematics_estimated.orientation.y_val, g*state.kinematics_estimated.angular_velocity.y_val]
	riz = [state.kinematics_estimated.position.z_val, -state.kinematics_estimated.linear_velocity.z_val]
	return rix, riy, riz

def get_states_with_diff(drone_id):
	rix, riy, riz = get_states(drone_id)
	rix_diff = [0, 0, 0, 0]
	riy_diff = [0, 0, 0, 0]
	riz_diff = [0, 0]
	for i in  range (4):
		rix_diff[i] = rix[i] - relative_state_xy[drone_id][i]
		riy_diff[i] = riy[i] - relative_state_xy[drone_id][i + 4]
	for i in range(2):
		riz_diff[i] = riz[i] - relative_state_z[drone_id][i]
		
	return rix_diff, riy_diff, riz_diff


# connect to the AirSim simulator and get current state
client = airsim.MultirotorClient()
client.confirmConnection()
for drone in drone_names: 
	client.enableApiControl(True, drone)
	client.armDisarm(True, drone)
	f = client.takeoffAsync(vehicle_name= drone)
	f.join()
	states.append(client.getMultirotorState(vehicle_name=drone))

#MPC controller for current step
timestep = 0
while(timestep <= timestep_max):
	timestep = timestep + timestep_rate
	Mx = []
	My = []
	for drone_id in range(drone_num):
		Mxi = 0
		Myi = 0
		rix_hat, riy_hat, riz_hat = get_states_with_diff(drone_id)
		for j in range(drone_num+1):
			rjx_hat, rjy_hat, rjz_hat = get_states_with_diff(j)
			Mxi = Mxi - L_matrix[drone_id][j]*(beta_gains[0]*(rix_hat[0] - rjx_hat[0]) + beta_gains[1]*(rix_hat[1] - rjx_hat[1]) + 
														    beta_gains[2] *(rix_hat[2] - rjx_hat[2]) + beta_gains[3] * (rix_hat[3] - rjx_hat[3]))
			Myi = Myi - L_matrix[drone_id][j]*(beta_gains[0]*(riy_hat[0] - rjy_hat[0]) + beta_gains[1]*(riy_hat[1] - rjy_hat[1]) + 
														    beta_gains[2] *(riy_hat[2] - rjy_hat[2]) + beta_gains[3] * (riy_hat[3] - rjy_hat[3]))
			Ti = Ti - L_matrix[drone_id][j]*(gamma_gains[0]*(riz_hat[0] - rjz_hat[0]) + gamma_gains[1]*(riz_hat[1] - rjz_hat[1]))
		desired_roll_rate = -g/Iyy*Mxi*timestep_rate
		desired_pitch_rate = g/Ixx*Myi*timestep_rate
		desired_yaw_rate = 0
		desired_throttle = Ti/max_thrust
		client.moveByAngleRatesThrottleAsync(desired_roll_rate, desired_pitch_rate, desired_yaw_rate, desired_throttle, timestep_rate, drone_names[drone_id])


airsim.wait_key('Press any key to reset to original state')

for drone in drone_names: 
	client.armDisarm(False, drone)
client.reset()

# that's enough fun for now. let's quit cleanly
for drone in drone_names: 
	client.enableApiControl(False, drone)


