import airsim
import numpy as np
import signal
from pysigset import sigaddset


STOP_SIGNAL = signal.SIGABRT
START_SIGNAL = signal.SIGFPE

def handle_stop_signal(signum, frame):
	print("received stop signal")
	global controller_reset
	#controller_reset = True


def handle_start_signal(signum, frame):
	print("received start signal")
	global controller_started
	controller_started = True

def thrust_to_pwm(thrust):
	max_thrust = 4.179446268
	air_density = 1.293
	standard_air_density = 1.225
	air_density_ratio = air_density / standard_air_density
	pwm = np.zeros(4)
	for i in range(thrust.size):
		pwm[i] = max(0.0, min(1.0,0.594 + thrust[i] / (air_density_ratio * max_thrust)))
	return pwm

def moments_to_pwm(Mx,My,T):
	Mx = Mx/10
	My = My/10
	k = 0.0000155
	l = 0.2275
	w1sq = (T/(4*k)) + (Mx/(4*k*l)) + (My/(4*k*l)) 
	w2sq = (T/(4*k)) - (Mx/(4*k*l)) - (My/(4*k*l)) 
	w3sq = (T/(4*k)) - (Mx/(4*k*l)) + (My/(4*k*l)) 
	w4sq = (T/(4*k)) + (Mx/(4*k*l)) - (My/(4*k*l)) 
	F1 = k*w1sq
	F2 = k*w2sq
	F3 = k*w3sq
	F4 = k*w4sq
	f_u = np.array([F1, F2, F3, F4])
	pwm = thrust_to_pwm(f_u)
	return pwm


#initial setup 
drone_num = 3
drone_names = ["Drone1", "Drone2", "Drone3"]
leader_names = ["Leader"]
g = 9.81
timestep_max = 300.0
timestep_rate = 0.1
L_matrix = [[3, -1, -1, -1],[-1, 3, -1, -1],[-1, -1, 3, -1],[0, 0, 0, 0]]
A_matrix = [[0, 1, 1, 0],[1, 0, 1, 1],[1, 1, 0, 0],[0, 0, 0, 0]]
#state in x,y is [x,y, u, v, g*theta, g*phi, gq, gp]
drone_init_state_xy = [[0, 0, 0, 0, -2, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 2, 0, 0, 0],[4, 0, 0, 0, 0, 0, 0, 0]]
#state in z is z,w
drone_init_state_z = [[-2, 0],[-2, 0],[-2, 0],[-2, 0]]
relative_state_xy = [[],[],[]]
relative_state_z = [[],[],[]]

#signal handling
controller_started = True
controller_reset = False
signal.signal(STOP_SIGNAL, handle_stop_signal)
signal.signal(START_SIGNAL, handle_start_signal)
signals = {START_SIGNAL, STOP_SIGNAL}
# Block the signals so that they can be waited for with sigwaitinfo
for sig in signals:
    sigaddset(signal.SIG_BLOCK, sig)



#compute relative state between each drone and the leader 
for drone_id in range(drone_num):
	for pose_id in range(8):
		relative_state_xy[drone_id].append(drone_init_state_xy[drone_id][pose_id] - drone_init_state_xy[-1][pose_id])
	for pose_id in range(2):
		relative_state_z[drone_id].append(drone_init_state_z[drone_id][pose_id] - drone_init_state_z[-1][pose_id])
print("relative states", relative_state_xy)
print("relative states z", relative_state_z)

#setup gains 
beta_gains = [-2, -1, 0.3, 0]
gamma_gains = [2.9, 3.9]

#setup useful functions 
def get_states(drone_id):
	drone_name = drone_names[drone_id]
	state = client.getMultirotorState(vehicle_name=drone_name)
	rix = [state.kinematics_estimated.position.x_val, state.kinematics_estimated.linear_velocity.x_val, 
	-g*state.kinematics_estimated.orientation.x_val, -g*state.kinematics_estimated.angular_velocity.x_val]
	riy = [state.kinematics_estimated.position.y_val, state.kinematics_estimated.linear_velocity.y_val, 
	g*state.kinematics_estimated.orientation.y_val, g*state.kinematics_estimated.angular_velocity.y_val]
	riz = [state.kinematics_estimated.position.z_val, state.kinematics_estimated.linear_velocity.z_val]
	return rix, riy, riz

def get_leader_state(leader_id):
	leader_name = leader_names[leader_id]
	state = client.getMultirotorState(vehicle_name=leader_name)
	rix = [state.kinematics_estimated.position.x_val, state.kinematics_estimated.linear_velocity.x_val, 
	-g*state.kinematics_estimated.orientation.x_val, -g*state.kinematics_estimated.angular_velocity.x_val]
	riy = [state.kinematics_estimated.position.y_val, state.kinematics_estimated.linear_velocity.y_val, 
	g*state.kinematics_estimated.orientation.y_val, g*state.kinematics_estimated.angular_velocity.y_val]
	riz = [state.kinematics_estimated.position.z_val, state.kinematics_estimated.linear_velocity.z_val]
	return rix, riy, riz

def get_states_with_diff(drone_id):
	rx_leader, ry_leader, rz_leader = get_leader_state(0)
	if drone_id == drone_num:
		rix_diff = [0,0,0,0]
		riy_diff = [0,0,0,0]
		for i in  range (4):
			rix_diff[i] = drone_init_state_xy[-1][i] + rx_leader[i]
			riy_diff[i] = drone_init_state_xy[-1][i + 4] + ry_leader[i]
		riz_diff = rz_leader
		return rix_diff, riy_diff, riz_diff
	rix, riy, riz = get_states(drone_id)
	rix_diff = [0, 0, 0, 0]
	riy_diff = [0, 0, 0, 0]
	riz_diff = [0, 0]
	print("rix is", rix)
	print("riy is", riy)
	for i in  range (4):
		rix_diff[i] = drone_init_state_xy[drone_id][i] + rix[i] - relative_state_xy[drone_id][i]
		riy_diff[i] = drone_init_state_xy[drone_id][i + 4] + riy[i] - relative_state_xy[drone_id][i + 4]
	for i in range(2):
		riz_diff[i] = riz[i] - relative_state_z[drone_id][i]


	print("states with diff for drone", drone_id)
	print(rix_diff)
		
	return rix_diff, riy_diff, riz_diff

# connect to the AirSim simulator and get current state
client = airsim.MultirotorClient()
client.confirmConnection()
for drone in drone_names: 
	client.enableApiControl(True, drone)
	client.armDisarm(True, drone)
	# f = client.takeoffAsync(vehicle_name= drone)
	# f.join()

for leader in leader_names: 
	client.enableApiControl(True, leader)
	client.armDisarm(True, leader)
	# f = client.takeoffAsync(vehicle_name= leader)
	# f.join()

# f1 = client.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone1")
# f2 = client.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone2")
# f3 = client.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Drone3")
# f4 = client.moveByMotorPWMsAsync(0.6, 0.6, 0.6, 0.6, 5, vehicle_name="Leader")
# f1.join()
# f2.join()
# f3.join()
# f4.join()

#MPC controller for current step
timestep = 0
iteration = 0
while(timestep <= timestep_max):
	#wait for command for setup ended
	if controller_reset == True:
		print('controller asked reset')
		timestep = 0
		iteration = 0
		controller_started = False
		controller_reset = False
	if controller_started == False:
		print("Waiting for signal to start")
		siginfo = signal.sigwaitinfo({START_SIGNAL})
	print('mpc started')
	#relative_state_xy, relative_state_z = get_current_relative_states(relative_state_xy, relative_state_z)
	timestep = timestep + timestep_rate
	iteration = iteration +1
	Mxi = 0
	Myi = 0
	Ti = 0
	drone_commands = []
	for drone_id in range(drone_num):
		Mxi = 0
		Myi = 0
		Ti = 0
		rix_hat, riy_hat, riz_hat = get_states_with_diff(drone_id)
		for j in range(drone_num+1):
			rjx_hat, rjy_hat, rjz_hat = get_states_with_diff(j)
			Mxi = Mxi - A_matrix[drone_id][j]*(beta_gains[0]*(riy_hat[0] - rjy_hat[0]) + beta_gains[1]*(riy_hat[1] - rjy_hat[1]) + 
														    beta_gains[2] *(rix_hat[2] - rjx_hat[2]) + beta_gains[3] * (rix_hat[3] - rjx_hat[3]))
			print("Mx for drone", drone_id, "and j ", j, " is ", Mxi)
			Myi = Myi - A_matrix[drone_id][j]*(beta_gains[0]*(rix_hat[0] - rjx_hat[0]) + beta_gains[1]*(rix_hat[1] - rjx_hat[1]) + 
														    beta_gains[2] *(riy_hat[2] - rjy_hat[2]) + beta_gains[3] * (riy_hat[3] - rjy_hat[3]))
			Ti = Ti + A_matrix[drone_id][j]*(gamma_gains[0]*(riz_hat[0] - rjz_hat[0]) + gamma_gains[1]*(riz_hat[1] - rjz_hat[1]))
		pwm = moments_to_pwm(Mxi,Myi,Ti)
		print(pwm[0], pwm[1], pwm[2], pwm[3])
		f1 = client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], timestep_rate/2, drone_names[drone_id])
		drone_commands.append(f1)
	rx_leader, ry_leader, rz_leader = get_leader_state(0)
	#f_leader = client.moveByMotorPWMsAsync(0.61, 0.61, 0.61, 0.61, timestep_rate/2, vehicle_name="Leader")
	#f_leader = client.moveByRollPitchYawZAsync(0.0, 0.1, 0, rz_leader[0]-0.2, timestep_rate/2, vehicle_name="Leader")
	#drone_commands.append(f_leader)
	for f in drone_commands:
		f.join()

airsim.wait_key('Press any key to reset to original state')

for drone in drone_names: 
	client.armDisarm(False, drone)
client.reset()

# that's enough fun for now. let's quit cleanly
for drone in drone_names: 
	client.enableApiControl(False, drone)


