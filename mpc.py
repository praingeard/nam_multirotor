import airsim
import numpy as np
import signal
from random import uniform
import configparser

#from pysigset import sigaddset


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
		#pwm[i] = max(0.0, min(1.0,0.594 + thrust[i] / (air_density_ratio * max_thrust)))
		pwm[i] = max(0.0, min(1.0,0.594 + thrust[i] / max_thrust))
	return pwm

def moments_to_pwm(Mx,My,T):
	max_torque = 0.055
	#print(Mx)
	Mx = np.sign(Mx) * min(np.abs(Mx/286), max_torque)
	My = np.sign(My) * min(np.abs(My/275.39), max_torque)
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

def set_relative_states_delta(delta, lenlist, widthlist):
	(wmin,wmax) = widthlist
	(lmin,lmax) = lenlist
	d1x = -(lmax - lmin)/2*delta
	d3x = -(lmax - lmin)*delta 
	d1y = -wmax/2 + delta*wmax/2
	d3y = wmax/2 - delta*wmax/2
	d2x = lmin/2*delta
	leaderx = lmin
	#state in x,y is [x,y, u, v, g*theta, g*phi, gq, gp]
	current_state_xy = [[d1x, 0, 0, 0, d1y, 0, 0, 0],[d2x, 0, 0, 0, 0, 0, 0, 0],[d3x, 0, 0, 0, d3y, 0, 0, 0],[leaderx, 0, 0, 0, 0, 0, 0, 0]]
	#state in z is z,w
	current_state_z = [[-2,0],[-2,0],[-2,0],[-2,0]]
	(relative_state_xy, relative_state_z) = setup_relative_states(current_state_xy, current_state_z)
	return relative_state_xy, relative_state_z

def set_init_state(lenlist, widthlist):
	(wmin,wmax) = widthlist
	(lmin,lmax) = lenlist
	d1x = 0
	d3x = 0
	d1y = -wmax/2
	d3y = wmax/2
	d2x = 0
	leaderx = lmin
	#state in x,y is [x,y, u, v, g*theta, g*phi, gq, gp]
	init_state_xy = [[d1x, 0, 0, 0, d1y, 0, 0, 0],[d2x, 0, 0, 0, 0, 0, 0, 0],[d3x, 0, 0, 0, d3y, 0, 0, 0],[leaderx, 0, 0, 0, 0, 0, 0, 0]]
	#state in z is z,w
	init_state_z = [[-2,0],[-2,0],[-2,0],[-2,0]]
	(relative_state_xy, relative_state_z) = setup_relative_states(init_state_xy, init_state_z)
	return init_state_xy, init_state_z, relative_state_xy, relative_state_z


#compute relative state between each drone and the leader 
def setup_relative_states(current_state_xy, current_state_z):
	relative_state_xy = [[],[],[]]
	relative_state_z = [[],[],[]]
	for drone_id in range(drone_num):
		for pose_id in range(8):
			relative_state_xy[drone_id].append(current_state_xy[drone_id][pose_id] - current_state_xy[-1][pose_id])
		for pose_id in range(2):
			relative_state_z[drone_id].append(current_state_z[drone_id][pose_id] - current_state_z[-1][pose_id])
	#print("relative states", relative_state_xy)
	#print("relative states z", relative_state_z)
	return relative_state_xy, relative_state_z

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

def get_states_with_diff(drone_init_state_xy, drone_id, rel_state_xy, rel_state_z):
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
	#print("rix is", rix)
	#print("riy is", riy)
	for i in  range (4):
		rix_diff[i] = drone_init_state_xy[drone_id][i] + rix[i] - rel_state_xy[drone_id][i]
		riy_diff[i] = drone_init_state_xy[drone_id][i + 4] + riy[i] - rel_state_xy[drone_id][i + 4]
	for i in range(2):
		riz_diff[i] = riz[i] - rel_state_z[drone_id][i]


	#print("states with diff for drone", drone_id)
	#print(rix_diff)
		
	return rix_diff, riy_diff, riz_diff

def change_controller_gains():
	config = configparser.ConfigParser()
	config.read('config.ini')
	V = float(config['DEFAULT']['V'])
	print(V)
	V = 1.2
	betaval = [[-2.,-1.],[-7.,-3.],[10.,30.],[2.5,4.]]
	gammaval = [[0.0,100.],[0.0,100.]]
	new_beta_gains_x = []
	new_beta_gains_y = []
	new_gamma_gains = []
	for beta in betaval:
		new_gain_val = compute_new_gain(beta[0], beta[1],V)
		new_beta_gains_x.append(new_gain_val)
		new_beta_gains_y.append(new_gain_val)
	for gamma in gammaval:
		new_gamma_gains.append(compute_new_gain(gamma[0], gamma[1], V))
	return new_beta_gains_x,new_beta_gains_y,new_gamma_gains

def compute_new_gain(gainmin, gainmax, v):
	alpha = 3*gainmax
	beta = gainmin
	new_gain = alpha*q_approx(v) + beta
	return new_gain

def q_approx(x):
	poly = 1/12*np.exp(-x**2/2) + 1/4*np.exp(-2*x**2/3)
	return poly

#initial setup 
drone_num = 3
drone_names = ["Drone1", "Drone2", "Drone3"]
leader_names = ["Leader"]
g = 9.81
timestep_max = 300.0
timestep_rate = 0.001
lenlist = (4, 8)
widthlist = (0, 4)
L_matrix = [[2, -1, -1, 0],[-1, 2, -1, -1],[-1, -1, 2, 0],[0, 0, 0, 0]]
A_matrix = [[0, 1, 1, 0],[0, 1, 1, 1],[0, 1, 1, 0],[0, 0, 0, 0]]
(init_state_xy, init_state_z, relative_state_xy, relative_state_z) = set_init_state(lenlist, widthlist)

#signal handling
controller_started = True
controller_reset = False
signal.signal(STOP_SIGNAL, handle_stop_signal)
signal.signal(START_SIGNAL, handle_start_signal)
signals = {START_SIGNAL, STOP_SIGNAL}
# Block the signals so that they can be waited for with sigwaitinfo
# for sig in signals:
#     sigaddset(signal.SIG_BLOCK, sig)

#setup gains 
#beta_gains = [-5, -1, 0.3, 0]
#beta_gains = [-2.5,-3.9,0,0]
beta_gains_x = [-3,-5, 15, 3]
beta_gains_y = [-3,-5, 15, 3]
gamma_gains = [2.9, 3.9]


# connect to the AirSim simulator and get current state
client = airsim.MultirotorClient()
for drone in drone_names: 
	client.armDisarm(False, drone)
client.reset()
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
change = False
while(timestep <= timestep_max):
	# Read the current value of delta from a configuration file
	config = configparser.ConfigParser()
	config.read('config.ini')
	delta = float(config['DEFAULT']['delta'])
	controller_rate = int(config['DEFAULT']['controller_rate'])

	if iteration%controller_rate == 0:
		#print("delta", delta)
		#(relative_state_xy, relative_state_z) = set_relative_states_delta(delta, lenlist, widthlist)
		(beta_gains_x, beta_gains_y, gamma_gains) = change_controller_gains()
		#print(beta_gains_x, gamma_gains)
		print('new gains are', beta_gains_x, gamma_gains)
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
	#print('mpc started')
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
		rix_hat, riy_hat, riz_hat = get_states_with_diff(init_state_xy, drone_id, relative_state_xy, relative_state_z)
		j_list_speed = []
		speed_amount = 0
		for j in range(drone_num+1):
			rjx_hat, rjy_hat, rjz_hat = get_states_with_diff(init_state_xy, j, relative_state_xy, relative_state_z)
			Mxi = Mxi - A_matrix[drone_id][j]*(beta_gains_x[0]*(riy_hat[0] - rjy_hat[0]) + beta_gains_x[1]*(riy_hat[1] - rjy_hat[1]) + 
															beta_gains_x[2] *(rix_hat[2] - rjx_hat[2]) + beta_gains_x[3] * (rix_hat[3] - rjx_hat[3]))
			#print("Mx for drone", drone_id, "and j ", j, " is ", Mxi)
			Myi = Myi - A_matrix[drone_id][j]*(beta_gains_y[0]*(rix_hat[0] - rjx_hat[0]) + beta_gains_y[1]*(rix_hat[1] - rjx_hat[1]) + 
															beta_gains_y[2] *(riy_hat[2] - rjy_hat[2]) + beta_gains_y[3] * (riy_hat[3] - rjy_hat[3]))
			Ti = Ti + A_matrix[drone_id][j]*(gamma_gains[0]*(riz_hat[0] - rjz_hat[0]) + gamma_gains[1]*(riz_hat[1] - rjz_hat[1]))
			if iteration % 5 == 0:
				j_list_speed.append(riy_hat[1] - rjy_hat[1])
				speed_amount+=(A_matrix[drone_id][j]* beta_gains_x[1] * (riy_hat[1] - rjy_hat[1]))
		pwm = moments_to_pwm(Mxi,Myi,Ti)
		#print(pwm[0], pwm[1], pwm[2], pwm[3])
		f1 = client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], timestep_rate/30, drone_names[drone_id])
		drone_commands.append(f1)
	rx_leader, ry_leader, rz_leader = get_leader_state(0)
	# f_leader = client.moveByMotorPWMsAsync(0.594, 0.594, 0.594, 0.594, timestep_rate/30, vehicle_name="Leader")
	# #f_leader = client.moveByRollPitchYawZAsync(0.0, 0.05, 0, rz_leader[0]-0.2, timestep_rate/2, vehicle_name="Leader")
	# drone_commands.append(f_leader)
	for f in drone_commands:
		f.join()

airsim.wait_key('Press any key to reset to original state')

for drone in drone_names: 
	client.armDisarm(False, drone)
client.reset()

# that's enough fun for now. let's quit cleanly
for drone in drone_names: 
	client.enableApiControl(False, drone)


