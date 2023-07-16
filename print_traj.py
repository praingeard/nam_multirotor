import airsim 
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

class MultiRotorVehicle():
    
    def __init__(self, vhc_name, vhc_start):
        self.client = airsim.MultirotorClient()
        self.name = vhc_name
        self.traj_x = []
        self.traj_y = []
        self.vhc_start = vhc_start
        sim_target = "Goal_4" 
        camera_name = '0'
        image_type = airsim.ImageType.Scene
        # Set detection radius in [cm]
        self.client.simSetDetectionFilterRadius(camera_name,  image_type, 100 * 200)
        # Add desired object name to detect in wild card/regex format, useful to change drone pose
        self.client.simAddDetectionFilterMeshName(camera_name,  image_type, sim_target)
        self.target = [float(i) for i in self.client.simGetObjectPose(sim_target).position][:2]
        
    def update_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.name)
        self.traj_x.append(state.kinematics_estimated.position.x_val + self.vhc_start[0])
        self.traj_y.append(state.kinematics_estimated.position.y_val + self.vhc_start[1])
        
    def dist_to_target(self):
        dist = np.sqrt((self.traj_x[-1]-4.0-self.target[0])**2 + (self.traj_y[-1]-self.target[1])**2)
        return dist
    

class AirsimPrinter():
    def __init__(self, update_rate=1, max_dist=20, size_rect = (0.1,0.1), vhc_start = [(0.0,-2.0),(0.0,0.0),(0.0,2.0),(4.0,0.0)]):
       vhc_names = ["Drone1", "Drone2", "Drone3", "Leader"]
       self.vehicle_list = [MultiRotorVehicle(vhc_name=vhc_names[i], vhc_start=vhc_start[i]) for i in range(len(vhc_names))]
       self.client = airsim.MultirotorClient()
       self.timestamp = self.client.getMultirotorState().timestamp
       self.last_traj_update = 0
       self.update_rate = update_rate
       self.max_dist = max_dist
       self.size_rect = size_rect
       self.fig, self.ax = plt.subplots()
       self.create_base_graph()
       
       
    def sleep_until_next_pos(self):
        while self.timestamp < self.last_traj_update + self.update_rate:
            self.timestamp = self.client.getMultirotorState().timestamp
        self.last_traj_update = self.timestamp
        
    def check_end_ep(self):
        leader = self.vehicle_list[-1]
        if leader.dist_to_target() < self.max_dist:
            print("done")
            return True
        return False
        
        
    def get_new_pos(self):
        for vhc in self.vehicle_list:
            vhc.update_position()
            
    def create_base_graph(self):
        realobs_list = [(7.4,-3.6),(11.6,0.7),(8.0,5.1)]
        fakeobs_list = [(3.5,3.0),(6.0,2.8)]
        goal = (30.0,0.0)
        for realobs in realobs_list:
            rect = patches.Rectangle(realobs, self.size_rect[0], self.size_rect[1], linewidth=1, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect)
        for fakeobs in fakeobs_list:
            rect = patches.Rectangle(fakeobs, self.size_rect[0], self.size_rect[1], linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)

            
        
    def print_graphs(self):
        for vhc in self.vehicle_list:
            print(vhc.name, vhc.traj_x, vhc.traj_y)
            plt.plot(vhc.traj_x, vhc.traj_y, label=vhc.name)
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.savefig('trajformtrans.png')
        
def main():
    
    printer = AirsimPrinter(update_rate=100, max_dist=20, size_rect = (0.5,1.0))
    printer.get_new_pos()
    while not printer.check_end_ep():
        printer.get_new_pos()
        time.sleep(0.5)
        printer.sleep_until_next_pos()
    printer.print_graphs()
    
        
if __name__ == "__main__":
    main()
