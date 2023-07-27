# nam_multirotor

# Reinforcement Learning for UAV formations using Airsim
This code is an implementation of SAC and MADDPG for UAV formations.

# Prerequisites
You need to have installed the airsim simulator at https://microsoft.github.io/AirSim/. The main branch is operating on windows 10/11, the other branches on Ubuntu 20.04.

MADDPG needs to be installed for using the maddpg branch.

For simulation environments, please copy the Content folder onto your Unreal map directory. 

# Branches
In the main branch, test surrounding real time gain changes are conducted. In the computerlab branch, main simulations using formation transformation are done. Finally, the maddpg branch includes all the scripts surrounding MADDPG. 

# Main Algorithms

- **Airgym** : this is where gym environments are defined. You can change the used script by dqn_leader.py in the init file of this module.
- **Controller launch**: mpc.py is the main script for defining the controller parameters. It is launched as well as a full simulation if ran using the master_mpc.py script. 
- **reinforcement_learning**: this folder includes logs and necessary scripts for running simulations and models. dqn_leader.py runs the simulator to train a model, while dqn_run.py reruns the subsequent model.

#  Relevant Environments
For standard consensus: sacobsimage_mpc 
For gain change: sacobsimage_mpc_divergent
For formation transformation: sacobsimage_mpc_change
For sac-multi: sacobsimage


# How to launch 

main branch: 
- Consensus launch without gain change: run master_mpc.py
- Consensus launch with gain change : comment/uncomment line 251 in the mpc.py script and change the 

computerlab branch:
- MPC launch with/without formation transformation: run master_mpc.py and comment/uncomment line 255 in the mpc.py script
- Multi SAC launch: execute dqn_runsubproc.py

For each launch, please make sure that the script is launching the correct environment following the previous setion.
