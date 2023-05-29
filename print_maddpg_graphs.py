import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./reinforcement_learning/learning_curves/maddpg_rewards.pkl', 'rb') as f:
    rewards = pickle.load(f)
    #print(rewards)

with open('./reinforcement_learning/learning_curves/maddpg_agrewards.pkl', 'rb') as f:
    agrewards = pickle.load(f)
    #print(agrewards)

with open('./reinforcement_learning/learning_curves/maddpg_len.pkl', 'rb') as f:
    ep_length = pickle.load(f)
    #print(ep_length)

#print(len(rewards))

drone_1_rew = []
drone_2_rew = []
drone_3_rew = []
drone_4_rew = []
drone_1_agrew = []
drone_2_agrew = []
drone_3_agrew = []
drone_4_agrew = []
drone_1_len = []
drone_2_len = []
drone_3_len = []
drone_4_len = []

for i in range(len(rewards)):
    if i%4 == 0:
        drone_1_rew.append(rewards[i])
        drone_1_agrew.append(agrewards[i])
        drone_1_len.append(ep_length[i])
    if i%4 == 1:
        drone_2_rew.append(rewards[i])
        drone_2_agrew.append(agrewards[i])
        drone_2_len.append(ep_length[i])
    if i%4 == 2:
        drone_3_rew.append(rewards[i])
        drone_3_agrew.append(agrewards[i])
        drone_3_len.append(ep_length[i])
    if i%4 == 3:
        drone_4_rew.append(rewards[i])
        drone_4_agrew.append(agrewards[i])
        drone_4_len.append(ep_length[i])

episodes = np.linspace(1,int(len(agrewards)),int(len(agrewards)))
print (len(rewards))
print(len(agrewards))
print(len(ep_length))

plt.plot(episodes,agrewards)
# plt.plot(episodes,drone_2_agrew)
# plt.plot(episodes,drone_3_agrew)
# plt.plot(episodes,drone_4_agrew)
# plt.plot(episodes,drone_1_len)
# plt.plot(episodes,drone_2_len)
# plt.plot(episodes,drone_3_len)
# plt.plot(episodes,drone_4_len)
#print(drone_4_len)
plt.show()