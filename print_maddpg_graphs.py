import pickle


with open('./reinforcement_learning/learning_curves/maddpg_rewards.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

with open('./reinforcement_learning/learning_curves/maddpg_agrewards.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

with open('./reinforcement_learning/learning_curves/maddpg_len.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

