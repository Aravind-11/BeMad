from rl.agents.dqn import TrainDQN
from rl.data import get_train_test_val, load_csv, get_behaviour_data
from rl.utils import rounded_dict
from tensorflow.keras.layers import Dense, Dropout
import pickle
from keras.regularizers import l2,l1

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

# Hyper-parameters.
episodes = 60000  # Total number of episodes
warmup_steps = 1700  # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 10000
collect_steps_per_episode = 100
collect_every = 1

target_update_period = 800  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 1

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(5, activation=None)]  # No activation, pure Q-values

learning_rate = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.5  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon``

# {'backdoor': 7842, 'banker': 4875, 'cryptominer': 990, 'deceptor': 4614, 'downloader': 9274, 'normal': 8961, 'pua': 7149, 'ransomware': 2236, 'spyware': 11585}
# {'backdoor': 0.13632096791016235, 'banker': 0.0847442895386434, 'cryptominer': 0.017209609567847582, 'deceptor': 0.08020721065257448, 'downloader': 0.1612140597295136, 'normal': 0.1557730417550325, 'pua': 0.12427424121266906, 'ransomware': 0.03886938080172444, 'spyware': 0.20138719883183256}

# {'backdoor': 0, 'banker': 1, 'cryptominer': 2, 'deceptor': 3, 'downloader': 4, 'normal': 5, 'pua': 6, 'ransomware': 7, 'spyware': 8}


# Dropping categorical columns and columns that have the same value for all rows.
X_train, y_train, X_test, y_test = load_csv(r"C:\Users\ATHARVA\Documents\Atharva\Malware Detection IP\Malware-IP\rl_v2\train.csv",r"C:\Users\ATHARVA\Documents\Atharva\Malware Detection IP\Malware-IP\rl_v2\test.csv","Label",["Flow ID","Timestamp"], normalization=False)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,val_frac=0.3)

# no_rows = 2
# import numpy as np
# X_train, y_train, X_test, y_test, X_val, y_val = get_behaviour_data(no_rows, X_train, y_train, X_test, y_test, X_val, y_val)

model_path = r"C:\Users\ATHARVA\Documents\Atharva\Malware Detection IP\Malware-IP\rl_v2\rl.pkl"
model = TrainDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update, model_path=model_path)

# loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
model.compile_model(X_train, y_train, layers=layers)
model.q_net.summary()
h = model.train(X_val, y_val, "Accuracy")
import json
with open('final.csv', 'w') as f:
    json.dump(list(map(str, h)), f)
# import matplotlib.pyplot as plt
# plt.plot(h)
# plt.xlabel("epochs")
# plt.ylabel("Loss")
# print(len(h))
# print(h[0], "<<<<<<<")

# plt.savefig("history.png")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
