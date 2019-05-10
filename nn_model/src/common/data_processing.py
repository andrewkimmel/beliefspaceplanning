import scipy.io
import numpy as np
import pickle

DATA = scipy.io.loadmat('../data/sim_data_discrete_v13_d4_m1.mat')['D']
DATA = np.array(DATA)

state_min_arr = np.amin(DATA, axis=0)
state_max_arr = np.amax(DATA, axis=0)
state_mean_arr = np.mean(DATA, axis=0)
state_std_arr = np.std(DATA, axis=0)

d_Data = DATA[:4] - DATA[6:10]
delta_min_arr = np.amin(d_Data, axis=0)
delta_max_arr = np.amax(d_Data, axis=0)
delta_mean_arr = np.mean(d_Data, axis=0)
delta_std_arr = np.std(d_Data, axis=0)

with open('../data/state_min_arr', 'wb') as pickle_file:
    pickle.dump(state_min_arr, pickle_file)

with open('../data/state_max_arr', 'wb') as pickle_file:
    pickle.dump(state_max_arr, pickle_file)

with open('../data/state_mean_arr', 'wb') as pickle_file:
    pickle.dump(state_mean_arr, pickle_file)

with open('../data/state_std_arr', 'wb') as pickle_file:
    pickle.dump(state_std_arr, pickle_file)

with open('../data/delta_min_arr', 'wb') as pickle_file:
    pickle.dump(delta_min_arr, pickle_file)

with open('../data/delta_max_arr', 'wb') as pickle_file:
    pickle.dump(delta_max_arr, pickle_file)

with open('../data/delta_mean_arr', 'wb') as pickle_file:
    pickle.dump(delta_mean_arr, pickle_file)

with open('../data/delta_std_arr', 'wb') as pickle_file:
    pickle.dump(delta_std_arr, pickle_file)
