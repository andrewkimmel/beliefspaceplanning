''' 
Author: Avishai Sintov
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/sim_nn_node/common/')
from data_normalization import *
import pickle
import random
import signal

ratio = '0.99'

class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


class predict_nn:
    def __init__(self):

        save_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/models/'
        model_name = 'sim_cont_trajT_bs512_model512_BS64_loadT_ho40.pkl' # Name of the model we want to depickle
        self.model_path = save_path + model_name

        print('[predict_nn] Loading NN model with ' + ratio + ' of data...')
        with open(save_path + '/normalization_arr_sim_cont_trajT_bs512_model512_BS64_loadT_ho' + ratio + '_py2', 'rb') as pickle_file:
            x_norm_arr, y_norm_arr = pickle.load(pickle_file)

        self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
        self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]

        with open(self.model_path, 'rb') as pickle_file:
            self.model = torch.load(pickle_file, map_location='cpu')

    def normalize(self, data):
        return (data - self.x_mean_arr[:data.shape[-1]]) / self.x_std_arr[:data.shape[-1]]

    def denormalize(self, data, ):
        return data * self.y_std_arr[:data.shape[-1]] + self.y_mean_arr[:data.shape[-1]]

    def predict(self, sa):

        # print "In predict"
        inpt = self.normalize(sa)

        flag = True
        while flag:
            try:
                # with Timeout(10):
                inpt = torch.tensor(inpt, dtype=torch.float)
                state_delta = self.model(inpt)
                
                flag = False
            except:
                print "Timeout"

        state_delta = state_delta.detach().numpy()
        state_delta = self.denormalize(state_delta)
        next_state = (sa[...,:4] + state_delta)
        # print "Out predict"
        return next_state


# if __name__ == "__main__":
#     task = 'real_A' 
#     held_out = .1
#     test_traj = 2
#     _ , arg1, arg2, arg3 = argv
#     nn_type = '1'
#     method = ''
#     save_path = 'save_model/robotic_hand_real/pytorch/'

#     trajectory_path_map = {
#         'real_A': '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/testpaths_cyl35_d_v0.pkl', 
#         'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
#         'transferA2B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
#         'transferB2A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl',
#         }
#     trajectory_path = trajectory_path_map[task]

#     with open(trajectory_path, 'rb') as pickle_file:
#         trajectory = pickle.load(pickle_file)#, encoding='latin1')

#     def make_traj(trajectory, test_traj):
#         acts = trajectory[0][test_traj][:-1]
#         real_positions = trajectory[1][test_traj][:,[0,1,11,12]]
#         return np.append(real_positions, acts, axis=1)

#     NN = predict_nn()
#     state_dim = 4
#     action_dim = 6

#     states = []
#     BATCH = True
#     BATCH = False
#     if BATCH:
#         batch_size = 4
#         out=[make_traj(trajectory, i) for i in range(batch_size)]

#         lengths = [len(traj) for traj in out]
#         min_length = min(lengths)
#         batches = [traj[0:min_length] for traj in out]
#         traj = np.stack(batches,0)

#         true_states = traj[:,:,:state_dim]
#         state = traj[:,0,:state_dim]
#         start_state = np.copy(state)

#         actions = traj[..., state_dim:state_dim+action_dim]

#         for i in range(traj.shape[1]):
#             states.append(state)
#             action = actions[:,i]
#             action = np.append(action, start_state, axis=1)
#             sa = np.concatenate((state, action), -1)
#             state = NN.predict(sa)
#         states = np.stack(states, 1)

#     else:
#         traj = make_traj(trajectory, test_traj)
#         true_states = traj[:,:state_dim]
#         state = traj[0][:state_dim]
#         start_state = np.copy(state)
    
#         for i, point in enumerate(traj):
#             states.append(state)
#             action = point[state_dim:state_dim+action_dim]
#             action = np.concatenate((action, start_state), axis=0)
#             if cuda: action = action.cuda() 
#             pdb.set_trace()
#             sa = np.concatenate((state, action), 0)
#             state = NN.predict(sa)
#         states = np.stack(states, 0)

#     if BATCH:
#         plt.figure(1)
#         ix = 3
#         plt.scatter(traj[ix, 0, 0], traj[ix, 0, 1], marker="*", label='start')
#         plt.plot(traj[ix, :, 0], traj[ix, :, 1], color='blue', label='Ground Truth', marker='.')
#         plt.plot(states[ix, :, 0], states[ix, :, 1], color='red', label='NN Prediction')
#         plt.axis('scaled')
#         plt.title('Bayesian NN Prediction -- pos Space')
#         plt.legend()
#         plt.show()
#     else:
#         plt.figure(1)
#         plt.scatter(traj[0, 0], traj[0, 1], marker="*", label='start')
#         plt.plot(traj[:, 0], traj[:, 1], color='blue', label='Ground Truth', marker='.')
#         plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
#         plt.axis('scaled')
#         plt.title('Bayesian NN Prediction -- pos Space')
#         plt.legend()
#         plt.show()

