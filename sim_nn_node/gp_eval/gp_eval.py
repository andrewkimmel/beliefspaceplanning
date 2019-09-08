#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from gpup_gp_node.srv import one_transition
from sim_nn_node.srv import load_model, critic
from sklearn.neighbors import KDTree

o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
lm_srv = rospy.ServiceProxy('/nn/load_model', load_model)
critic_srv = rospy.ServiceProxy('/nn/critic', critic)
rospy.init_node('gp_eval', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

R = [0.5, 0.6, 0.7, 0.8, 0.9]
# R = [0.8, 0.9]

with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
    D = pickle.load(f)

# F = []
# for d in D:
#     for g in d:
#         F.append(g[2:4])
# F = np.array(F)
# plt.plot(F[:,0], F[:,1], '.')
# plt.axis('equal')

# for d in D:
# # d = D[400]
#     if np.any(d[:,2:4] > 35):
#         plt.figure(1)
#         plt.plot(d[:,0],d[:,1])

#         plt.figure(2)
#         plt.plot(d[:,2],d[:,3])

#         plt.show()
# exit(1)

if 0:
    for ratio in R:
        lm_srv(ratio)

        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
            D = pickle.load(f)
        r = 577#int((1-ratio)*len(D)) Always use the last 20%
        del D[:r] # Delete data that was used for training
            
        try:
            with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/error_points_r' + str(ratio) + '_v1.pkl', 'rb') as f: 
                O, M, E = pickle.load(f)
                O = list(O)
                E = list(E)
                M = list(M)
        except:
            O = []
            M = []
            E = []
        N = 1000000
        for k in range(len(O), N):
            ix = np.random.randint(len(D))
            l = np.random.randint(10,30)
            jx = np.random.randint(D[ix].shape[0]-l)

            h = 1
            while h < l and np.all(D[ix][jx, 4:6] == D[ix][jx + h, 4:6]):
                h += 1
            if h < 10:
                continue
            l = np.minimum(h, l)

            S = D[ix][jx:jx+l,:4]
            A = D[ix][jx:jx+l,4:6]
            S_next = D[ix][jx:jx+l,6:]
        
            Sp = []
            state = S[0]
            Sp.append(state)
            i = 0
            for a in A:
                state = o_srv(state.reshape(-1,1), a.reshape(-1,1)).next_state
                state = np.array(state)
                Sp.append(state)
            Sp = np.array(Sp)
            e = tracking_error(S, Sp)

            o = np.concatenate((S[0], A[0]), axis = 0)
            O.append(o)
            M.append(l)
            E.append(e)

            print ratio, k, len(E), A[0], l, e

            if k > 1 and not k % 2000:
                O1 = np.array(O)
                M1 = np.array(M)
                E1 = np.array(E)

                with open(path + 'error_points_r' + str(ratio) + '_v1.pkl', 'wb') as f: 
                    pickle.dump([O1, M1, E1], f)
    else:
        with open(path + 'error_points_r' + str(ratio) + '_v1.pkl', 'r') as f: 
            O, L, E = pickle.load(f)

# exit(1)


        
# Create data files
if 0:
    for ratio in R:
        with open(path + 'error_points_r' + str(ratio) + '_v1.pkl', 'r') as f: 
            O, L, E = pickle.load(f)
        print "Data of size %d loaded."%O.shape[0]

        X = np.concatenate((O[:,:6], L.reshape(-1,1)), axis = 1)
        with open(path + 'data_r' + str(ratio) + '_v1.pkl', 'wb') as f: 
            pickle.dump([X, E], f)

        import warnings
        warnings.filterwarnings("ignore")

        kdt = KDTree(X, leaf_size=100, metric='euclidean')
        with open(path + 'kdt_r' + str(ratio) + '_v1.pkl', 'wb') as f: 
            pickle.dump(kdt, f)
    exit(1) # disable this and restart the nn_node with critic

# Evaluate
if 1:
    if 0:
        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
            D = pickle.load(f)
        D = D[360:577]
        np.random.seed(100)
        # I = np.random.randint(len(D), size=200)
        # D = [D[j] for j in I]

        H = []
        jj = 0
        for d in D:
            for _ in range(100):
                print str(jj/(len(D)*100.)*100) + '%'
                jj += 1
                l = np.random.randint(10,30)
                jx = np.random.randint(d.shape[0]-l)

                h = 1
                while h < l and np.all(d[jx, 4:6] == d[jx + h, 4:6]):
                    h += 1
                if h < 10:
                    continue
                l = np.minimum(h, l)

                S = d[jx:jx+l,:4]
                A = d[jx:jx+l,4:6]
                S_next = d[jx:jx+l,6:]
            
                Sp = []
                state = S[0]
                Sp.append(state)
                i = 0
                for a in A:
                    state = o_srv(state.reshape(-1,1), a.reshape(-1,1)).next_state
                    state = np.array(state)
                    Sp.append(state)
                Sp = np.array(Sp)
                e = tracking_error(S, Sp)

                H += [(S, A, l, e)]

        F = np.zeros((len(H), len(R)))
        for i in range(len(R)):
            print 'Ratio: %f'%R[i]
            lm_srv(R[i])

            for j in range(len(H)):
                h = H[j]
                state = h[0][0]
                action = h[1][0]
                l = h[2]
                e = h[3]

                F[j,i] = np.abs(e - critic_srv(state, action, l).err) # e_predict
        with open(path + 'results_rAll_v1.pkl', 'wb') as f: 
            pickle.dump([H, F], f)
    else:
        with open(path + 'results_rAll_v1.pkl', 'rb') as f: 
            H, F = pickle.load(f)

    f = np.mean(F, axis=0)
    f = np.flipud(f)
    R = 1.0 - np.flipud(R)
    plt.plot(R, f)
    plt.show()

    
            

