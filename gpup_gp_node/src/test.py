import numpy as np
import var
import pickle

# np.random.seed(10)

state_dim = var.state_dim_
tr = '3'
stepSize = var.stepSize_

A = np.concatenate( (np.array([[-1., -1.] for _ in range(int(150*1./stepSize))]), 
                np.array([[-1.,  1.] for _ in range(int(100*1./stepSize))]), 
                np.array([[ 1.,  0.] for _ in range(int(100*1./stepSize))]), 
                np.array([[ 1., -1.] for _ in range(int(70*1./stepSize))]),
                np.array([[-1.,  1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

f = path + 'ver_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize)
with open(f + '.pkl') as f:  
    Pro = pickle.load(f) 


Smean = []
Sstd = []
for i in range(A.shape[0]+1):
    F = []
    for j in range(len(Pro)): 
        if Pro[j].shape[0] > i:
            F.append(Pro[j][i])
    Smean.append( np.mean(np.array(F), axis=0) )
    Sstd.append( np.std(np.array(F), axis=0) )
Smean = np.array(Smean)
Sstd = np.array(Sstd)

from gp import GaussianProcess
from data_load import data_load
DL = data_load(simORreal = 'sim', discreteORcont = 'discrete', K = 100)

E1 = []
E2 = []
V = []
Kk = range(5, 1000, 10)
for K in Kk: 
    e1 = []
    e2 = []
    v = []
    for i in range(40):
            print K, i

            s = np.copy(Smean[i,:])
            a = A[i,:]
            sa = np.concatenate((s, a), axis=0)
            sa = DL.normz( sa ) 
            Theta, _ = DL.get_theta(sa)

            K1 = 1000
            idx = DL.kdt.query(np.copy(sa).reshape(1,-1), k = K1, return_distance=False)[0]
            idx = idx[np.random.choice(len(idx), K, replace=False)]  
            X_nn = DL.Xtrain[idx,:].reshape(K, DL.state_action_dim)
            Y_nn = DL.Ytrain[idx,:].reshape(K, DL.state_dim)
            ds_next = np.zeros((DL.state_dim,))
            std_next_normz = np.zeros((DL.state_dim,))
            for i in range(DL.state_dim):
                gp_est = GaussianProcess(X_nn[:,:DL.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i])
                mm, vv = gp_est.predict(sa[:DL.state_action_dim])
                ds_next[i] = mm
                std_next_normz[i] = np.sqrt(vv)
            sa_normz = sa[:DL.state_dim] + ds_next
            s_next = DL.denormz( sa_normz )
            std_next = DL.denormz_change( std_next_normz )

            # print s_next#, std_next
            # print Smean[21,:]

            e1.append(np.linalg.norm(s_next-Smean[21,:]))
            e2.append(np.linalg.norm(s_next[:2]-Smean[21,:2]))
            v.append(std_next)

        # print np.linalg.norm(s_next-Smean[21,:]), np.linalg.norm(s_next[:2]-Smean[21,:2])
    E1.append(np.mean(np.array(e1)))
    E2.append(np.mean(np.array(e2)))

    v = np.array(v)
    V.append(np.mean(v, axis=0))

with open(path + 'test_den.pkl', 'w') as f: 
    pickle.dump([Kk, E1, E2, V], f)

import matplotlib.pyplot as plt

plt.figure(1)
ax1 = plt.subplot(1,2,1)
plt.plot(Kk, E1, 'b')
plt.plot(Kk, E2, 'k')
plt.xlabel('NN')
plt.ylabel('Error')

ax1 = plt.subplot(1,2,2)
plt.plot(Kk, V)

plt.title('Density')
plt.show()