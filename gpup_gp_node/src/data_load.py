
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KDTree 
import os.path
import pickle
import matplotlib.pyplot as plt
import var

class data_load(object):
    # Dillute = 100000

    def __init__(self, simORreal = 'sim', discreteORcont = 'discrete', K = 100, K_manifold=-1, sigma=-1, dim=-1, Dillute = var.N_dillute_, dr = 'diff'):
        
        self.Dillute = Dillute
        self.postfix = '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(var.stepSize_)
        self.file = simORreal + '_data_' + discreteORcont + self.postfix + '.mat'
        self.path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
        # self.path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'
        self.load()
        self.dr = dr

        if os.path.exists(self.path + 'opt_data_discrete' + self.postfix + '.obj'):
            with open(self.path + 'opt_data_discrete' + self.postfix + '.obj', 'rb') as f: 
                _, self.theta_opt, self.opt_kdt = pickle.load(f)
            print('[data_load] Loaded hyper-parameters data from ' + self.file)
        else:
            self.precompute_hyperp(K, K_manifold, sigma, dim)

    def load(self):

        print('[data_load] Loading data from "' + self.file + '"...' )
        Q = loadmat(self.path + self.file)
        Qtrain = Q['D']
        # plt.plot(Qtrain[:,0],Qtrain[:,1],'.')
        # plt.show()

        is_start = 470# 30532 #1540#int(Q['is_start'])#100080
        # while is_start < 200000:
        #     if np.all(Qtrain[is_start, 2:4] == np.array([16., 16.])):
        #         break
        #     is_start += 1
        # print is_start
        is_end = is_start + 350#int(Q['is_end'])
        self.Qtest = Qtrain[is_start:is_end, :]
        Qtrain = np.delete(Qtrain, range(is_start, is_end), 0)

        # plt.plot(self.Qtest[:,0], self.Qtest[:,1],'.-k')
        # plt.plot(self.Qtest[0,0], self.Qtest[0,1],'o')
        # plt.show()
        # exit(1)

        if 'Dreduced' in Q:
            self.Xreduced = Q['Dreduced']

        if self.Dillute > 0:
            Qtrain = Qtrain[np.random.choice(Qtrain.shape[0], self.Dillute, replace=False),:] # Dillute
        print('[data_load] Loaded training data of ' + str(Qtrain.shape[0]) + '.')

        self.state_action_dim = var.state_action_dim_
        self.state_dim = var.state_dim_

        self.Xtrain = Qtrain[:,:self.state_action_dim]
        self.Ytrain = Qtrain[:,self.state_action_dim:]

        # Normalize
        self.x_max_X = np.max(self.Xtrain, axis=0)
        self.x_min_X = np.min(self.Xtrain, axis=0)
        self.x_max_Y = np.max(self.Ytrain, axis=0)
        self.x_min_Y = np.min(self.Ytrain, axis=0)

        for i in range(self.state_dim):
            tmp = np.max([self.x_max_X[i], self.x_max_Y[i]])
            self.x_max_X[i] = tmp
            self.x_max_Y[i] = tmp
            tmp = np.min([self.x_min_X[i], self.x_min_Y[i]])
            self.x_min_X[i] = tmp
            self.x_min_Y[i] = tmp

        for i in range(self.state_action_dim):
            self.Xtrain[:,i] = (self.Xtrain[:,i]-self.x_min_X[i])/(self.x_max_X[i]-self.x_min_X[i])
        for i in range(self.state_dim):
            self.Ytrain[:,i] = (self.Ytrain[:,i]-self.x_min_Y[i])/(self.x_max_Y[i]-self.x_min_Y[i])

        self.Ytrain -= self.Xtrain[:,:self.state_dim] # The target set is the state change

        print('[data_load] Loading data to kd-tree...')
        if 0 and os.path.exists(self.path + 'kdtree' + self.postfix + '.obj'):
            with open(self.path + 'kdtree' + self.postfix + '.obj', 'rb') as f: 
                self.kdt = pickle.load(f)
        else:
            self.kdt = KDTree(self.Xtrain, leaf_size=100, metric='euclidean')
            with open(self.path + 'kdtree' + self.postfix + '.obj', 'wb') as f:
                pickle.dump(self.kdt, f)
        print('[data_load] kd-tree ready.')

    def normz(self, x):
        d = len(x)
        return (x-self.x_min_X[:d])/(self.x_max_X[:d]-self.x_min_X[:d])

    def denormz(self, x):
        d = len(x)
        return  x*(self.x_max_X[:d]-self.x_min_X[:d])+self.x_min_X[:d]

    def normz_change(self, dx):
        d = len(dx)
        return dx/(self.x_max_X[:d]-self.x_min_X[:d])

    def denormz_change(self, dx):
        d = len(dx)
        return  dx*(self.x_max_X[:d]-self.x_min_X[:d])

    def normz_batch(self, X):
        d = X.shape[1]
        for i in range(d):
            X[:,i] = (X[:,i]-self.x_min_X[i])/(self.x_max_X[i] - self.x_min_X[i])
        return X

    def denormz_batch(self, X):
        d = X.shape[1]
        for i in range(d):
            X[:,i] = X[:,i]*(self.x_max_X[i]-self.x_min_X[i]) + self.x_min_X[i]
        return X

    def precompute_hyperp(self, K = 100, K_manifold=-1, sigma=-1, dim=-1):
        print('[data_load] Pre-computing GP hyper-parameters data.')

        if K_manifold > 0:
            if self.dr == 'diff':
                from dr_diffusionmaps import DiffusionMap
                DR = DiffusionMap(sigma=sigma, embedding_dim=dim)
            elif self.dr == 'spec':
                from spectralEmbed import spectralEmbed
                DR = spectralEmbed(embedding_dim=dim)

        def reduction(sa, X, Y, K_manifold):
            inx = DR.ReducedClosestSetIndices(sa, X, k_manifold = K_manifold)

            return X[inx,:][0], Y[inx,:][0]

        from gp import GaussianProcess
        import pickle

        SA_opt = []
        theta_opt = []
        N = 1000
        for i in range(N):
            print('[data_load] Computing hyper-parameters for data point %d out of %d.'% (i, N))
            sa = self.Xtrain[np.random.randint(self.Xtrain.shape[0]), :]

            idx = self.kdt.query(sa.reshape(1,-1), k = K, return_distance=False)
            X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
            Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

            if K_manifold > 0:
                X_nn, Y_nn = reduction(sa, X_nn, Y_nn, K_manifold)

            gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,0], optimize = True, theta=None) # Optimize to get hyper-parameters
            theta = gp_est.cov.theta

            SA_opt.append(sa)
            theta_opt.append(theta)

        self.SA_opt = np.array(SA_opt)
        self.theta_opt = np.array(theta_opt)

        self.opt_kdt = KDTree(SA_opt, leaf_size=20, metric='euclidean')

        with open(self.path + 'opt_data_discrete' + self.postfix + '.obj', 'wb') as f: 
            pickle.dump([self.SA_opt, self.theta_opt, self.opt_kdt], f)
        print('[data_load] Saved hyper-parameters data.')

    def get_theta(self, sa):
        idx = self.opt_kdt.query(sa.reshape(1,-1), k = 1, return_distance=False)        

        return self.theta_opt[idx,:].reshape((-1,))



