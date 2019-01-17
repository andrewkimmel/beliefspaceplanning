
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KDTree 
import os.path
import pickle

N_dillute = 300000 # Number of points to randomly select from data

class data_load(object):

    def __init__(self, simORreal = 'sim', discreteORcont = 'discrete', K = 100):
        
        self.postfix = '_v5_d6_m10'
        self.file = simORreal + '_data_' + discreteORcont + self.postfix + '.mat'
        self.path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
        # self.path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'
        self.load()
        self.K = K

        if os.path.exists(self.path + 'opt_data_discrete' + self.postfix + '.obj'):
            with open(self.path + 'opt_data_discrete' + self.postfix + '.obj', 'rb') as f: 
                _, self.theta_opt, self.opt_kdt = pickle.load(f)
            print('[data_load] Loaded hyper-parameters data.')
        else:
            self.precompute_hyperp()

    def load(self):

        print('[data_load] Loading data from "' + self.file + '"...' )
        Q = loadmat(self.path + self.file)
        Qtrain = Q['D']

        if 'Dreduced' in Q:
            self.Xreduced = Q['Dreduced']

        Qtrain = Qtrain[np.random.choice(Qtrain.shape[0], N_dillute, replace=False),:] # Dillute
        print('[data_load] Loaded training data of ' + str(Qtrain.shape[0]) + '.')

        self.state_action_dim = 6+2
        self.state_dim = 4+2

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
        self.kdt = KDTree(self.Xtrain, leaf_size=20, metric='euclidean')
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

    def precompute_hyperp(self):
        print('[data_load] Pre-computing GP hyper-parameters data.')

        from gp import GaussianProcess
        import pickle

        SA_opt = []
        theta_opt = []
        N = 10000
        for i in range(N):
            print('[data_load] Computing hyper-parameters for data point %d out of %d.'% (i, N))
            sa = self.Xtrain[np.random.randint(self.Xtrain.shape[0]), :]

            idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
            X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
            Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

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



