
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KDTree #pip install -U scikit-learn

N_dillute = 30000 # Number of points to randomly select from data

class data_load(object):

    def __init__(self, simORreal = 'sim', discreteORcont = 'discrete'):
        
        self.file = simORreal + '_data_' + discreteORcont + '_v5_d6_m10.mat'
        self.load()

    def load(self):

        print('[data_load] Loading data from "' + self.file + '"...' )
        Q = loadmat('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/' + self.file)
        Qtrain = Q['D']

        if 'Dreduced' in Q:
            self.Xreduced = Q['Dreduced']

        # Qtrain = Qtrain[np.random.choice(Qtrain.shape[0], N_dillute, replace=False),:] # Dillute
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


