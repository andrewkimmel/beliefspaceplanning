
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy import signal
from sklearn.neighbors import KDTree #pip install -U scikit-learn
import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var


class transition_experience():
    path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    # path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'

    def __init__(self, Load=True, discrete = False, postfix=''):

        if discrete:
            self.mode = 'discrete'
        else:
            self.mode = 'cont'

        self.recorder_data = True if postfix == '_bu' else False
            
        # self.file = 'acrobot_raw_' + self.mode + '_v' + str(var.data_version_) + postfix
        self.file_name = '/home/pracsys/Dropbox/transfer/transition_data/Acrobot/noisy_acrobot_discrete_withObstacles/acrobot_data'
        # self.path + self.file # + '.obj

        if Load:
            self.load()
        else:
            self.clear()
       
    def add(self, state, action, next_state, done):
        self.memory += [(state, action, next_state, done)]
        
    def clear(self):
        self.memory = []

    def load(self):
        if os.path.isfile(self.file_name + '.txt'):
            print('Loading data from ' + self.file_name + '.txt')
            # with open(self.file_name, 'rb') as filehandler:
                # self.memory = pickle.load(filehandler)
            # D = np.concatenate((np.loadtxt(self.file_name  + '.txt', delimiter=','), np.loadtxt(self.file_name + '2' + '.txt', delimiter=',')), axis=0)
            D = np.loadtxt(self.file_name  + '.txt', delimiter=',')
            self.memory = []
            for d in D:
                state = np.array(d[:4])
                action = np.array(d[4])
                next_state = np.array(d[5:-1])
                done = d[-1]==0
                self.memory += [(state, action, next_state, done)]

            print('Loaded transition data of size %d.'%self.getSize())
        else:
            self.clear()

    def add_rollout_data(self):
        # Include rollout data in transitions DB
        with open(self.path +'rollout_tmp.pkl', 'rb') as filehandler:
        # with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/rollout_tmp.pkl', 'rb') as filehandler:
            roll_memory = pickle.load(filehandler)

        self.memory += roll_memory


    def getComponents(self):

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])

        return states, actions, next_states

    def save(self):
        print('Saving data...')
        file_pi = open(self.file_name, 'wb')
        pickle.dump(self.memory, file_pi)
        print('Saved transition data of size %d.'%self.getSize())
        file_pi.close()

    def getSize(self):
        return len(self.memory)

    def plot_data(self):

        states = np.array([item[0] for item in self.memory])
        done = [item[3] for item in self.memory]

        for s in states:
            for i in range(2):
                s[i] += 2*np.pi if s[i] < -np.pi else 0.0
                s[i] -= 2*np.pi if s[i] >  np.pi else 0.0

        # Start dist.
        St = []
        for i in range(states.shape[0]-1):
            if done[i] and not done[i+1]:
                St.append(states[i+1,:4])
        St = np.array(St)
        s_start = np.mean(St, 0)
        s_std = np.std(St, 0)
        print "start mean: ", s_start
        print "start std.: ", s_std

        print np.max(states, axis=0)
        print np.min(states, axis=0)

        failed_states = states[done]

        plt.figure(1)
        ax1 = plt.subplot(211)
        # ax1.set_aspect('equal')
        ax1.plot(states[:,0],states[:,2],'.k')
        # ax1.plot(states[:,0],states[:,1],'.y')
        ax1.plot(failed_states[:,0],failed_states[:,2],'.r')
        ax1.set(title='Link 1')
        plt.xlabel('Angle')
        plt.ylabel('Angular velocity')
        # plt.xlim(-100., 100.)
        # plt.ylim(40., 140.)
        
        ax2 = plt.subplot(212)
        ax2.plot(states[:,1],states[:,3],'.k')
        ax2.plot(failed_states[:,1],failed_states[:,3],'.r')
        ax2.set(title='Link 2')
        plt.xlabel('Angle')
        plt.ylabel('Angular velocity')
        
        plt.show()

    def save_to_file(self):

        filen = self.path + self.file + '.db'

        n = self.getSize()

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        inx = np.where(done)

        M = np.concatenate((states, actions, next_states), axis=1)
        M = np.delete(M, inx, 0)

        np.savetxt(filen, M, delimiter=' ')

    def process_transition_data(self, stepSize = 1, plot = False):
       
        def clean(D, done):
            print('[transition_experience] Cleaning data...')

            jj = range(6,8) 
            i = 0
            inx = []
            f = 2.5
            while i < D.shape[0]:
                if np.linalg.norm( D[i, 0:2] - D[i, jj] ) < 1.0 or done[i]:
                    inx.append(i)
                elif (D[i, 0] > f  and D[i, 6] < -f) or (D[i, 0] < -f  and D[i, 6] > f) or (D[i, 1] > f  and D[i, 7] < -f) or (D[i, 1] < -f  and D[i, 7] > f):
                    inx.append(i)
                i += 1
            return D[inx,:], done[inx]

        def multiStep(D, done, stepSize): 
            Dnew = []
            ia = range(4,5)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize, ia]) or np.any(done[i:i+stepSize]):
                    continue

                Dnew.append( np.concatenate((D[i,:ia[0]], a, D[i+stepSize-1, ia[-1]+1:]), axis=0) )

            return np.array(Dnew)

        print('[transition_experience] Saving transition data...')
        is_start = 1
        is_end = 277

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        for s in states:
            for i in range(2):
                s[i] += 2*np.pi if s[i] < -np.pi else 0.0
                s[i] -= 2*np.pi if s[i] >  np.pi else 0.0

        # For data from recorder
        if self.recorder_data:
            next_states = np.roll(states, -1, axis=0)
        else:
            for s in next_states:
                for i in range(2):
                    s[i] += 2*np.pi if s[i] < -np.pi else 0.0
                    s[i] -= 2*np.pi if s[i] >  np.pi else 0.0

        # states = states[:, [0, 1, 2, 3]]
        # next_states = next_states[:, [0, 1, 2, 3]]
        self.state_dim = states.shape[1]

        D = np.concatenate((states, actions.reshape(-1,1), next_states), axis = 1)
        # D, done = clean(D, done)

        inx = np.where(done)[0]
        D = np.delete(D, inx, 0)
        done = np.delete(done, inx, 0)

        if stepSize > 1:
            D = multiStep(D, done, stepSize)

        D = D[np.random.choice(D.shape[0], int(0.2*D.shape[0]), replace=False),:] # Dillute
        self.D = D

        savemat(self.path + 'acrobot_data_' + self.mode + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.mat', {'D': D, 'is_start': is_start, 'is_end': is_end})
        print "Saved mat file with " + str(D.shape[0]) + " transition points."

        if plot:
            ax1 = plt.subplot(121)
            plt.scatter(D[:,0], D[:,2])
            ax2 = plt.subplot(122)
            plt.scatter(D[:,1], D[:,3])
            plt.show()
    
    def process_svm(self, stepSize = 1):

        def multiStep(D, done, stepSize): 
            Dnew = []
            done_new = []
            ia = range(4,5)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize, ia]):
                    continue
                
                if np.any(done[i:i+stepSize]):
                    done_new.append(True)
                else:
                    done_new.append(False)

                Dnew.append( np.concatenate((D[i,:ia[0]], a), axis=0) )

            return np.array(Dnew), np.array(done_new)

        from sklearn import svm
        from sklearn.preprocessing import StandardScaler

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        for s in states:
            for i in range(2):
                s[i] += 2*np.pi if s[i] < -np.pi else 0.0
                s[i] -= 2*np.pi if s[i] >  np.pi else 0.0

        SA = np.concatenate((states, actions.reshape(-1,1)), axis=1)
        if stepSize > 1:
            SA, done = multiStep(SA, done, stepSize)
        print('Transition data with steps size %d has now %d points'%(stepSize, SA.shape[0]))

        inx_fail = np.where(done)[0]
        inx_fail = inx_fail[np.random.choice(inx_fail.shape[0], 15000, replace=False)]
        print "Number of failed states " + str(inx_fail.shape[0])
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], inx_fail.shape[0], replace=False)]
        SA = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)

        with open(self.path + 'acrobot_svm_data_' + self.mode + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.obj', 'wb') as f: 
            pickle.dump([SA, done], f)
        print('Saved svm data.')

        ##  Test data
        # Normalize
        scaler = StandardScaler()
        SA = scaler.fit_transform(SA)
        x_mean = scaler.mean_
        x_std = scaler.scale_

        # Test data
        ni = 40
        T = np.where(done)[0]
        inx_fail = T[np.random.choice(T.shape[0], ni, replace=False)]
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], ni, replace=False)]
        SA_test = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done_test = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)

        SA = np.delete(SA, inx_fail, axis=0)
        SA = np.delete(SA, inx_suc, axis=0)
        done = np.delete(done, inx_fail, axis=0)
        done = np.delete(done, inx_suc, axis=0)

        print 'Fitting SVM...'
        clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
        clf.fit( list(SA), 1*done )
        print 'SVM fit with %d classes: '%len(clf.classes_) + str(clf.classes_)

        s = 0
        s_suc = 0; c_suc = 0
        s_fail = 0; c_fail = 0
        for i in range(SA_test.shape[0]):
            p = clf.predict_proba(SA_test[i].reshape(1,-1))[0]
            fail = p[1]>0.5
            # print p, done_test[i], fail
            s += 1 if fail == done_test[i] else 0
            if done_test[i]:
                c_fail += 1
                s_fail += 1 if fail else 0
            else:
                c_suc += 1
                s_suc += 1 if not fail else 0
        print 'Success rate: ' + str(float(s)/SA_test.shape[0]*100)
        print 'Drop prediction accuracy: ' + str(float(s_fail)/c_fail*100)
        print 'Success prediction accuracy: ' + str(float(s_suc)/c_suc*100)





