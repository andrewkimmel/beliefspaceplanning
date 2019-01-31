
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy import signal
from sklearn.neighbors import KDTree #pip install -U scikit-learn


class transition_experience():
    path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    # path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'

    def __init__(self, Load=True, discrete = False):

        if discrete:
            self.mode = 'discrete'
        else:
            self.mode = 'cont'
        
        self.file = 'sim_raw_' + self.mode + '_v5'
        self.file_name = self.path + self.file + '.obj'

        if Load:
            self.load()
        else:
            self.clear()
        
    def add(self, state, action, next_state, done):
        self.memory += [(state, action, next_state, done)]
        
    def clear(self):
        self.memory = []

    def load(self):
        if os.path.isfile(self.file_name):
            print('Loading data from ' + self.file_name)
            with open(self.file_name, 'rb') as filehandler:
            # filehandler = open(self.file_name, 'r')
                self.memory = pickle.load(filehandler)
            print('Loaded transition data of size %d.'%self.getSize())
        else:
            self.clear()

    def add_rollout_data(self):
        # Include rollout data in transitions DB
        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/rollout_tmp.pkl', 'rb') as filehandler:
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

        states = [item[0] for item in self.memory]
        done = [item[3] for item in self.memory]
        states = np.array(states)
        failed_states = states[done]

        plt.figure(1)
        ax1 = plt.subplot(121)
        #ax1.plot(states[:,0],states[:,1],'-k')
        ax1.plot(states[:,0],states[:,1],'.y')
        #ax1.plot(failed_states[:,0],failed_states[:,1],'.r')
        ax1.set(title='Object position')
        plt.xlim(-100., 100.)
        plt.ylim(0., 150.)
        
        ax2 = plt.subplot(122)
        ax2.plot(states[:,2],states[:,3],'.k')
        ax2.plot(failed_states[:,2],failed_states[:,3],'.r')
        ax2.set(title='Actuator loads')
        
        # ax3 = plt.subplot(223)
        # ax3.plot(ep_mean_r)
        # ax3.set(title='Mean Episode Rewards')

        # ax4 = plt.subplot(224)
        # ax4.plot(c_losses)
        # ax4.set(title='Q-value losses')

        plt.show()

    def save_to_file(self):

        filen = self.path + self.file + '.db'

        n = self.getSize()

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        inx = np.where(done)

        # for i in range(len(done)):
        #     if done[i]:
        #         next_states[i] = np.array([-1000.,-1000.,-1000.,-1000.])

        M = np.concatenate((states, actions, next_states), axis=1)
        M = np.delete(M, inx, 0)

        np.savetxt(filen, M, delimiter=' ')

    def process_transition_data(self, mode = 1, stepSize = 1, plot = False):
        '''
        mode:
            1 - Position and load
            2 - Position and velocity
            3 - Postion, load and velocity
        '''
        def clean(D, done, mode):
            print('[transition_experience] Cleaning data...')

            jj = range(6,8) if mode == 1 or mode == 2 else range(8,10)
            i = 0
            inx = []
            while i < D.shape[0]:
                if np.linalg.norm( D[i, 0:2] - D[i, jj] ) < 3:
                    inx.append(i)
                i += 1
            return D[inx,:], done[inx]

        def multiStep(D, done, stepSize, mode): 
            Dnew = []
            ia = range(4,6) if mode == 1 or mode == 2 else range(6,8)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize+1, ia]) or np.any(done[i:i+stepSize+1]):
                    continue

                Dnew.append( np.concatenate((D[i,:ia[0]], a, D[i+stepSize, ia[-1]+1:]), axis=0) )

            return np.array(Dnew)

        print('[transition_experience] Saving transition data...')
        is_start = 1
        is_end = 277

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        if mode == 1:
            states = states[:, [0, 1, 2, 3]]
            next_states = next_states[:, [0, 1, 2, 3]]
        elif mode == 2:
            states = states[:, [0, 1, 4, 5]]
            next_states = next_states[:, [0, 1, 4, 5]]
        elif mode == 3:
            states = np.concatenate((states[:, :4], signal.medfilt(states[:, 4], kernel_size = 21).reshape((-1,1)), signal.medfilt(states[:, 5], kernel_size = 21).reshape((-1,1))), axis=1)
            next_states = np.concatenate((next_states[:, :4], signal.medfilt(next_states[:, 4], kernel_size = 21).reshape((-1,1)), signal.medfilt(next_states[:, 5], kernel_size = 21).reshape((-1,1))), axis=1)
        self.state_dim = states.shape[1]

        inx = np.where(done)
        D = np.concatenate((states, actions, next_states), axis = 1)
        D = np.delete(D, inx, 0)
        done = np.delete(done, inx, 0)

        D, done = clean(D, done, mode)

        if stepSize > 1:
            D = multiStep(D, done, stepSize, mode)

        self.D = D
        Dreduced = []

        savemat(self.path + 'sim_data_discrete_v5_d' + str(6 if mode == 3 else 4) + '_m' + str(stepSize) + '.mat', {'D': D, 'Dreduced': Dreduced, 'is_start': is_start, 'is_end': is_end})
        print "Saved mat file with " + str(D.shape[0]) + " transition points."

        if plot:
            plt.scatter(D[:,0], D[:,1])
            plt.show()
    
    def process_svm(self, mode = 1, stepSize = 1):

        def multiStep(D, done, stepSize, mode): 
            Dnew = []
            done_new = []
            ia = range(4,6) if mode == 1 or mode == 2 else range(6,8)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize+1, ia]):
                    continue
                
                if np.any(done[i:i+stepSize+1]):
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

        SA = np.concatenate((states, actions), axis=1)
        SA, done = multiStep(SA, done, stepSize, mode)
        print('Transition data with steps size %d has now %d points'%(stepSize, SA.shape[0]))

        inx_fail = np.where(done)[0]
        print "Number of failed states " + str(inx_fail.shape[0])
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], 1000, replace=False)]
        SA = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)

        with open(self.path + 'svm_data_' + self.mode + '_v5_d' + str(6 if mode == 3 else 4) + '_m' + str(stepSize) + '.obj', 'wb') as f: 
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





