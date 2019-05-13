
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

recorder_data = False

class transition_experience():
    path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    # path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'

    def __init__(self, Load=True, discrete = False, postfix=''):

        if discrete:
            self.mode = 'discrete'
        else:
            self.mode = 'cont'
        
        self.file = 'sim_raw_' + self.mode + '_v' + str(var.data_version_) + postfix
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

        if var.data_version_ == 13 and len(self.memory) < 5e5: # v13d only recorder position and load
            states = np.array([item[0] for item in self.memory])
            next_states = np.array([item[2] for item in self.memory])
        else:
            states = np.array([item[0] for item in self.memory])
        done = [item[3] for item in self.memory]

        # Start dist.
        St = []
        for i in range(states.shape[0]-1):
            if done[i] and not done[i+1]:
                St.append(states[i+1,:])
        St = np.array(St)
        s_start = np.mean(St, 0)
        s_std = np.std(St, 0)
        print "start mean: ", s_start
        print "start std.: ", s_std

        print np.max(states, axis=0)
        print np.min(states, axis=0)

        # For data from recorder
        if recorder_data:
            states = states[:, [0, 1, 2, 3, 4, 5]]
            # states[:,:2] *= 1000 # m to mm

        failed_states = states[done]

        # Obs1 = np.array([42, 90, 12.])
        # Obs2 = np.array([-45, 101, 7.])
        Obs1 = np.array([33, 110, 4.]) # Right
        Obs2 = np.array([-27, 118, 2.5]) # Left

        plt.figure(1)
        ax1 = plt.subplot(211)
        ax1.set_aspect('equal')
        #ax1.plot(states[:,0],states[:,1],'-k')
        ax1.plot(states[:,0],states[:,1],'.y')
        ax1.plot(failed_states[:,0],failed_states[:,1],'.r')
        ax1.set(title='Object position')
        obs1 = plt.Circle((Obs1[0], Obs1[1]), 1.5*Obs1[2], zorder=9, color='g')
        ax1.add_artist(obs1)
        obs1 = plt.Circle((Obs1[0], Obs1[1]), Obs1[2], zorder=10, color='b')
        ax1.add_artist(obs1)
        obs2 = plt.Circle((Obs2[0], Obs2[1]), 1.5*Obs2[2], zorder=9, color='g')
        ax1.add_artist(obs2)
        obs2 = plt.Circle((Obs2[0], Obs2[1]), Obs2[2], zorder=10, color='b')
        ax1.add_artist(obs2)
        plt.xlim(-100., 100.)
        plt.ylim(40., 140.)
        
        ax2 = plt.subplot(212)
        ax2.plot(states[:,2],states[:,3],'.k')
        ax2.plot(failed_states[:,2],failed_states[:,3],'.r')
        ax2.set(title='Actuator loads')

        # ax3 = plt.subplot(313)
        # ax3.plot(states[:,4],states[:,5],'.k')
        # ax3.plot(failed_states[:,4],failed_states[:,5],'.r')
        # ax3.set(title='Object velocity')
        
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

    def process_transition_data(self, mode = 1, stepSize = 1, plot = False):
        '''
        mode:
            1 - Position and load
            2 - Position
            3 - Position, load and velocity
            4 - Position, load and joints or velocities(4)
            5 - Position and joints
        '''
        def clean(D, done, mode):
            print('[transition_experience] Cleaning data...')

            if mode == 1:
                jj = range(6,8) 
            elif mode == 2:
                jj = range(4,6)
            elif mode == 3:
                jj = range(8,10)
            elif mode == 4:
                jj = range(10, 12)
            elif mode == 5:
                jj = range(8, 10)
            i = 0
            inx = []
            while i < D.shape[0]:
                if np.linalg.norm( D[i, 0:2] - D[i, jj] ) < 3 or done[i]:
                    inx.append(i)
                i += 1
            return D[inx,:], done[inx]

        def uniform_actions(D):
            print('[transition_experience] Applying action uniformity to the data...')
            DA = D[:,8:10]
            A, Ix, N = np.unique(DA, axis=0, return_inverse=True, return_counts=True)

            n = np.minimum(np.min(N), 65000)

            P = []
            for i in range(A.shape[0]):
                ix = np.where(Ix == i)[0]

                Da = D[ix, :]
                Da = Da[np.random.choice(Da.shape[0], n, replace=False), :] # Dillute to minimal number of actions

                P.append(Da)

            Dnew = np.concatenate((P[0], P[1]), axis=0)
            for i in range(2, len(P)):
                Dnew = np.concatenate((Dnew, P[i]), axis=0)

            # Dnew = np.random.shuffle(Dnew)
            print('[transition_experience] Applyed action uniformity to the data, now with size %d.'%Dnew.shape[0])

            return Dnew

        def multiStep(D, done, stepSize, mode): 
            Dnew = []
            ia = range(4,6) if mode == 1  else range(8,10) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize, ia]) or np.any(done[i:i+stepSize]):
                    continue

                Dnew.append( np.concatenate((D[i,:ia[0]], a, D[i+stepSize-1, ia[-1]+1:]), axis=0) )

            return np.array(Dnew)

        print('[transition_experience] Saving transition data...')
        is_start = 1
        is_end = 277

        if var.data_version_ == 13 and len(self.memory) < 5e5: # v13d only recorder position and load
            states = np.array([item[0] for item in self.memory])
            next_states = np.array([item[2] for item in self.memory])
        else:
            states = np.array([item[0] for item in self.memory])
            next_states = np.array([item[2] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        # For data from recorder
        if recorder_data:
            next_states = np.roll(states, -1, axis=0)

        for i in range(done.shape[0]):
            if done[i]:
                done[i-2:i] = True

        if mode == 1:
            states = states[:, [0, 1, 2, 3]]
            next_states = next_states[:, [0, 1, 2, 3]]
        elif mode == 2:
            states = states[:, [0, 1]]
            next_states = next_states[:, [0, 1]]
        elif mode == 3:
            states = np.concatenate((states[:, :4], signal.medfilt(states[:, 4], kernel_size = 21).reshape((-1,1)), signal.medfilt(states[:, 5], kernel_size = 21).reshape((-1,1))), axis=1)
            next_states = np.concatenate((next_states[:, :4], signal.medfilt(next_states[:, 4], kernel_size = 21).reshape((-1,1)), signal.medfilt(next_states[:, 5], kernel_size = 21).reshape((-1,1))), axis=1)
        elif mode == 4:
            states = states#[:, [0, 1, 2, 3, 6, 7, 8, 9]]
            next_states = next_states#[:, [0, 1, 2, 3, 6, 7, 8, 9]]
        elif mode == 5:
            states = states[:, [0, 1, 6, 7, 8, 9]]
            next_states = next_states[:, [0, 1, 6, 7, 8, 9]]
        self.state_dim = states.shape[1]

        D = np.concatenate((states, actions, next_states), axis = 1)
        D, done = clean(D, done, mode)

        inx = np.where(done)[0]
        D = np.delete(D, inx, 0)
        done = np.delete(done, inx, 0)

        if stepSize > 1:
            D = multiStep(D, done, stepSize, mode)

        # D = D[np.random.choice(D.shape[0], int(0.6*D.shape[0]), replace=False),:] # Dillute
        # D = uniform_actions(D)
        self.D = D

        savemat(self.path + 'sim_data_discrete_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '_J.mat', {'D': D, 'is_start': is_start, 'is_end': is_end})
        print "Saved mat file with " + str(D.shape[0]) + " transition points."

        if plot:
            plt.scatter(D[:,0], D[:,1])
            plt.show()
    
    def process_svm(self, mode = 1, stepSize = 1):

        def multiStep(D, done, stepSize, mode): 
            Dnew = []
            done_new = []
            ia = range(4,6) if mode == 1 else range(8,10)
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

        if var.data_version_ == 13 and len(self.memory) < 5e5: 
            states = np.array([item[0] for item in self.memory])
        else:
            states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        # For data from recorder

        if mode == 1:
            states = states[:, [0, 1, 2, 3]]
        elif mode == 2:
            states = states[:, [0, 1]]
        elif mode == 4:
            states = states#[:, [0, 1, 2, 3, 6, 7, 8, 9]]
        elif mode == 5:
            states = states[:, [0, 1, 6, 7, 8, 9]]

        for i in range(done.shape[0]):
            if done[i]:
                done[i-2:i] = True

        SA = np.concatenate((states, actions), axis=1)
        if stepSize > 1:
            SA, done = multiStep(SA, done, stepSize, mode)
        print('Transition data with steps size %d has now %d points'%(stepSize, SA.shape[0]))

        inx_fail = np.where(done)[0]
        print "Number of failed states " + str(inx_fail.shape[0])
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], inx_fail.shape[0], replace=False)]
        SA = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)

        # exit(1) 

        with open(self.path + 'svm_data_' + self.mode + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.obj', 'wb') as f: 
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





