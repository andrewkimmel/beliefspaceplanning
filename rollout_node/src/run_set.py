#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_node.srv import rolloutReq
import time
import glob

rollout = 1

# path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'
# path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

# comp = 'juntao/catkin_ws/'
# comp = 'pracsys/catkin_ws/'
comp = 'akimmel/repositories/pracsys/'
Set = '6'

K = 20 # Number of rollouts

# set_modes = ['robust_particles_pc_svmHeuristic', 'robust_particles_pc']
set_modes = ['robust_particles_pc', 'mean_only', 'naive_with_svm']
# set_modes = ['robust_particles_pc_svmHeuristic', 'mean_only', 'naive_with_svm']

############################# Rollout ################################
if rollout:
    rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
    rospy.init_node('run_rollout_set', anonymous=True)
    state_dim = 2
    
    for set_mode in set_modes:
        path = '/home/' + comp + 'src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

        files = glob.glob(path + set_mode + "*plan.txt")

        for i in range(len(files)):

            action_file = files[i]
            if action_file.find('traj') > 0:
                continue
            pklfile = action_file[:-3] + 'pkl'

            print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.')

            A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

            Af = A.reshape((-1,))
            Pro = []
            for j in range(K):
                print("Rollout number " + str(j) + ".")
                
                Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

                Pro.append(Sro)

                if not (j % 2):
                    with open(pklfile, 'w') as f: 
                        pickle.dump(Pro, f)

############################# Plot ################################

import sys
sys.path.insert(0, '/home/' + comp + 'src/beliefspaceplanning/toy_simulator/src/')
import varz as V

    

# Goal centers - set 1
C = np.array(
    [[V.GOALX,V.GOALY]])

r = 0.1

path = '/home/' + comp + 'src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

# Plot individual solutions
if 1:
    results_path = '/home/' + comp + 'src/beliefspaceplanning/rollout_node/set/set' + Set + '/results/'

    for set_mode in set_modes:

        fo  = open(results_path + set_mode + '.txt', 'wt') 

        files = glob.glob(path + "*.pkl")

        for k in range(len(files)):

            pklfile = files[k]
            if pklfile.find(set_mode) < 0:
                continue
            ctr = C[int(pklfile[pklfile.find('goal')+4]), :] # Goal center
            print ctr
            # exit(1)

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Plotting file number ' + str(k+1) + ': ' + file_name)
            
            with open(pklfile) as f:  
                Pro = pickle.load(f)

            i = 0
            while i < len(Pro):
                if Pro[i].shape[0] == 1:
                    del Pro[i]
                else:
                    i += 1 

            A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
            maxR = A.shape[0]+1
            maxX = np.max([x.shape[0] for x in Pro])
            
            c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])

            # print maxR, maxX
            # for s in Pro:
            #     print s.shape
            # print "-----"

            Smean = []
            Sstd = []
            for i in range(min(maxR, maxX)):
                F = []
                for j in range(len(Pro)): 
                    if Pro[j].shape[0] > i:
                        F.append(Pro[j][i])
                Smean.append( np.mean(np.array(F), axis=0) )
                Sstd.append( np.std(np.array(F), axis=0) )
            Smean = np.array(Smean)
            Sstd = np.array(Sstd)
            
            c = float(c) / len(Pro)*100
            print("Finished episode success rate: " + str(c) + "%")

            # fig = plt.figure(k)
            fig, ax = plt.subplots()
            rectangle = plt.Rectangle((-V.B, -1*V.YSIZE), 2*V.B, V.YSIZE/2, fc=('0.1'))
            plt.gca().add_patch(rectangle)
            rectangle = plt.Rectangle((-V.B, -0.5*V.YSIZE), 2*V.B, V.YSIZE/2, fc=('0.3'))
            plt.gca().add_patch(rectangle)
            rectangle = plt.Rectangle((-V.B, -0.), 2*V.B, V.YSIZE/2, fc=('0.5'))
            plt.gca().add_patch(rectangle)
            rectangle = plt.Rectangle((-V.B, 0.5*V.YSIZE), 2*V.B, V.YSIZE/2, fc=('0.7'))
            plt.gca().add_patch(rectangle)

            p = 0
            for S in Pro:
                plt.plot(S[:,0], S[:,1], 'r')
                if S.shape[0] < maxR:
                    plt.plot(S[-1,0], S[-1,1], 'oc')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100
            print("Reached goal success rate: " + str(p) + "%")


            plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)

            plt.plot(Straj[:,0], Straj[:,1], '-y', linewidth=3.5, label='Planned path')

            plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(file_name + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
            # plt.axis('equal')

            plt.axis('square')
            plt.axis([-V.XSIZE, V.XSIZE, -V.YSIZE, V.YSIZE])
            plt.xlabel('x')
            plt.ylabel('y')


            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
            plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png')
            # plt.show()

        fo.close()
        

# Compare three methods
if 0:
    results_path = '/home/' + comp + 'src/beliefspaceplanning/rollout_node/set/set' + Set + '/results_goal/'
    PL = {set_modes[0]: 0., set_modes[1]: 0., set_modes[2]: 0.}

    fo  = open(results_path + 'set' + Set + '.txt', 'wt') 

    for goal_num in range(C.shape[0]):
        ctr = C[goal_num]

        fo.write('\ngoal ' + str(goal_num) + ': ' + str(C[goal_num]) + '\n')
        
        fig = plt.figure(figsize=(20,4))

        a = 0
        for set_mode in set_modes:

            files = glob.glob(path + set_mode + '*.pkl')

            found = False
            for k in range(len(files)):
                pklfile = files[k]
                ja = pklfile.find('goal')+4
                ja = pklfile[ja:ja+2] if not pklfile[ja+1] == '_' else pklfile[ja]
                if int(ja) == goal_num:
                    found = True
                    break


            n = len(fig.axes)
            for i in range(n):
                fig.axes[i].change_geometry(1, n+1, i+1)
            ax = fig.add_subplot(1, n+1, n+1)
            # plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), 8, color='w')
            ax.add_artist(goal_plan)
            # plt.ylim([80,130])
            # plt.xlim([-100, 100])
            
            if not found:
                plt.title(set_mode + ", No plan")
                fo.write(set_mode + ': No plan\n')
                continue
            PL[set_mode] += 1.
            pklfile = files[k]

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            with open(pklfile) as f:  
                Pro = pickle.load(f) 

            A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
            maxR = A.shape[0] 
            maxX = np.max([x.shape[0] for x in Pro])
            
            c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])

            Smean = []
            Sstd = []
            for i in range(min(maxR, maxX)):
                F = []
                for j in range(len(Pro)): 
                    if Pro[j].shape[0] > i:
                        F.append(Pro[j][i])
                Smean.append( np.mean(np.array(F), axis=0) )
                Sstd.append( np.std(np.array(F), axis=0) )
            Smean = np.array(Smean)
            Sstd = np.array(Sstd)

            c = float(c) / len(Pro)*100

            p = 0
            t = True
            for S in Pro:
                if t:
                    plt.plot(S[:,0], S[:,1], 'r', label='rollouts')
                    t = False
                else:
                    plt.plot(S[:,0], S[:,1], 'r')

                if S.shape[0] < maxR:
                    plt.plot(S[-1,0], S[-1,1], 'oc')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth=3.5, label='Planned path')

            plt.plot(Smean[:,0], Smean[:,1], '-b', linewidth=3.0, label='rollout mean')
            X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(set_mode + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
            # plt.axis('equal')

            plt.plot(Smean[0,0], Smean[0,1], 'og', markersize=14 , label='start state')
            
            plt.legend()

            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            fo.write(set_mode + ': ' + str(c) + ', ' + str(p) + '\n')
        plt.savefig(results_path + 'set' + Set + '_goal' + str(goal_num) + '.png')    
    
    fo.write('\n\nPlanning success rate: \n')
    for k in list(PL.keys()):
        fo.write(k + ': ' + str(PL[k]/13.*100.) + '\n')
    
    fo.close()
    # plt.show()


        
        

    