#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from control.srv import pathTrackReq
import time
import glob
from scipy.io import loadmat

rollout = 1

# path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'
# path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

comp = 'juntao'
# comp = 'pracsys'

Set = '7'
# set_modes = ['robust_particles_pc','naive_with_svm']#'robust_particles_pc_svmHeuristic','naive_with_svm', 'mean_only_particles']
set_modes = ['robust_particles_pc','naive_with_svm', 'mean_only_particles']
control_met = '_gpc'

# if Set == '1':
#     # Goal centers - set 1
#     C = np.array([
#         [17, 117],
#         [75, 75],
#         [-83, 66],
#         [-52, 77],
#         [48, 114],
#         [-31, 100],
#         [5.4, 108],
#         [87, 65]])

if Set == '1' or Set == '2':
    # Goal centers
    C = np.array([
        [-24, 115],
        [58, 76],
        [-56, 90],36.23284149169921875000,110.15370
        [79, 76],
        [-66, 97],36.23284149169921875000,110.15370
        [-46, 77],
        [-73, 63],
        [60, 100],
        [35, 106],
        [27, 104]])

    if Set == '1':
        Obs = np.array([[42, 90, 12], [-45, 101, 7]])

if Set == '3':
    # Goal centers
    C = np.array([
        [40, 95],
        [50, 111],
        [25, 98],
        [-32, 104]])

    Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

if Set == '4' or Set == '6' or Set == '7':
    # Goal centers
    C = np.array([
        [-37, 119],
        [-33, 102],
        [-22, 129],
        [-52, 112],
        [67, 80],
        [-63, 91]])

    Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

if Set == '5':
    # Goal centers
    C = np.array([
        [50, 111]])

    Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])



############################# Rollout ################################
if rollout:
    track_srv = rospy.ServiceProxy('/control', pathTrackReq)
    rospy.init_node('run_control_set', anonymous=True)
    state_dim = 8

    while 1:
        for set_mode in set_modes:
            path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

            files = glob.glob(path + set_mode + "*.txt")
            files_pkl = glob.glob(path + set_mode + "*.pkl")

            if len(files) == 0:
                continue

            for i in range(len(files)):

                path_file = files[i]
                if path_file[80:].find('plan') > 0:
                    continue
                if any(path_file[:-4] + control_met + '.pkl' in f for f in files_pkl):
                    continue
                pklfile = path_file[:-4] + control_met + '.pkl'

                ctr = np.concatenate((C[int(pklfile[pklfile.find('goal')+4]), :], np.array([0,0,0,0,0,0])), axis=0) # Goal center

                # To distribute rollout files between computers
                # ja = pklfile.find('goal')+4
                # if int(pklfile[ja]) <= 4:
                #     continue

                print('Rolling-out file number ' + str(i+1) + ': ' + path_file + '.')

                S = np.loadtxt(path_file, delimiter=',', dtype=float)[:,:state_dim]
                S = np.append(S, [ctr], axis=0)

                Pro = []
                Aro = []
                Suc = []
                for j in range(2):
                    print("Rollout number " + str(j) + ".")
                    
                    res = track_srv(S.reshape((-1,)))
                    Sreal = np.array(res.real_path).reshape(-1, S.shape[1])
                    Areal = np.array(res.actions).reshape(-1, 2)
                    success = res.success

                    Pro.append(Sreal)
                    Aro.append(Areal)
                    Suc.append(success)

                    with open(pklfile, 'w') as f: 
                        pickle.dump([Pro, Aro, Suc], f)

############################# Plot ################################

rp = 7.
r = 10.

set_num = Set
set_modes = ['robust_particles_pc','naive_with_svm', 'mean_only_particles']

if not rollout and 1:

    file = 'sim_data_discrete_v14_d8_m10.mat'
    path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    Q = loadmat(path + file)
    Data = Q['D']

    results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/cl_results/'

    for set_mode in set_modes:

        path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

        fo  = open(results_path + set_mode + '.txt', 'wt') 

        files = glob.glob(path + "*.pkl")

        for k in range(len(files)):

            pklfile = files[k]
            if pklfile[80:].find('plan') > 0:
                continue
            if pklfile.find(set_mode) < 0 or pklfile.find(control_met) < 0:
                continue
            print "\nRunning pickle file: " + pklfile
            ctr = C[int(pklfile[pklfile.find('goal')+4]), :] # Goal center
            print ctr
            # exit(1)

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            trajfile = pklfile[:-(12 if control_met == '_gpc' else 8)] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Plotting file number ' + str(k+1) + ': ' + file_name)
            
            with open(pklfile) as f:  
                Pro, Aro, Suc = pickle.load(f)

            # if file_name == 'robust_particles_pc_goal1_run1_plan':
            #     for s in Pro:
            #         print s
            #         raw_input()

            i = 0
            while i < len(Pro):
                if Pro[i].shape[0] == 1:
                    del Pro[i]
                else:
                    i += 1 

            A = np.loadtxt(pklfile[:-8] + '.txt', delimiter=',', dtype=float)[:,:2]
            maxR = A.shape[0]+1
            maxX = np.max([x.shape[0] for x in Pro])
            
            c = np.sum([(1 if x else 0) for x in Suc])

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

            # fig = plt.figure()
            fig, ax = plt.subplots(figsize=(12,12))
            plt.plot(Data[:,0], Data[:,1], '.y', zorder=0)

            p = 0
            for S in Pro:
                plt.plot(S[:,0], S[:,1], '.--r')
                if S.shape[0] < maxR:
                    plt.plot(S[-1,0], S[-1,1], 'oc')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100
            print("Reached goal success rate: " + str(p) + "%")

            plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), 8, color='w')
            ax.add_artist(goal_plan)

            try:
                for o in Obs:
                    obs = plt.Circle(o[:2], o[2])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '.-k', linewidth=3.5, label='Planned path')

            plt.plot(Smean[:,0], Smean[:,1], '.-b', label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(file_name + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
            plt.axis('equal')

            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
            plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png', dpi=250)
            # plt.show()

        fo.close()
        
    # plt.show()

if not rollout and 0:
    results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/results_goal/'
    PL = {set_modes[0]: 0., set_modes[1]: 0., set_modes[2]: 0.}

    fo  = open(results_path + 'set' + set_num + '.txt', 'wt') 

    for goal_num in range(C.shape[0]):
        ctr = C[goal_num]
        print "ctr: ", ctr

        fo.write('\ngoal ' + str(goal_num) + ': ' + str(C[goal_num]) + '\n')
        
        fig = plt.figure(figsize=(20,7))

        a = 0
        for set_mode in set_modes:

            path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

            files = glob.glob(path + set_mode + "*.pkl")

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
            ax = fig.add_subplot(1, n+1, n+1)#, aspect='equal')
            plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), rp, color='w')
            ax.add_artist(goal_plan)
            # plt.ylim([40,130])
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

            Smean = []
            Sstd = []
            for i in range(min(maxR, maxX)):
                F = []
                for S in Pro: 
                    if S.shape[0] > i:
                        F.append(S[i])
                Smean.append( np.mean(np.array(F), axis=0) )
                Sstd.append( np.std(np.array(F), axis=0) )
            Smean = np.array(Smean)
            Sstd = np.array(Sstd)

            c = float(c) / len(Pro)*100

            p = 0
            t = True
            for S in Pro:
                if t:
                    plt.plot(S[:,0], S[:,1], '--r', label='rollouts')
                    t = False
                else:
                    plt.plot(S[:,0], S[:,1], '--r')

                if S.shape[0] < maxR:
                    plt.plot(S[-1,0], S[-1,1], 'oc')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100

            try:
                for o in Obs:
                    obs = plt.Circle(o[:2], o[2])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth=3.5, label='Planned path')

            plt.plot(Smean[:,0], Smean[:,1], '-b', linewidth=3.0, label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
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
        plt.savefig(results_path + 'set' + str(set_num) + '_goal' + str(goal_num) + '.png', dpi=300) 
        # plt.show()   
    
    fo.write('\n\nPlanning success rate: \n')
    for k in list(PL.keys()):
        fo.write(k + ': ' + str(PL[k]/13.*100.) + '\n')
    
    fo.close()
    # plt.show()


        
        

    