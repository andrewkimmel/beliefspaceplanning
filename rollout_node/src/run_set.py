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
from scipy.io import loadmat
from scipy.spatial import ConvexHull, convex_hull_plot_2d

rollout = 0

# path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'
# path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

# comp = 'juntao'
comp = 'pracsys'

Set = '19c_7'
# set_modes = ['robust_particles_pc', 'naive_with_svm']#'robust_particles_pc_svmHeuristic','naive_with_svm', 'mean_only_particles']
# set_modes = ['naive_with_svm']
# set_modes = ['robust_particles_pc']
# set_modes = ['mean_only_particles']
# set_modes = ['naive_withCriticThreshold', 'naive_withCriticCost', 'naive_goal']
# set_modes_no = ['naive_goal', 'naive_withCriticPredict']
set_modes = ['naive_withCriticCost']#, 'naive_goal']
# set_modes = ['naive_goal']#, 'naive_goal']



############################# Rollout ################################
if rollout:
    rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
    rospy.init_node('run_rollout_set', anonymous=True)
    state_dim = 4

    while 1:
    # for _ in range(10):
        for set_mode in set_modes:
            path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

            files = glob.glob(path + set_mode + "*.txt")
            files_pkl = glob.glob(path + set_mode + "*.pkl")

            if len(files) == 0:
                continue

            for i in range(len(files)):

                action_file = files[i]
                if action_file.find('traj') > 0:
                    continue
                files_pkl = glob.glob(path + set_mode + "*.pkl")
                if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                    continue
                # if int(action_file[action_file.find('run')+3]) > 0:
                #     continue
                # if action_file.find(set_modes_no[0]) > 0 or action_file.find(set_modes_no[1]) > 0:
                #     continue
                pklfile = action_file[:-3] + 'pkl'

                # To distribute rollout files between computers
                ja = pklfile.find('goal')+4
                jb = ja + 1
                while not (pklfile[jb] == '_'):
                    jb += 1
                num = int(pklfile[ja:jb])#int(pklfile[ja]) if pklfile[ja+1] == '_' else int(pklfile[ja:ja+2])
                # if num == 124:
                #     continue

                print('Rolling-out goal number ' + str(num) + ': ' + action_file + '.')

                try:
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                except:
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)
                    print A.shape

                Af = A.reshape((-1,))
                Pro = []
                for j in range(10):
                    print("Rollout number " + str(j) + ".")
                    
                    Sro = np.array(rollout_srv(Af, [0,0,0,0]).states).reshape(-1,state_dim)

                    Pro.append(Sro)

                    with open(pklfile, 'w') as f: 
                        pickle.dump(Pro, f)

############################# Plot ################################


if Set == '8c_nn' or Set.find('t9c') >= 0:
    # C = np.zeros((1000,2))
    C = np.loadtxt('/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set8c_nn/random_goals.txt', delimiter=',', dtype=float)[:,:2]

    with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs.pkl', 'r') as f: 
        Obs = pickle.load(f)

if Set.find('10c') >= 0 or Set.find('11c') >= 0:
    C = np.array([[90, 60 ],
        [-90, 60]])

    if 0:
        Obs1 = []
        for x in range(0, 71, 10):
            for y in range(50, 145, 13):
                Obs1.append([x, y, 0.75])
        Obs1 = np.array(Obs1)
        np.random.seed(170)
        n = 60
        Obs2 = np.concatenate((np.random.random(size=(n,1))*-71, np.random.random(size=(n,1))*95+50, 0.75*np.ones((n,1))), axis=1)
        Obs3 = np.concatenate((Obs1, Obs2), axis = 0)

        Obs = []
        for o in Obs3:
            if (o[1] < 87 and np.abs(o[0]) < 44) or o[1] > 133 or (o[0] > 55 and o[1] > 104) or (o[1] < 70 and np.abs(o[0]) < 61) or np.linalg.norm(o[:2] - np.array([0.,118.])) < 5.:
                continue
            Obs.append(o)
        Obs = np.array(Obs)

        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs.pkl', 'w') as f: 
            pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('12c') >= 0:
    C = np.array([[25, 104 ],[25, 104 ]])

    if 0:
        Obs1 = []
        y = 112
        for x in range(17, 35, 1):
            Obs1.append([x, y, 0.5])
            y -= 0.27
        Obs2 = []
        y = 112
        for x in np.arange(17, 14, -0.2):
            Obs2.append([x, y, 0.5])
            y -= 0.75
        Obs3 = []
        y = 101
        for x in range(15, 33, 1):
            Obs3.append([x, y, 0.5])
            y -= 0.27
        Obs = np.concatenate((np.array(Obs1), np.array(Obs2), np.array(Obs3)), axis = 0)
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_12.pkl', 'w') as f: 
                pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_12.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('13c') >= 0:
    C = np.array([[25-4, 104+19 ], [25-4, 104+19 ]])

    if 0:
        bx = 4
        by = 19
        Obs1 = []
        y = 112+by
        for x in range(17, 35, 1):
            Obs1.append([x-bx, y, 0.5])
            y -= 0.27
        Obs2 = []
        y = 112+by
        for x in np.arange(17, 14, -0.2):
            Obs2.append([x-bx, y, 0.5])
            y -= 0.75
        Obs3 = []
        y = 101+by
        for x in range(15, 33, 1):
            Obs3.append([x-bx, y, 0.5])
            y -= 0.27
        Obs = np.concatenate((np.array(Obs1), np.array(Obs2), np.array(Obs3)), axis = 0)
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_13.pkl', 'w') as f: 
                pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_13.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('14c') >= 0:
    C = np.array([[25-4, 104+19 ], [25-4, 104+19 ]])

    if 1:
        bx = 4
        by = 19
        Obs1 = []
        y = 112+by
        for x in range(17, 35, 1):
            Obs1.append([x-bx, y, 0.5])
            y -= 0.27
        Obs2 = []
        y = 112+by
        for x in np.arange(17, 14, -0.2):
            Obs2.append([x-bx, y, 0.5])
            y -= 0.75
        Obs3 = []
        y = 101+by
        for x in range(15, 33, 1):
            Obs3.append([x-bx, y, 0.5])
            y -= 0.27
        Obs4 = []
        y = 115.5
        for x in np.arange(28, 29.5, 0.3):
            Obs4.append([x, y, 0.5])
            y += 0.7
        Obs = np.concatenate((np.array(Obs1), np.array(Obs2), np.array(Obs3), np.array(Obs4)), axis = 0)
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_14.pkl', 'w') as f: 
                pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_14.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('15c') >= 0:
    C = np.array([[90, 60 ],
        [-90, 60]])

    with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs.pkl', 'r') as f: 
        Obs = pickle.load(f)

if Set.find('16c') >= 0:
    C = np.array([[49, 92 ],
        [62, 83],
        [41, 116],
        [70, 72], 
        [20, 123],
        [16, 107]])

if Set.find('17c') >= 0:
    C = np.array([[72, 86 ], [67, 76], [47, 96]])

    if 0:
        Obs = np.array([[30, 116.5, 0.65], [39,104, 0.65], [58, 98, 0.65]])
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_17.pkl', 'w') as f: 
                pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_17.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('18c') >= 0:
    C = np.array([[-66, 80 ], [-41, 100], [-62, 96], [-49, 86], [-55, 92]])

    if 1:
        Obs1 = []
        y = 122
        x = -33
        for _ in range(6):
            Obs1.append([x, y, 0.5])
            y -= 4
            x += 2.7
        Obs2 = []
        y = 105
        x = -55
        for _ in range(6):
            Obs2.append([x, y, 0.5])
            y -= 4
            x += 2.7
        Obs = np.concatenate((np.array(Obs1), np.array(Obs2)), axis = 0)
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_18.pkl', 'w') as f: 
            pickle.dump(Obs, f)
    else:
        with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_18.pkl', 'r') as f: 
            Obs = pickle.load(f)

if Set.find('19c') >= 0:
    C = np.array([[-66, 80],
     [-41, 100], 
     [-62, 96], 
     [-49, 86], 
     [-55, 92],
     [59, 78],
     [31, 102],
     [60, 100],
     [52, 95],
     [-78, 67],
     [31, 125],
     [-26, 125],
     [0, 107],
     [3, 130],
     [-48, 114],
     [69, 78],
     ])

    if 0:
        np.random.seed(170)
        n = 60
        Obs = np.concatenate((np.random.random(size=(n,1))*160-80, np.random.random(size=(n,1))*95+50, 0.75*np.ones((n,1))), axis=1)
        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_19.pkl', 'w') as f: 
            pickle.dump(Obs, f)
    else:
        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_19.pkl', 'r') as f: 
            Obs = pickle.load(f)


fig, ax = plt.subplots(figsize=(8,3.5))
# plt.plot(Data[:,0], Data[:,1], '.y', zorder=0)

file = 'sim_data_discrete_v14_d8_m10.mat'
path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
Q = loadmat(path + file)
Data = Q['D']
hull = ConvexHull(Data[:,:2])
H1 = []
for simplex in hull.vertices:
    H1.append(Data[simplex, :2])
H2 = np.array([[-87,41],[-83,46],[-76,52],[-60,67],[-46,79],[-32,90],[-18,100],[0,105],[18,100],[32,90],[46,79],[60,67],[76,52],[83,46],[87,41]])
H = np.concatenate((np.array(H1)[2:,:], H2), axis=0)
pgon = plt.Polygon(H, color='y', alpha=1, zorder=0)
ax.add_patch(pgon)

inx = [0, 7, 8, 15, 2]

j = 1
for i in inx:
    ctr = C[i]
    goal_plan = plt.Circle((ctr[0], ctr[1]), 8., color='m')
    ax.add_artist(goal_plan)
    plt.text(ctr[0]-2.5, ctr[1]-2, str(j), fontsize=20)
    j += 1

for os in Obs:
    obs = plt.Circle(os[:2], os[2], color=[0.4,0.4,0.4])#, zorder=10)
    ax.add_artist(obs)
    
plt.plot(0, 119, 'ok', markersize=16, color ='r')
plt.text(-15, 123, 'start state', fontsize=16, color ='r')
plt.ylim([60, 140])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/goals.png', dpi=200)
# plt.show()
exit(1)



# ===============================================
    
def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    l = 0.
    for i in range(1,S2.shape[0]):
        l += np.linalg.norm(S2[i,:2] - S2[i-1,:2])

    return np.sqrt(Sum / S1.shape[0]), l

rp = 5.0
r = 8.

set_num = Set
# set_modes = ['robust_particles_pc','naive_with_svm', 'mean_only_particles']
set_modes = ['naive_goal', 'naive_withCriticThreshold', 'naive_withCriticCost', 'naive_withCriticPredict']

results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/results/'

Pc = []
Acc = []
Pn = []
Acn = []
if not rollout and 1:

    file = 'sim_data_discrete_v14_d8_m10.mat'
    path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    Q = loadmat(path + file)
    Data = Q['D']

    # Sum = {set_mode[:set_mode.find('_')]: np.zeros((C.shape[0],5)) for set_mode in set_modes} 
    Sum = {set_mode: np.zeros((C.shape[0],5)) for set_mode in set_modes} 
    # run_num = 0

    for set_mode in set_modes:

        path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

        fo  = open(results_path + set_mode + '.txt', 'wt') 

        files = glob.glob(path + "*.pkl")

        for k in range(len(files)):

            pklfile = files[k]
            if pklfile.find('traj') > 0:
                continue
            if pklfile.find(set_mode) < 0:
                continue
            # if int(pklfile[pklfile.find('goal')+4]) < 5:
            #     continue
            print "\nRunning pickle file: " + pklfile
            
            ja = pklfile.find('goal')+4
            jb = ja + 1
            while not (pklfile[jb] == '_'):
                jb += 1
            num = int(pklfile[ja:jb])
            try:
                ctr = C[num, :] # Goal center
            except:
                ctr = np.array([0,0])
            print(("Goal number %d with center: "%num), ctr)

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            planner = set_mode#file_name[:file_name.find('_')]

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Plotting file number ' + str(k+1) + ': ' + file_name)
            
            with open(pklfile) as f:  
                Pro = pickle.load(f)
            
            i = 0
            Ss = []
            while i < len(Pro):
                if Pro[i].shape[0] == 1 or np.linalg.norm(Pro[i][0,:2] - np.array([0,118])) > 5. :
                    del Pro[i]
                else:
                    Ss.append(Pro[i][0,:2])
                    i += 1 

            A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
            maxR = Straj.shape[0]
            maxX = np.max([x.shape[0] for x in Pro])
            
            c = np.sum([(1 if x.shape[0]>maxR-20 else 0) for x in Pro])

            e = np.zeros((maxR, 1))
            Pro_suc = []
            for S in Pro:
                if not (S.shape[0] > maxR-20):
                    continue
                Pro_suc.append(S)
            
            Smean = []
            Sstd = []
            for i in range(min(maxR, maxX)):
                F = []
                for j in range(len(Pro_suc)): 
                    F.append(Pro_suc[j][i])
                Smean.append( np.mean(np.array(F), axis=0) )
                Sstd.append( np.std(np.array(F), axis=0) )
            Smean = np.array(Smean)
            Sstd = np.array(Sstd)
            # Smean = Smean[:Straj.shape[0],:]
            
            c = float(c) / len(Pro)*100
            print("Finished episode success rate: " + str(c) + "%")

            # fig = plt.figure(k)
            fig, ax = plt.subplots(figsize=(12,12))
            # plt.plot(Data[:,0], Data[:,1], '.y', zorder=0)

            # -----
            hull = ConvexHull(Data[:,:2])
            H1 = []
            for simplex in hull.vertices:
                H1.append(Data[simplex, :2])
            H2 = np.array([[-87,41],[-83,46],[-76,52],[-60,67],[-46,79],[-32,90],[-18,100],[0,105],[18,100],[32,90],[46,79],[60,67],[76,52],[83,46],[87,41]])
            H = np.concatenate((np.array(H1)[2:,:], H2), axis=0)
            pgon = plt.Polygon(H, color='y', alpha=1, zorder=0)
            ax.add_patch(pgon)
            # -----

            print len(Pro)

            for kk in range(len(Pro)):
                S = Pro[kk]
                if S.shape[0] < maxR:
                    ii = S.shape[0]-1
                    while np.linalg.norm(S[ii,:2]-S[ii-1,:2]) > 5:
                        ii -= 1
                    Pro[kk] = S[:ii-1]

            p = 0
            for S in Pro:
                # if (Set != '11c_7' and np.linalg.norm(S[-1,:2]-ctr) > r) or (Set == '11c_7' and (S[-1,0] < 70 and S[-1,0] > -70)): #S.shape[0] < maxR or 
                if np.linalg.norm(S[-1,:2]-ctr) > r:
                    plt.plot(S[:,0], S[:,1], '-r')
                    plt.plot(S[-1,0], S[-1,1], 'or')
                else:
                    if set_mode == 'robust_particles_pc' and num == 0:
                        plt.plot(S[-8,0], S[-8,1], 'ob')
                        plt.plot(S[:-7,0], S[:-7,1], '-b')
                    else:
                        plt.plot(S[-1,0], S[-1,1], 'ob')
                        plt.plot(S[:,0], S[:,1], '-b')
                    p += 1
            p = float(p) / len(Pro)*100
            print("Reached goal success rate: " + str(p) + "%")

            if Set.find('10c') >= 0 or Set.find('11c') >= 0:
                import matplotlib.patches as patches
                goal_plan = patches.Rectangle((70, 38), 20, 65, color='m')
                ax.add_artist(goal_plan)
                goal_plan = patches.Rectangle((-70-20, 38), 20, 65, color='m')
                ax.add_artist(goal_plan)
            else:
                goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
                ax.add_artist(goal_plan)

            try:
                for os in Obs:
                    obs = plt.Circle(os[:2], os[2], color=[0.4,0.4,0.4])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            # if pklfile.find('withCriticSeq_goal7_run1') > 0:
            print Smean.shape, Straj.shape
                # exit(1)

            if len(Pro_suc) > 0:
                e, l = tracking_error(Smean, Straj)
            else:
                e, l = -1, -1
            print "Error: " + str(e) + 'mm, path length: ' + str(l) + 'mm'
            Sum[planner][num, 4] += 1
            if p >= Sum[planner][num, 1] and (Sum[planner][num, 2] == 0 or Sum[planner][num, 2] > e):
                Sum[planner][num, 0] = 100 - c # Percent failures
                Sum[planner][num, 1] = p # Success rate
                Sum[planner][num, 2] = round(e, 2) # Tracking error
                Sum[planner][num, 3] = round(l, 2) # Planned path length

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')

            # plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(file_name + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%, RMSE: " + str(round(e, 2)) + ' mm', fontsize = 17)
            plt.axis('equal')

            if 1:# num == 0:
                if set_mode == 'naive_goal':
                    Pn.append(p)
                    Acn.append(e)
                if set_mode == 'naive_withCriticCost':
                    Pc.append(p)
                    Acc.append(e)
                
            fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
            plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png', dpi=200)
            # plt.show()
            # exit(1)

        fo.close()
        
    # plt.show()

    download_dir = results_path + '/summary.csv' 
    csv = open(download_dir, "w") 
    csv.write("Goal #,")
    for key in Sum.keys():
        for _ in range(Sum[key].shape[1]):
            csv.write(key + ',')
    csv.write('\n')
    csv.write(',fail rate, reached goal, tracking RMSE, path length, planning rate, fail rate, reached goal, tracking RMSE, path length, planning rate, fail rate, reached goal, tracking RMSE, path length, planning rate\n')
    for goal in range(C.shape[0]):
        csv.write(str(goal) + ',')
        for key in Sum.keys():
            if np.all(Sum[key][goal,:] == 0):
                csv.write('-,-,-,-,0,')
            else:
                for j in range(Sum[key].shape[1]):
                    if j == 4:
                        csv.write(str(Sum[key][goal, j]/2.0*100.) + ',')
                    else:
                        csv.write(str(Sum[key][goal, j]) + ',')
        csv.write('\n')

    print
    print "Naive: "
    print "Mean success rate: ", np.mean(np.array(Pn))
    print "Mean error: ", np.mean(np.array(Acn))
    print "Critic: "
    print "Mean success rate: ", np.mean(np.array(Pc))
    print "Mean error: ", np.mean(np.array(Acc))


if 0:
    set_w = '2c_nn'
    set_wo = '2woc_nn'
    path = lambda set: '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + set + '/results/'

    files_w = glob.glob(path(set_w) + "*.png")
    files_wo = glob.glob(path(set_wo) + "*.png")

    for F in files_w:
        
        if F.find('goal') < 0 or F.find('naive') < 0:
            continue
        goal = int(F[F.find('goal')+4])
        fwo = -1
        for g in files_wo:
            if int(g[g.find('goal')+4]) == goal:
                fwo = g
                break
        if fwo < 0:
            continue

        print
        print F
        print 'Saving goal %d'%goal
        Iw = plt.imread(F)
        Iwo = plt.imread(fwo)  
        I = np.concatenate((Iwo, Iw), axis=1)  
        plt.imsave(path(set_w) + 'c' + set_w + '_goal' + str(goal) + '.png', I)  

if 0:
    run = 0
    path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/results/'
    path_b = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/naive_baseline/'

    files = glob.glob(path + "*.png")
    files_b = glob.glob(path_b + "*.png")

    for File in files:
        ja = File.find('goal')+4
        jb = ja + 1
        while not (File[jb] == '_'):
            jb += 1
        goal = int(File[ja:jb])
        print 'Goal ' + str(goal)

        G = []
        for F in files_b:
            ja = F.find('goal')+4
            jb = ja + 1
            while not (F[jb] == '_'):
                jb += 1
            goal_b = int(F[ja:jb])
            if goal_b == goal:
                break

        I = plt.imread(File)
        Ib = plt.imread(F)
        I = np.concatenate((I, Ib), axis=1)

        # i = 0
        # for S in set_modes:
        #     f = [s for s in G if S in s]#[0]
        #     Iw = plt.imread(f[0]) if f else np.zeros((2400,2400,4))
        #     I = np.concatenate((I, Iw), axis=1) if i > 0 else Iw
        #     i += 1
        plt.imsave(path + 'c' + Set + '_goal' + str(goal) + '.png', I)  




    # for F in files_w:
        
    #     if F.find('goal') < 0 or F.find('naive') < 0:
    #         continue
    #     goal = int(F[F.find('goal')+4])
    #     fwo = -1
    #     for g in files_wo:
    #         if int(g[g.find('goal')+4]) == goal:
    #             fwo = g
    #             break
    #     if fwo < 0:
    #         continue

    #     print
    #     print F
    #     print 'Saving goal %d'%goal
    #     Iw = plt.imread(F)
    #     Iwo = plt.imread(fwo)  
    #     I = np.concatenate((Iwo, Iw), axis=1)  
    #     plt.imsave(path(set_w) + 'c' + set_w + '_goal' + str(goal) + '.png', I)  
