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

comp = 'juntao'
# comp = 'pracsys'

Set = '4c_nn'
# set_modes = ['robust_particles_pc', 'naive_with_svm']#'robust_particles_pc_svmHeuristic','naive_with_svm', 'mean_only_particles']
set_modes = ['naive_with_svm']
# set_modes = ['robust_particles_pc']
# set_modes = ['mean_only_particles']

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
                if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                    continue
                if int(action_file[action_file.find('run')+3]) > 0:
                    continue
                pklfile = action_file[:-3] + 'pkl'

                # To distribute rollout files between computers
                ja = pklfile.find('goal')+4
                num = int(pklfile[ja]) if pklfile[ja+1] == '_' else int(pklfile[ja:ja+2])
                # if num > 9:
                #     continue

                print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.')

                A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

                Af = A.reshape((-1,))
                Pro = []
                for j in range(10):
                    print("Rollout number " + str(j) + ".")
                    
                    Sro = np.array(rollout_srv(Af, [0,0,0,0]).states).reshape(-1,state_dim)

                    Pro.append(Sro)

                    with open(pklfile, 'w') as f: 
                        pickle.dump(Pro, f)

############################# Plot ################################

if Set == '5':
    # Goal centers
    C = np.array([
        [50, 111]])

    Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

if Set == '8' or Set == '9':
    C = np.array([[-37, 119],
    [-33, 102],
    [-60, 90],
    [-40, 100],
    [-80, 65],
    [-80, 80],
    [-50, 90],
    [60, 90],
    [80, 80],
    [50, 90],
    [40, 100],
    [80, 65],
    [-52, 112],
    [67, 80],
    [-63, 91]])

    Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

if Set == '10':
    C = np.array([[-37, 119], 
    [-33, 102],
    [-40, 100],
    [-80, 80],
    [-50, 90],
    [50, 90],
    [40, 100],
    [-52, 112],
    [67, 80],
    [-63, 91]])

    # Obs = np.array([[-11, 111, 2.6], [-12, 118, 2.55], [11, 113, 2.5], [12, 120, 2.5]])

# if Set == '11':
#     C = np.array([[-27,111],
#     [-23, 112],
#     [-36, 107]])

#     Obs = np.array([[-11, 111, 2.6], [-12, 118, 2.55]])

if Set == '11': # New
    C = np.array([[-27,107],
    [-23, 112],
    [-36, 107]])

    Obs = np.array([[-11, 108, 2.7], [-12, 115, 2.7]])

if Set == '12': # New
    C = np.array([[-27,107],
    [-23, 112],
    [-36, 107]])

    Obs = np.array([[-10, 111.7, 2.7], [-12, 118, 2.7]])

if Set == '14_nn': # New
    C = np.array([[37, 119], 
        [-33, 102],
        [-40, 100],
        [-80, 80],
        [-50, 90],
        [50, 90],
        [40, 100],
        [-52, 112],
        [67, 80],
        [-63, 91]])

    Obs = np.array([[-11, 111, 2.6], [-12, 118, 2.55]])

if Set == '15_nn' or Set == '16_nn': # New
    C = np.array([[-21, 104],
        [-35, 100],
        [-27, 98],
        [-23, 106]])

    Obs = np.array([[-9., 107.2, 2.7], [-12, 114, 2.551]])

# if Set == '17_nn': # New
#     C = np.array([[-40, 97]])

#     Obs1 = np.array([-24, 115.2, 2.55]) # Upper
#     Obs2 = np.array([-22., 108.7, 2.8]) # Lowerr
#     Obs = np.array([Obs1, Obs2])

if Set == '17_nn': # New
    C = np.array([[-37, 119], 
        [-33, 102],
        [-40, 100],
        [-80, 80],
        [-50, 90],
        [50, 90],
        [40, 100],
        [-52, 112],
        [67, 80],
        [-63, 91]])

if Set == '18_nn': # New
    C = np.array([[-63, 91],
        [-50, 90]])

    Obs1 = np.array([-38, 116.7, 4.]) # Upper
    Obs2 = np.array([-33., 106, 4.]) # Lower
    Obs = np.array([Obs1, Obs2])

if Set == '19_nn': # New
    C = np.array([[-59, 90], [-42, 94], [53,93]])

    Obs = np.array([[-38, 117.1, 4.],
        # [-33., 105., 4.],
        [-33., 106.2, 4.],
        [-52.5, 105.2, 4.],
        [-51., 105.5, 4.],
        [43., 111.5, 6.],
        [59., 80., 3.],
        [36.5, 94., 4.]
        ])
    
if Set == '20_nn':
    C = np.array([[40,91],
                [69,72],
                [75,81],
                [-38,92],
                [-75,72],
                [-55,100],
                [-62,78]
    ])

    Obs = np.array([[-47, 111, 5.],
        [-22, 107, 4.],
        [-60, 87, 4.],
        [50., 104, 3.],
        [61., 87., 3.],
        [32, 102., 6.]
        ])

if Set == '21_nn': # New
    C = np.array([[-58, 80],
            [50,78],
            [73,76],
            [-26,96],
            [57,103],
    ])
    Obs = np.array([[-38, 117.1, 4.],
        [-33., 105., 4.],
        [-52.5, 105.2, 4.],
        [43., 111.5, 6.],
        [59., 80., 3.],
        [36.5, 94., 4.]
        ])

if Set == '0c_nn' or Set == '1c_nn':
   C = np.array([[-37, 119 ],
        [-33, 102],
        [-40, 100],
        [-80, 80],
        [-50, 90],
        [50, 90],
        [40, 100],
        [-52, 112],
        [67, 80],
        [-63, 91]])

if Set == '2c_nn' or Set == '2woc_nn' or Set == '4c_nn' or Set == '3c_nn':
    C = np.array([[-37, 119 ],
        [-33, 102],
        [-40, 100],
        [-80, 80],
        [-50, 90],
        [50, 90],
        [40, 100],
        [-52, 112],
        [67, 80],
        [-63, 91],
        [75,75]])

    Obs = np.array([[-38, 117.1, 4.],
        [-33., 100., 4.],
        [-52.5, 105.2, 4.],
        [43., 111.5, 6.],
        [59., 80., 3.],
        [36.5, 94., 4.]
        ])


# ===============================================
    
def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    l = 0.
    for i in range(1,S1.shape[0]):
        l += np.linalg.norm(S1[i,:2] - S1[i-1,:2])

    return np.sqrt(Sum / S1.shape[0]), l

rp = 7.
r = 10.

set_num = Set
# set_modes = ['robust_particles_pc','naive_with_svm', 'mean_only_particles']
set_modes = ['naive_withCriticThreshold', 'naive_withCriticCost', 'naive_withCriticPredict', 'naive']

results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/results/'


if not rollout and 0:

    file = 'sim_data_discrete_v14_d8_m10.mat'
    path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
    Q = loadmat(path + file)
    Data = Q['D']

    Sum = {set_mode[:set_mode.find('_')]: np.zeros((C.shape[0],5)) for set_mode in set_modes} 
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
            num = int(pklfile[ja]) if pklfile[ja+1] == '_' else int(pklfile[ja:ja+2])
            ctr = C[num, :] # Goal center
            print(("Goal number %d with center: "%num), ctr)
            # exit(1)

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            planner = file_name[:file_name.find('_')]

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
            maxR = A.shape[0]+1
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

            p = 0
            for S in Pro:
                if S.shape[0] < maxR or np.linalg.norm(S[-1,:2]-ctr) > r:
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

            # plt.plot(ctr[0], ctr[1], 'om')
            # goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            # ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal_plan)

            try:
                kj = 0
                for os in Obs:
                    o = np.copy(os)
                    if Set == '19_nn':
                        if kj == 2 and num > 0:
                            kj += 1
                            continue
                        if kj == 3 and num == 0:
                            kj += 1
                            continue
                    if num == 0 and kj == 1:
                        o[1] = 105.
                    
                    obs = plt.Circle(o[:2], o[2], color=[0.4,0.4,0.4])#, zorder=10)
                    ax.add_artist(obs)
                    kj += 1
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')

            # plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
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

            e, l = tracking_error(Smean, Straj)
            Sum[planner][num, 4] += 1
            if p >= Sum[planner][num, 1] and (Sum[planner][num, 2] == 0 or Sum[planner][num, 2] > e):
                Sum[planner][num, 0] = 100 - c # Percent failures
                Sum[planner][num, 1] = p # Success rate
                Sum[planner][num, 2] = e # Tracking error
                Sum[planner][num, 3] = l # Planned path length

            fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
            plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png', dpi=300)
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


if 1: # Plot just plans

    for set_mode in set_modes:

        file = 'sim_data_discrete_v14_d8_m10.mat'
        path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
        Q = loadmat(path + file)
        Data = Q['D']

        path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

        files = glob.glob(path + "*.txt")

        for k in range(len(files)):

            trajfile = files[k]
            if trajfile.find('traj') < 0 or trajfile.find(set_mode) < 0:
                continue
            print "\nRunning pickle file: " + trajfile
            
            ja = trajfile.find('goal')+4
            num = int(trajfile[ja]) if trajfile[ja+1] == '_' else int(trajfile[ja:ja+2])
            ctr = C[num, :] # Goal center
            print(("Goal number %d with center: "%num), ctr)

            for j in range(len(trajfile)-1, 0, -1):
                if trajfile[j] == '/':
                    break
            file_name = trajfile[j+1:-4]

            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Plotting file number ' + str(k+1) + ': ' + file_name)

            fig, ax = plt.subplots(figsize=(12,12))
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

            goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal_plan)

            try:
                for o in Obs:
                    obs = plt.Circle(o[:2], o[2], color=[0.4,0.4,0.4])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')
            plt.title(file_name)
            plt.axis('equal')

            for i in range(len(trajfile)-1, 0, -1):
                if trajfile[i] == '/':
                    break

            plt.savefig(results_path + '/' + trajfile[i+1:-4] + '.png', dpi=300)