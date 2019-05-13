#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse
from control.srv import pathTrackReq

import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var


class Plot():

    counter = 0
    obj_pos = np.array([0.,0.])
    start_pos = np.array([0.,0.])
    freq = 15
    goal = np.array([0.,0.])
    clear = False
    Sref = []
    ref_flag = False

    def __init__(self):
        
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Service('/plot/clear', Empty, self.callbackClear)
        rospy.Service('/plot/ref', pathTrackReq, self.callbackPlot)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)

        rospy.init_node('Plot_obj_pos', anonymous=True)
        rate = rospy.Rate(self.freq) # 15hz
        while not rospy.is_shutdown():
            # plt.show()
            rospy.spin()
            # rate.sleep()

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)
        self.obj_pos = self.obj_pos[:2]*1000 # m to mm

        if self.counter % 100 == 0:
            if self.clear:
                plt.gcf().clear()
                self.clear = False
            if self.ref_flag:
                plt.plot(self.Sref[:,0], self.Sref[:,1],'--k')
                self.ref_flag = False

            # ax = plt.subplot()
            # Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])
            # for o in Obs:
            #     obs = plt.Circle(o[:2], o[2])#, zorder=10)
            #     ax.add_artist(obs)

            plt.plot(self.obj_pos[0], self.obj_pos[1],'.r')
            # plt.plot(self.start_pos[0], self.start_pos[1],'*b')
            plt.plot(self.goal[0], self.goal[1],'*g')
            plt.axis("equal")
            plt.axis([-100, 100, 90, 150])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Real-time object position:' + str(self.obj_pos))
            plt.draw()
            plt.pause(0.00000000001)

        self.counter += 1

        if self.counter <= 5:
            self.start_pos = self.obj_pos

    def callbackClear(self, msg):
        self.clear = True
        return EmptyResponse()

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def callbackPlot(self, req):
        self.Sref = np.array(req.desired_path).reshape(-1,var.state_dim_)
        self.ref_flag = True

        return {'real_path': [], 'success' : True}

    


if __name__ == '__main__':
    
    try:
        Plot()
    except rospy.ROSInterruptException:
        pass