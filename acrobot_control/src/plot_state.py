#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse
from control.srv import pathTrackReq
from prx_simulation.srv import simulation_observation_srv

class Plot():

    counter = 0
    state = np.array([0.,0.])
    start_pos = np.array([0.,0.])
    freq = 15
    goal = np.array([0.,0.])
    clear = False
    Sref = []
    ref_flag = False

    def __init__(self):
        
        rospy.Service('/plot/clear', Empty, self.callbackClear)
        rospy.Service('/plot/ref', pathTrackReq, self.callbackPlot)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)
        obs_srv = rospy.ServiceProxy('/getObservation', simulation_observation_srv)

        rospy.init_node('Plot_state', anonymous=True)
        rate = rospy.Rate(self.freq) # 15hz
        
        while 1:
            self.state = np.array(obs_srv().state)

            if self.counter % 100 == 0:
                if self.clear:
                    plt.gcf().clear()
                    self.clear = False
                if self.ref_flag:
                    plt.plot(self.Sref[:,0], self.Sref[:,1],'--k')
                    self.ref_flag = False

                plt.plot(self.state[0], self.state[1],'.r')
                # plt.plot(self.start_pos[0], self.start_pos[1],'*b')
                plt.plot(self.goal[0], self.goal[1],'*g')
                plt.axis("equal")
                plt.axis([-np.pi, np.pi, -np.pi, np.pi])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Real-time object position:' + str(self.state))
                plt.draw()
                plt.pause(0.00000000001)

            self.counter += 1

            if self.counter <= 5:
                self.start_pos = self.state

    def callbackClear(self, msg):
        self.clear = True
        return EmptyResponse()

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def callbackPlot(self, req):
        self.Sref = np.array(req.desired_path).reshape(-1,4)
        self.ref_flag = True

        return {'real_path': [], 'success' : True}

    


if __name__ == '__main__':
    
    try:
        Plot()
    except rospy.ROSInterruptException:
        pass