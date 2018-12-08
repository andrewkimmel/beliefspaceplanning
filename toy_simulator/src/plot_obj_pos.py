#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from action_node.srv import empty
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse


SIZE = 1.

class Plot():

    counter = 0
    obj_pos = np.array([0.,0.])
    freq = 100
    clear = False
    goal = np.array([0.,0.])

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

        if self.counter % 50 == 0:
            if self.clear:
                plt.gcf().clear()
                self.clear = False
            plt.plot(self.obj_pos[0], self.obj_pos[1],'.r')
            plt.plot(self.goal[0], self.goal[1],'og')
            plt.axis([-SIZE, SIZE, -SIZE, SIZE])
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Real-time object position:' + str(self.obj_pos))
            plt.draw()
            plt.pause(0.0001)

        self.counter += 1
    
    def callbackPlot(self, msg):
        self.clear = True
        return EmptyResponse()

    def callbackGoal(self, msg):
        self.goal = msg.data

    def __init__(self):
        
        rospy.Subscriber('/toy/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/toy/goal', Float32MultiArray, self.callbackGoal)
        rospy.Service('/plot/clear', Empty, self.callbackPlot)

        rospy.init_node('Plot_obj_pos', anonymous=True)
        rate = rospy.Rate(self.freq) # 15hz
        while not rospy.is_shutdown():
            plt.show()
            rospy.spin()
            # rate.sleep()

if __name__ == '__main__':
    
    try:
        Plot()
    except rospy.ROSInterruptException:
        pass