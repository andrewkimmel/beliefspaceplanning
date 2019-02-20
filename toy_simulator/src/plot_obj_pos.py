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
    start_pos = np.array([0.,0.])
    freq = 50
    goal = np.array([0.,0.])
    clear = False
    memory = np.empty((0,2), float)
    memory_new = np.empty((0,2), float)

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

        self.memory_new = np.append(self.memory_new, [self.obj_pos], axis=0)

        if self.counter % 100 == 0 and self.memory.shape[0] > 0:
            # print(self.memory.shape)
            if self.clear:
                plt.gcf().clear()
                # plt.close()
                plt.figure(1)
                rospy.sleep(.1)
                self.clear = False
            plt.plot(self.memory[1:,0], self.memory[1:,1],'-r')
            plt.plot(self.memory[1,0], self.memory[1,1],'*b')
            circle=plt.Circle((self.memory[-1,0],self.memory[-1,1]),2)
            plt.plot(self.memory[-1,0], self.memory[-1,1],'*m')
            plt.plot(self.goal[0], self.goal[1],'*g')
            plt.axis("equal")
            plt.axis([-SIZE, SIZE, -SIZE, SIZE])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(b=True)
            plt.title('Real-time object position:' + str(self.obj_pos))
            plt.draw()
            plt.pause(0.00000000001)

        self.counter += 1

    def callbackClear(self, msg):
        # self.memory = np.empty((0,2), float)
        self.memory_new = np.empty((0,2), float)
        return EmptyResponse()

    def callbackPlot(self, msg):

        self.memory = np.empty((0,2), float)
        self.memory = np.copy(self.memory_new)
        # self.memory_new = np.empty((0,2), float)
        self.clear = True
 
        return EmptyResponse()

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def __init__(self):
        
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Service('/plot/clear', Empty, self.callbackClear)
        rospy.Service('/plot/plot', Empty, self.callbackPlot)

        rospy.init_node('Plot_obj_pos_pilco', anonymous=True)
        rate = rospy.Rate(self.freq) # 15hz
        while not rospy.is_shutdown():
            plt.show()
            # rospy.spin()
            rate.sleep()


if __name__ == '__main__':
    
    try:
        Plot()
    except rospy.ROSInterruptException:
        pass


# SIZE = 1.

# class Plot():

#     counter = 0
#     obj_pos = np.array([0.,0.])
#     freq = 100
#     clear = False
#     goal = np.array([0.,0.])

#     def callbackObj(self, msg):
#         self.obj_pos = np.array(msg.data)

#         if self.counter % 50 == 0:
#             if self.clear:
#                 plt.gcf().clear()
#                 self.clear = False
#             plt.plot(self.obj_pos[0], self.obj_pos[1],'.r')
#             plt.plot(self.goal[0], self.goal[1],'og')
#             plt.axis([-SIZE, SIZE, -SIZE, SIZE])
#             plt.axis('equal')
#             plt.xlabel('x')
#             plt.ylabel('y')
#             plt.title('Real-time object position:' + str(self.obj_pos))
#             plt.draw()
#             plt.pause(0.0001)

#         self.counter += 1
    
#     def callbackPlot(self, msg):
#         self.clear = True
#         return EmptyResponse()

#     def callbackGoal(self, msg):
#         self.goal = msg.data

#     def __init__(self):
        
#         rospy.Subscriber('/toy/obj_pos', Float32MultiArray, self.callbackObj)
#         rospy.Subscriber('/toy/goal', Float32MultiArray, self.callbackGoal)
#         rospy.Service('/plot/clear', Empty, self.callbackPlot)

#         rospy.init_node('Plot_obj_pos', anonymous=True)
#         rate = rospy.Rate(self.freq) # 15hz
#         while not rospy.is_shutdown():
#             plt.show()
#             rospy.spin()
#             # rate.sleep()

# if __name__ == '__main__':
    
#     try:
#         Plot()
#     except rospy.ROSInterruptException:
#         pass