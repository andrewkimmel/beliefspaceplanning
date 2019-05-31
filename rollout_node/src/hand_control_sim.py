#!/usr/bin/python 

import rospy
import numpy as np 
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import TargetAngles, IsDropped, observation, MoveServos
import math

state_form = 'pos_load_vel' # 'pos_load' or 'pos_vel' or 'pos_load_vel' or 'pos_load_vel_joints' or 'pos_load_joints' or 'all'

class hand_control():

    finger_opening_position = np.array([0.005, 0.005])
    finger_closing_position = np.array([0.08, 0.08])
    finger_move_step_size = np.array([0.00015, 0.00015])
    closed_load = np.array(20.)

    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    gripper_load_prev = np.array([0., 0.])
    joint_states = np.array([0., 0., 0., 0.])
    joint_velocities = np.array([0., 0., 0., 0.])
    lift_status = False # True - lift up, False - lift down
    object_grasped = False
    base_pos = [0.,0.]
    base_theta = 0
    obj_pos = [0.,0.]
    obj_vel = [0.,0.]
    D_load = np.array([0., 0.])
    R = []
    count = 1
    OBS = True

    gripper_status = 'open'

    move_servos_srv = 0.

    def __init__(self):
        rospy.init_node('hand_control_sim', anonymous=True)
        
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/gripper/lift_status', String, self.callbackLiftStatus)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand/obj_vel', Float32MultiArray, self.callbackObjVel)
        rospy.Subscriber('/hand/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/hand/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)
        pub_gripper_status = rospy.Publisher('/hand_control/gripper_status', String, queue_size=10)
        pub_drop = rospy.Publisher('/hand_control/cylinder_drop', Bool, queue_size=10)

        rospy.Service('/hand_control/ResetGripper', Empty, self.ResetGripper)
        rospy.Service('/hand_control/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/hand_control/IsObjDropped', IsDropped, self.CheckDropped)
        rospy.Service('/hand_control/observation', observation, self.GetObservation)

        self.move_servos_srv = rospy.ServiceProxy('/MoveServos', MoveServos)
        self.move_lift_srv = rospy.ServiceProxy('/LiftHand', Empty)
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        #### Later I should remove the angles from hands.py and set initial angles here at the start ####

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            pub_gripper_status.publish(self.gripper_status)
            pub_drop.publish(not self.object_grasped)

            self.D_load = np.copy(self.gripper_load) - np.copy(self.gripper_load_prev)
            self.gripper_load_prev = np.copy(self.gripper_load)

            # print(self.obj_pos)
            # rospy.spin()
            rate.sleep()

    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

    def callbackLiftStatus(self, msg):
        self.lift_status = True if msg.data == 'up' else False

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)

        # b = 0.5e-3 # For screwdriver scenario
        b = -0.1 # For regular cylinder scenario

        if Obj_pos[2] < b or (self.joint_states[0] > 1.5 and self.joint_states[2] > 1.5) or self.joint_states[0] > 2.5 or self.joint_states[2] > 2.5: 
            self.object_grasped = False
        else:
            self.object_grasped = True
        # self.object_grasped = True if abs(Obj_pos[2]) < 1e-2 else False

        self.obj_pos = Obj_pos[:2]*1000 # m to mm

    def callbackObjVel(self, msg):
        Obj_vel = np.array(msg.data)

        self.obj_vel = Obj_vel[:2]*1000 # m/s to mm/s

    def ResetGripper(self, msg):
        ratein = rospy.Rate(15)
        print('[hand_control_sim] Resetting gripper...')
        while 1:
            # Open gripper
            self.moveGripper(self.finger_opening_position)
            rospy.sleep(.5)
            # Drop lift down
            while self.lift_status:
                self.move_lift_srv.call()
                rospy.sleep(.1)
                ratein.sleep()
            # Reset object
            rospy.sleep(.2)
            self.reset_srv.call()
            # Close gripper
            no = np.random.random(2)*2. - 1.
            self.moveGripper(self.finger_closing_position+no/1000.)
            rospy.sleep(0.7)
            self.move_lift_srv.call()
            rospy.sleep(2.0)
            ratein.sleep()
            if self.object_grasped:# and self.lift_s, self.obj_veltatus:
                if self.wait2initialGrasp():
                    break
        
        self.gripper_status = 'closed'

        return EmptyResponse()

    def wait2initialGrasp(self):
        # This function waits for grasp to be stable (static) in its initial pose

        print('[hand_control_sim] Waiting for hand to stabilize...')
        ratein = rospy.Rate(15)
        c = 0
        suc = False
        while c < 150:
            c += 1
            # print('Load' + str(self.gripper_load))
            # if self.gripper_load[0]!=self.gripper_load[1]: # equal when stable in the initial position
            #     rospy.sleep(0.5)
            #     ratein.sleep()
            #     continue

            pos1 = self.obj_pos
            rospy.sleep(0.2)
            ratein.sleep()
            # print(pos1, self.obj_pos, np.linalg.norm(pos1 - self.obj_pos))
            if np.linalg.norm(pos1 - self.obj_pos) < 2e-2:
                suc = True
                print('[hand_control_sim] Hand stable.')
                break

        return suc

    def MoveGripper(self, msg):
        # This function should accept a vector of normalized incraments to the current angles: msg.angles = [dq1, dq2], where dq1 and dq2 can be equal to 0 (no move), 1,-1 (increase or decrease angles by finger_move_step_size)

        inc = np.array(msg.angles)
        inc_angles = np.multiply(self.finger_move_step_size, inc)

        desired = self.gripper_pos + inc_angles

        suc = self.moveGripper(desired)

        return {'success': suc}
    
    def moveGripper(self, angles):

        if angles[0] > 0.7 or angles[1] > 0.7 or angles[0] < 0.003 or angles[1] < 0.003:
            rospy.logerr('[hand_control_sim] Desired angles out of bounds.')
            return False

        if abs(self.gripper_load[0]) > 120 or abs(self.gripper_load[1]) > 120:
            rospy.logerr('[hand_control_sim] Pre-overload.')
            return False

        self.move_servos_srv.call(angles)

        Obs = np.array([[-38, 117.1, 4.],
        # [-33., 105., 4.],
        [-33., 106.2, 4.],
        [-52.5, 105.2, 4.],
        [-51., 105.5, 4.],
        [43., 111.5, 6.],
        [59., 80., 3.],
        [36.5, 94., 4.]
        ])

        # Obs = np.array([[-38, 117.1, 4.],
        #     [-33., 105., 4.],
        #     [-52.5, 105.2, 4.],
        #     [43., 111.5, 6.],
        #     [59., 80., 3.],
        #     [36.5, 94., 4.]
        # ])
        if self.OBS:
            for obs in Obs:
                if np.linalg.norm(self.obj_pos-obs[:2]) < obs[2]:
                    print('[hand_control_sim] Collision.')
                    return False

        return True

    def CheckDropped(self, msg):

        return {'dropped': not self.object_grasped}

    def GetObservation(self, msg):
        if state_form == 'all':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.joint_states, self.joint_velocities), axis=0)
        elif state_form == 'pos_load':
            obs = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
        elif state_form == 'pos_vel':   
            obs = np.concatenate((self.obj_pos, self.obj_vel), axis=0)
        elif state_form == 'pos_load_vel':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, self.D_load), axis=0)
        elif state_form == 'pos_load_vel_joints':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, self.joint_states), axis=0)
        elif state_form == 'pos_load_joints':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.joint_states), axis=0)

        # print obs
        return {'state': obs}


if __name__ == '__main__':
    
    try:
        hand_control()
    except rospy.ROSInterruptException:
        pass