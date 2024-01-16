#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from std_srvs.srv import SetBool
from vaccum_gripper import MinimalClientAsync

class AgrobotControl(Node):
    def __init__(self):
        super().__init__('agrobot_controller')

        self.joint_position_pub = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.minimal_client = MinimalClientAsync()
        self.goal_position = [900.0, 450, 100.5, 0.0, 0.00, 0.5]  # Goal pose
        # Initial point (UR10 home position)
        self.home_pos = [0.01, 356.1, 1427.99, 0.0, 0.0, 0.0]

        self.drop_position = [0.05, 358.0, 500, 0.00, 0.0, 0.0]  # Goal pose

        self.inverse_kinematics()

        th1, th2, th3, th4, th5, th6 = symbols(
            'theta1 theta2 theta3 theta4 theta5 theta6', real=True)

    """To compute the DH parameters matrix"""
    # to compute transformation matrix
    def Transformation_matrix(self, a, al, d, th):
        return Matrix([
            [cos(th),   -sin(th) * cos(al),
                sin(th) * sin(al), a * cos(th)],
            [sin(th),    cos(th) * cos(al),   -
                cos(th) * sin(al), a * sin(th)],
            [0,   sin(al),   cos(al), d],
            [0, 0, 0, 1]
        ])

    # To compute jacobian matrix
    def cal_jacobian(self, th1, th2, th3, th4, th5, th6):
        T0 = self.Transformation_matrix(0, rad(0), 0, rad(0))
        T1 = self.Transformation_matrix(0, rad(90), 128, th1+rad(180))
        T2 = self.Transformation_matrix(-612.7, rad(180), 0, th2-rad(90))
        T3 = self.Transformation_matrix(-571.6, rad(180), 0, th3)
        T4 = self.Transformation_matrix(0, rad(-90), 163.9, th4 + rad(90))
        T5 = self.Transformation_matrix(0, rad(90), 115.7, th5)
        T6 = self.Transformation_matrix(0, rad(0), 192.20, th6)
        T01 = (T1)
        T02 = (T01*T2)
        T03 = (T02*T3)
        T04 = (T03*T4)
        T05 = (T04*T5)
        T06 = (T05*T6)

        # To extract z vectors
        z0 = Matrix(T0[:, 2][:3])
        z1 = Matrix(T01[:, 2][:3])
        z2 = Matrix(T02[:, 2][:3])
        z3 = Matrix(T03[:, 2][:3])
        z4 = Matrix(T04[:, 2][:3])
        z5 = Matrix(T05[:, 2][:3])
        z6 = Matrix(T06[:, 2][:3])
        # To extract o vectors
        o0 = Matrix(T0[:, 3][:3])
        o1 = Matrix(T01[:, 3][:3])
        o2 = Matrix(T02[:, 3][:3])
        o3 = Matrix(T03[:, 3][:3])
        o4 = Matrix(T04[:, 3][:3])
        o5 = Matrix(T05[:, 3][:3])
        o6 = Matrix(T06[:, 3][:3])
        # To compute jacobian matrix
        J1 = Matrix([z0.cross(o6-o0), z0])
        J2 = Matrix([z1.cross(o6 - o1), z1])
        J3 = Matrix([z2.cross(o6 - o2), z2])
        J4 = Matrix([z3.cross(o6 - o3), z3])
        J5 = Matrix([z4.cross(o6 - o4), z4])
        J6 = Matrix([z5.cross(o6 - o5), z5])
        J = J1.row_join(J2).row_join(J3).row_join(
            J4).row_join(J5).row_join(J6)
        # pprint(J)
        return J, T0, T01, T02, T03, T04, T05, T06
    # To check for singularities
    def check_singularities(self, q1, q2, q3, q4, q5, q6):
        J, T0, T01, T02, T03, T04, T05, T06 = self.cal_jacobian(
            q1, q2, q3, q4, q5, q6)
        J_float = np.matrix(J).astype(np.float64)
        det = np.linalg.det(J_float)
        if abs(det) < 0.001:
            print("Singularities reached!")
            q1 += 0.002
            q2 += 0.002
            q3 += 0.002
            q4 += 0.002
            q5 += 0.002
            q6 += 0.002
            # incrementing joint angles
            return q1, q2, q3, q4, q5, q6
        else:
            return q1, q2, q3, q4, q5, q6
    # To compute inverse kinematics
    def inverse_kinematics(self):
        # to print the jacobian matrix initially when all joint angles are zero
        print("The initial Jacobian when robot at home position:")
        J, T0, T01, T02, T03, T04, T05, T06 = self.cal_jacobian(
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
        pprint(J)
        print("The Final transformation matrix when robot at home position:")
        pprint(T06)
        J_float = np.matrix(J).astype(np.float64)
        J_inv = np.linalg.inv(J_float)
        dt = 0.1  # change this to 0.01 to get smooth trajectory
        it = 200  # number of iterations
        i = 0
        trajectory_points = []
        q = np.zeros((6, 1))
        # linear velocities along x,y,z direction
        x_dot = ((self.goal_position[0]-self.home_pos[0])/it)
        y_dot = ((self.goal_position[1]-self.home_pos[1])/it)
        z_dot = ((self.goal_position[2]-self.home_pos[2])/it)
        # angular velocities along x,y,z directions
        wx = (self.goal_position[3]-self.home_pos[3])/it
        wy = (self.goal_position[4]-self.home_pos[4])/it
        wz = (self.goal_position[5]-self.home_pos[5])/it
        print("\nComputing to generate trajectory points....")
        print("please wait....")
        # This loop iterates and increments the joint angles
        while (i < it):
            # epsilon matrix which consists of linear and angular velocities
            E = np.matrix([[x_dot], [y_dot], [z_dot], [wx], [wy], [wz]])
            q_dot = (J_inv*E)  # computing velocities of joint angles
            q += q_dot*dt  # incrementing joint angles

            [q1, q2, q3, q4, q5, q6] = [q[i].item() for i in range(6)]
            # print("Joint angles: ", q1, q2, q3, q4, q5, q6)
            q1_updated, q2_updated, q3_updated, q4_updated, q5_updated, q6_updated = self.check_singularities(
                q1, q2, q3, q4, q5, q6)
            # print(q1_updated, q2_updated, q3_updated,
                #   q4_updated, q5_updated, q6_updated)

            # computing jacobian matrix with latest joint angles
            J, T0, T01, T02, T03, T04, T05, T06 = self.cal_jacobian(
                q1_updated, q2_updated, q3_updated, q4_updated, q5_updated, q6_updated)
            J_float = np.matrix(J).astype(np.float64)
            J_inv = np.linalg.inv(J_float)
            # computing the final transformation matrix with latest joint angles
            T = T06
            # pprint(T)
            x = T[0, 3]
            y = T[1, 3]
            z = T[2, 3]
            trajectory_points.append((x, y, z))
            # print("x: ", x, "y: ", y, "z: ", z)
            i = i+dt
            i = round(i, 3)
            # Publish the computed joint angles to the robot
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0, q1_updated, q2_updated,
                        q3_updated, q4_updated, q5_updated, q6_updated]

            self.joint_position_pub.publish(msg)
            # Check if the end effector is close to the target position
            if (i > 199.8):
                print("Goal position reached!")
                print("----------------------------------")
                print("Starting the vacuum gripper")
                #start the vacuum gripper
                status=True
                response = self.minimal_client.send_request(status)
                print("----------------------------------")
                print("Moving to drop position")
                temp = self.home_pos
                self.home_pos = self.goal_position #change the home position to goal position
                self.goal_position = self.drop_position #change the goal position to drop position
                x_dot = ((self.goal_position[0]-self.home_pos[0])/it)
                y_dot = ((self.goal_position[1]-self.home_pos[1])/it)
                z_dot = ((self.goal_position[2]-self.home_pos[2])/it)
                # angular velocities along x,y,z directions
                wx = (self.goal_position[3]-self.home_pos[3])/it
                wy = (self.goal_position[4]-self.home_pos[4])/it
                wz = (self.goal_position[5]-self.home_pos[5])/it
                i = 0
                while (i < it):
                    # epsilon matrix which consists of linear and angular velocities
                    E = np.matrix(
                        [[x_dot], [y_dot], [z_dot], [wx], [wy], [wz]])
                    q_dot = (J_inv*E)  # computing velocities of joint angles
                    q += q_dot*dt  # incrementing joint angles

                    [q1, q2, q3, q4, q5, q6] = [q[i].item() for i in range(6)]
                    # print("Joint angles: ", q1, q2, q3, q4, q5, q6)
                    q1_updated, q2_updated, q3_updated, q4_updated, q5_updated, q6_updated = self.check_singularities(
                        q1, q2, q3, q4, q5, q6)
                    # print(q1_updated, q2_updated, q3_updated,
                    #       q4_updated, q5_updated, q6_updated)
                    # computing jacobian matrix with latest joint angles
                    J, T0, T01, T02, T03, T04, T05, T06 = self.cal_jacobian(
                        q1_updated, q2_updated, q3_updated, q4_updated, q5_updated, q6_updated)
                    J_float = np.matrix(J).astype(np.float64)
                    J_inv = np.linalg.inv(J_float)
                    # computing the final transformation matrix with latest joint angles
                    T = T06
                    # pprint(T)
                    x = T[0, 3]
                    y = T[1, 3]
                    z = T[2, 3]
                    trajectory_points.append((x, y, z))
                    # print("x: ", x, "y: ", y, "z: ", z)
                    i = i+dt
                    i = round(i, 3)
                    # Publish the computed joint angles to the robot
                    msg = Float64MultiArray()
                    msg.data = [0.0, 0.0, q1_updated, q2_updated,
                                q3_updated, q4_updated, q5_updated, q6_updated]
                    self.joint_position_pub.publish(msg)
        #stop the vacuum gripper  
        print("Reached the drop position")
        print("----------------------------------")
        print("Stopping the vacuum gripper")
        print("----------------------------------")
        print("Target position reached!")
        status=False
        response = self.minimal_client.send_request(status)
                   
         # plot in 3D for end effector trajectory
        trajectory_points = np.array(trajectory_points)
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        ax.plot(trajectory_points[:, 0],
                trajectory_points[:, 1],
                trajectory_points[:, 2])
        ax.set_title('End-Effector Trajectory')
        ax.set_xlabel('X-axis', labelpad=20)
        ax.set_ylabel('Y-axis', labelpad=20)
        ax.set_zlabel('Z-axis', labelpad=20)
        ax.set_ylim(0.01, 400.0)
        plt.show()

        # Plot the end-effector trajectory in 2D
        trajectory_points = np.array(trajectory_points)
        fig = plt.figure(figsize=(8, 8))
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 2])
        plt.title('End-Effector Trajectory-2D plot')
        plt.xlabel('X-axis')
        plt.ylabel('Z-axis')
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    agrobot_control = AgrobotControl()
    rclpy.spin(agrobot_control)
    agrobot_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
