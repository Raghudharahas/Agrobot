#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from geometry_msgs.msg import Vector3
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


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

        self.timer = self.create_timer(1.0, self.publish_joint_positions)

    def publish_joint_positions(self):
        msg = Float64MultiArray()
        th1, th2, th3, th4, th5, th6 = symbols(
            'theta1 theta2 theta3 theta4 theta5 theta6', real=True)

        th1, th2, th3, th4, th5, th6 = 0, 0, 0, 0, 0, 0

        """To compute the DH parameters matrix"""

        def Transformation_matrix(a, al, d, th):
            return Matrix([
                [cos(th),   -sin(th) * cos(al),
                 sin(th) * sin(al), a * cos(th)],
                [sin(th),    cos(th) * cos(al),   -
                 cos(th) * sin(al), a * sin(th)],
                [0,   sin(al),   cos(al), d],
                [0, 0, 0, 1]
            ])

        # To compute jacobian matrix

        def cal_jacobian(th1, th2, th3, th4, th5, th6):
            T0 = Transformation_matrix(0, rad(0), 0, rad(0))
            T1 = Transformation_matrix(0, rad(90), 128, th1+rad(180))
            T2 = Transformation_matrix(-612.7, rad(180), 0, th2-rad(90))
            T3 = Transformation_matrix(-571.6, rad(180), 0, th3)
            T4 = Transformation_matrix(0, rad(-90), 163.9, th4 + rad(90))
            T5 = Transformation_matrix(0, rad(90), 115.7, th5)
            T6 = Transformation_matrix(0, rad(0), 192.2, th6)
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
            return J

        r = 100  # radius of circle to be drawn
        dt = 0.01  # change this to 0.001 to get smooth curve plot
        it = 20
        i = 0
        trajectory_points = []
        q = np.zeros((6, 1))
        phi_dot = (2*np.pi)/20

        # to print the jacobian matrix initially when all joint angles are zero
        J = cal_jacobian(0, 0, 0, 0, 0, 0)
        print("The initial Jacobian when robot at home position:")
        pprint(J)

        J_float = np.matrix(J).astype(np.float64)
        J_inv = np.linalg.pinv(J_float)
        print("\nComputing to generate trajectory points....")
        # This loop iterates and increments the joint angles
        while (i < it):
            phi = phi_dot*i
            # linear velocities along x,y,z direction
            x_dot = r*(np.cos(phi))*phi_dot
            y_dot = 0.0
            z_dot = -(r*(np.sin(phi))*phi_dot)
            wx, wy, wz = 0, 0, 0  # angular velocities along x,y,z directions
            # epsilon matrix which consists of linear and angular velocities
            E = np.matrix([[x_dot], [y_dot], [z_dot], [wx], [wy], [wz]])
            q_dot = J_inv*E  # computing velocities of joint angles
            q += q_dot*dt  # incrementing joint angles
            [q1, q2, q3, q4, q5, q6] = [q[i].item() for i in range(6)]
            J = cal_jacobian(q1, q2, q3, q4, q5, q6)
            J_float = np.matrix(J).astype(np.float64)
            J_inv = np.linalg.pinv(J_float)

            # computing the final transformation matrix with latest joint angles
            T0 = Transformation_matrix(0, rad(0), 0, rad(0))
            T1 = Transformation_matrix(0, rad(90), 128, q1+rad(180))
            T2 = Transformation_matrix(-612.7, rad(180), 0, q2-rad(90))
            T3 = Transformation_matrix(-571.6, rad(180), 0, q3)
            T4 = Transformation_matrix(0, rad(-90), 163.9, q4 + rad(90))
            T5 = Transformation_matrix(0, rad(90), 115.7, q5)
            T6 = Transformation_matrix(0, rad(0), 192.2, q6)
            T = T0*T1*T2*T3*T4*T5*T6
            # pprint(T)
            x = T[0, 3]
            y = T[1, 3]
            z = T[2, 3]
            trajectory_points.append((x, y, z))
            i = i+dt
            # print(i)
            # Publish the computed joint angles to the robot
            msg.data = [0.0,0.0,q1, q2, q3, q4, q5, q6]
            self.joint_position_pub.publish(msg)
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
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    agrobot_control = AgrobotControl()
    rclpy.spin(agrobot_control)
    agrobot_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
