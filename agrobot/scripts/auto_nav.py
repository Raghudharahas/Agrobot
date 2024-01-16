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


class MobilebotControl(Node):
    def __init__(self):
        super().__init__('mobilebot_controller')

        self.joint_position_pub = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(
            Float64MultiArray, '/velocity_controller/commands', 10)

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )       
        self.timer = self.create_timer(1.0, self.publish_vel_data)
        self.i=0
    def publish_vel_data(self):
        self.i=self.i+1
        msg = Float64MultiArray()
        # Set the joint positions here
        msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0]
        self.joint_position_pub.publish(msg)
        vel = Float64MultiArray()
        if(self.i<=8):
            vel.data = [-2.0, 2.0, 2.0, -2.0]
            self.wheel_velocities_pub.publish(vel)
        else:
            vel.data = [0.0, 0.0, 0.0, 0.0]
            self.wheel_velocities_pub.publish(vel)
            rclpy.shutdown()
        
        

def main(args=None):
    rclpy.init(args=args)
    mobilebot_control = MobilebotControl()
    rclpy.spin(mobilebot_control)
    mobilebot_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
