ENPM 662: Project 2
Project title: Modelling & Gazebo Simulation of AgroBot
Package Name: agrobot

#Dependencies:
-ROS Controller package and controller manager
```bash
sudo apt install ros-galactic-ros2-control ros-galactic-ros2-controllers ros-galactic-gazebo-ros2-control
sudo apt-get install ros-galactic-controller-manager

#Steps to execute the project:
1.Create a ROS2 workspace.
- Add the package: Unzip the project files and place them under the src folder in your ROS 2 workspace.

2.Build and Source: Source your bashrc and ROS 2 distribution, build the package, and source your workspace:
```bash
source /opt/ros/galactic/setup.bash
colcon build
source /path/to/your/workspace/install/setup.bash

3. Copy the models to your ./gazebo/models folder
   Go to src/agrobot/models folder in the package and copy them to ./gazebo/models using the following command:

cp -r ground_plane mud_box oak_tree pine_tree plant plant_2 static_tomato1 tomato tomato_new/ ~/.gazebo/models
(make sure to execute this to get the custom world)

4. Build and source 
a) Launch the gazebo world- ""ros2 launch agrobot gazebo.launch.py""

b) In the new terminal- execute this command to start the auto navgation-""ros2 run agrobot auto_nav.py""
(once the execution of this ends then proceed to next step)

c)open new terminal and run- "ros2 run agrobot ur10_final.py " (make sure to launch gazebo before this command)

5.execute the following for rviz and inverse kinematics validation:
 follow previous steps
1. Rviz visualisation: "ros2 launch agrobot display.launch.py"

2. Inverse_Validation : "ros2 run agrobot inverse_validation.py"

3. teleop: to do teleop for the mobile robot only-"ros2 run agrobot teleop.py"


#Note:
The zip folders submitted by either one of us has the same contents.

#Maintainers

-Raghu Dharahas Reddy Kotla (raghu17@umd.edu)
-Sai Dinesh Gelam (sgelam@umd.edu)
