# Pick and Place using state estimation from a 3D pointcloud and ROS2 

This project implements a pick and place task using a Franka Emika Panda robot. We use ROS2 Jazzy to interface with the robot and MuJoCo for simulation. The system takes in a 3D pointcloud of the scene, estimates the state of the objects, and executes a pick and place operation.

System Requirements: Ubuntu 24.04 and ROS2 Jazzy

## How to Use
First install the virtual environment using the requirements.txt file:
'''
uv install requirements.txt
'''
We can then build the package using the following alias:
'''
build =
'''
Using an alias makes it easier to build everytime instead of using the same commands

To start the simulation, run the following command inside your workspace:
'''
ros2 run franka_mujoco start_sim
'''
This command launches the simulation




