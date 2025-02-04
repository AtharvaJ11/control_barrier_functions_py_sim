# Python Simulator for Control Barrier function verification.

 The repo is for validating Control barrier function algorithms on Double integrator kinematics robots.


### Scripts & Functionalities

1. cbf.py - Contains the baseline Centralized Control Barrier Fucntion Optimization solution class
2. softmin_cbf.py - Contains the softmin Centralized Control Barrier Fucntion Optimization solution class
3. simulator.py - Contains the Double integrator Kinematics simulator class (contains nominal PID controller)
4. main.py - Main function to run a config of Optimzation and simulator
5. configs - Contains the start and goal postions of agents, named as swarm_{i}.json where i is the swarm size