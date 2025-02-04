import numpy as np
import time
from simulator import MultiRobotSimulator



if __name__ == "__main__":
    # Parameters
    num_agents = 4
    control_limit = 0.2  # Maximum acceleration
    safety_distance = 1.0  # Minimum safe distance
    steps = 200  # Number of simulation steps
    dt = 0.1  # Time step
    np.random.seed(31)
    velocity_limit = 1.0  # Maximum velocity

    # Run simulation
    simulator = MultiRobotSimulator(num_agents, control_limit, velocity_limit, safety_distance, dt, random=True)

    
    simulator.simulate(steps)
