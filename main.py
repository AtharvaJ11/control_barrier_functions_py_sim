import numpy as np
import time
from simulator import MultiRobotSimulator
from cbf import CentralizedControlBarrierFunction
from softmin_cbf import SoftminCentralizedControlBarrierFunction


if __name__ == "__main__":
    # Parameters
    num_agents = 4
    control_limit = 0.2  # Maximum acceleration
    safety_distance = 1.0  # Minimum safe distance
    steps = 300  # Max Number of simulation steps
    dt = 0.1  # Time step
    np.random.seed(31)
    velocity_limit = 1.0  # Maximum velocity

    optimizer = CentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
    # optimizer= SoftminCentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
    # Run simulation
    simulator = MultiRobotSimulator(num_agents, optimizer, control_limit, velocity_limit, safety_distance, dt)

    
    simulator.simulate(steps)
