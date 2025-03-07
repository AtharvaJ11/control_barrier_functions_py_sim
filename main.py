import numpy as np
import time
from simulator import MultiRobotSimulator
from cbf import CentralizedControlBarrierFunction
from softmin_cbf import SoftminCentralizedControlBarrierFunction
import argparse

def main(optimizer_type, num_agents, spawn):
    # Parameters
    control_limit = 0.2  # Maximum acceleration
    safety_distance = 1.0  # Minimum safe distance
    steps = 1000  # Max Number of simulation steps
    dt = 0.1  # Time step

    np.random.seed(31)
    
    velocity_limit = 1.0  # Maximum velocity

    if optimizer_type == "centralized":
        optimizer = CentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
    elif optimizer_type == "softmin":
        optimizer = SoftminCentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
    else:
        optimizer = None # Naive PID control with no barrier functions for safety

    # Run simulation
    simulator = MultiRobotSimulator(num_agents, optimizer, control_limit, velocity_limit, safety_distance, dt, spawn)
    simulator.simulate(steps, plot_on=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-robot simulation with CBF")
    parser.add_argument("--optimizer", type=str, choices=["centralized", "softmin"], required=False, help="Type of optimizer to use")
    parser.add_argument("--num_agents", type=int, required=True, help="Number of agents in the simulation")
    parser.add_argument("--spawn", type=str, default="random", choices=["circle", "random", "custom"], help="Use random seed (True or False)")
    args = parser.parse_args()

    main(args.optimizer, args.num_agents, args.spawn)
