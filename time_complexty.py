import numpy as np
import time
from simulator import MultiRobotSimulator
from cbf import CentralizedControlBarrierFunction
from softmin_cbf import SoftminCentralizedControlBarrierFunction
import argparse
import json
import matplotlib.pyplot as plt

def main():
    # Parameters
    control_limit = 0.2  # Maximum acceleration
    safety_distance = 1.0  # Minimum safe distance
    steps = 1000  # Max Number of simulation steps
    dt = 0.1  # Time step
    velocity_limit = 1.0  # Maximum velocity
    spawn = "circle"
    optimizers = ["centralized", "softmin"]
    
    num_steps_results = {opt: [] for opt in optimizers}
    avg_time_results = {opt: [] for opt in optimizers}
    done_results = {opt: [] for opt in optimizers}
    
    for num_agents in range(2, 50):
        print()
        print()
        print(num_agents)
        for optimizer_type in optimizers:
            print()
            print()
            print(optimizer_type)
            if optimizer_type == "centralized":
                optimizer = CentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
            elif optimizer_type == "softmin":
                optimizer = SoftminCentralizedControlBarrierFunction(num_agents, control_limit, velocity_limit, safety_distance, dt)
            else:
                optimizer = None # Naive PID control with no barrier functions for safety

            # Run simulation
            simulator = MultiRobotSimulator(num_agents, optimizer, control_limit, velocity_limit, safety_distance, dt, spawn)
            num_steps, avg_time, done = simulator.simulate(steps, plot_on=False)
            
            num_steps_results[optimizer_type].append(num_steps)
            avg_time_results[optimizer_type].append(avg_time)
            done_results[optimizer_type].append(done)
    
    results = {
        "num_steps_results": num_steps_results,
        "avg_time_results": avg_time_results,
        "done_results": done_results
    }

    with open('simulation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    colors = {
        "centralized": "blue",
        "softmin": "green"
    }

    plt.figure()
    for optimizer_type in optimizers:
        for i, num_agents in enumerate(range(2, 50)):
            color = colors[optimizer_type]
            if done_results[optimizer_type][i]:
                plt.plot(num_agents, num_steps_results[optimizer_type][i], 'o', color=color, label=f'{optimizer_type} - num_steps' if i == 0 else "")
            else:
                plt.plot(num_agents, num_steps_results[optimizer_type][i], 'x', color=color, markersize=4, label=f'{optimizer_type} - num_steps (not done)' if i == 0 else "")
    plt.xlabel('Number of Agents')
    plt.ylabel('Number of Steps')
    plt.title('Number of Steps for All Optimizers')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for optimizer_type in optimizers:
        for i, num_agents in enumerate(range(2, 50)):
            color = colors[optimizer_type]
            if done_results[optimizer_type][i]:
                plt.plot(num_agents, avg_time_results[optimizer_type][i], 'o', color=color, label=f'{optimizer_type} - avg_time' if i == 0 else "")
            else:
                plt.plot(num_agents, avg_time_results[optimizer_type][i], 'x', color=color, markersize=4, label=f'{optimizer_type} - avg_time (not done)' if i == 0 else "")
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Time per Step')
    plt.title('Average Time per Step for All Optimizers')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
