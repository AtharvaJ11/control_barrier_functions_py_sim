import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
import time
import json

'''
Double Integrator Dynamics with Matplotlib Visualization
'''
class MultiRobotSimulator:
    def __init__(self, num_agents, optimizer, control_limit, velocity_limit, safety_distance, dt, random=False):
        self.num_agents = num_agents
        self.control_limit = control_limit
        self.safety_distance = safety_distance
        self.gamma =1.0  # Barrier function parameter
        self.dt = dt
        self.velocity_limit = velocity_limit
        self.neighbourhood_distance = self.safety_distance + (np.power(4*self.control_limit/self.gamma, 1/3)+ 2*self.velocity_limit)**2/(4*self.control_limit)
        self.optimizer = optimizer

        self.position_threshold = 0.1  # Position error threshold
        self.velocity_threshold = 0.1  # Velocity error threshold
        self.positions = np.random.rand(num_agents, 2) * 10  # Random initial positions
        self.velocities = np.zeros((num_agents, 2))  # Initial velocities
        self.goal_positions = np.zeros((num_agents, 2))  # Initialize goal positions
        self.goal_velocities = np.zeros((num_agents, 2))  # Target velocities
        self.kp = 0.6  # Proportional gain
        self.kd = 2 * np.sqrt(self.kp)  # Derivative gain
        self.trajectories = [[] for _ in range(num_agents)]  # Store agent trajectories
        
        self.step_time_list = []
        
        json_file =f"configs/swarm_{self.num_agents}.json"
        if random:
            self.generate_points()
        else:
            self.load_positions_from_json(json_file)



    def load_positions_from_json(self, json_file):
        """ Load initial and goal positions from a JSON file """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.positions = np.array(data["positions"])
        self.goal_positions = np.array(data["goal_positions"])

    def generate_points(self,):
        # Generate starting positions at least 2.5x safety_distance apart
        for i in range(self.num_agents):
            while True:
                candidate_position = np.random.rand(2) * 10 - 10.0
                if all(np.linalg.norm(candidate_position - self.positions[j]) >= 2.0 * self.safety_distance
                    for j in range(i)):
                    self.positions[i] = candidate_position
                    break

        # Generate goal positions at least 2.5x safety_distance away from starting positions and each other
        for i in range(self.num_agents):
            while True:
                candidate_goal = np.random.rand(2) * 10 -10.0
                if (np.linalg.norm(candidate_goal - self.positions[i]) >= 1.5 * self.safety_distance and
                    all(np.linalg.norm(candidate_goal - self.goal_positions[j]) >= 3. * self.safety_distance
                        for j in range(i))):
                    self.goal_positions[i] = candidate_goal
                    break

    def pid_controller(self):
        """Generate nominal control inputs using a PD controller."""
        nominal_controls = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            error = self.goal_positions[i] - self.positions[i]
            derivative = self.goal_velocities[i] - self.velocities[i]
            nominal_control = self.kp * error + self.kd * derivative
            nominal_controls[i] = np.clip(nominal_control, -self.velocity_limit, self.velocity_limit)


        return nominal_controls

    def dynamics(self, nominal_controls):
        """Update positions and velocities using Euler integration."""
        self.positions += self.velocities * self.dt
        self.velocities += nominal_controls * self.dt

        # Append current positions to trajectories
        for i in range(self.num_agents):
            self.trajectories[i].append(self.positions[i].copy())

    """
    Plotting functions & simulte
    """
    def check_collision(self):
        """Check if any agents are within the safety distance."""
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                if distance <= self.safety_distance:
                    print(f"Collision detected between Agent {i+1} and Agent {j+1}")
                    return True
        return False


    '''
    simulate function
    '''
    def simulate(self, steps):
        """Run the simulation for a given number of steps."""
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.jet(np.linspace(0, 1, self.num_agents))

        for step in range(steps):
            # Get nominal controls from PID controller
            nominal_controls = self.pid_controller()
            
            
            start_time = time.time()
            
            if self.optimizer is None:
                controls = nominal_controls
            else:
                # Get optimized controls using centralized control barrier QP
                controls = self.optimizer.centralized_control_barrier_qp(nominal_controls, self.positions, self.velocities)
            
            end_time = time.time()
            step_time = end_time - start_time
            self.step_time_list.append(step_time)
            print(f"Step {step} took {step_time:.4f} seconds")

            # Check for collisions before updating dynamics
            if self.check_collision():
                print("Simulation stopped due to safety violation.")
                break

            # Run dynamics with the optimized controls
            self.dynamics(controls)

            # Check if all agents reached their goals
            position_error = np.linalg.norm(self.positions - self.goal_positions, axis=1)
            velocity_error = np.linalg.norm(self.velocities - self.goal_velocities, axis=1)

            if np.all(position_error < self.position_threshold) and np.all(velocity_error < self.velocity_threshold):
                print("All agents reached their goal positions and velocities. Simulation stopped.")
                break

            # Visualization of agent positions and dynamics
            ax.clear()
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_title(f"Step {step}")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Visualize each agent
            for i, color in enumerate(colors):
                # Draw agent as a circle representing its safety zone
                circle = Circle(self.positions[i], 0.5*self.safety_distance, color=color, alpha=0.3)
                ax.add_patch(circle)

                # Draw agent position
                ax.plot(self.positions[i, 0], self.positions[i, 1], 'o', color=color, label=f"Agent {i+1}")

                # Draw goal positions
                ax.plot(self.goal_positions[i, 0], self.goal_positions[i, 1], 'x', color=color, markersize=10)

                # Draw velocity arrows
                ax.arrow(self.positions[i, 0], self.positions[i, 1],
                        self.velocities[i, 0] * 0.5, self.velocities[i, 1] * 0.5,
                        color=color, head_width=0.2, head_length=0.3)

            ax.legend(loc="upper right")
            plt.draw()
            plt.pause(0.1)  # Pause for visualization

        plt.show()  # Show the figure and wait until it is closed

        # Plot final trajectories
        self.plot_trajectories(colors)
        self.plot_step_time()


    def plot_step_time(self):
        """Plot the time taken for each simulation step."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.step_time_list, 'o-', color='b')
        plt.title("Time Taken for Each Simulation Step")
        plt.xlabel("Step Number")
        plt.ylabel("Time (s)")
        plt.show()
        plt.close('all')

    def plot_trajectories(self, colors):
        """Plot the trajectories of all agents."""
        plt.figure(figsize=(8, 8))
        plt.title("Trajectories of All Agents")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

        for i, color in enumerate(colors):
            trajectory = np.array(self.trajectories[i])
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, label=f"Agent {i+1}")
            plt.plot(self.goal_positions[i, 0], self.goal_positions[i, 1], 'x', color=color, markersize=10)

        plt.legend(loc="upper right")
        plt.show()
        # plt.close('all')  # Close all figures after plotting




