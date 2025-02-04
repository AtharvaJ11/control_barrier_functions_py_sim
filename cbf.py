import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from scipy.optimize import minimize
import cvxpy as cp
import time

class CentralizedControlBarrierFunction:
    def __init__(self, num_agents, control_limit, velocity_limit, safety_distance, dt, random=False):
        self.num_agents = num_agents
        self.control_limit = control_limit
        self.safety_distance = safety_distance
        self.gamma =1.0  # Barrier function parameter
        self.dt = dt
        self.velocity_limit = velocity_limit
        self.neighbourhood_distance = self.safety_distance + (np.power(4*self.control_limit/self.gamma, 1/3)+ 2*self.velocity_limit)**2/(4*self.control_limit)


        self.positions = np.random.rand(num_agents, 2) * 10  # Random initial positions
        self.velocities = np.zeros((num_agents, 2))  # Initial velocities
        self.goal_positions = np.zeros((num_agents, 2))  # Initialize goal positions
        self.goal_velocities = np.zeros((num_agents, 2))  # Target velocities
        self.kp = 0.6  # Proportional gain
        self.kd = 2 * np.sqrt(self.kp)  # Derivative gain
        self.trajectories = [[] for _ in range(num_agents)]  # Store agent trajectories
        if random:
            self.generate_points()

        print(self.positions)


    def generate_points(self,):
        # Generate starting positions at least 2.5x safety_distance apart
        for i in range(self.num_agents):
            while True:
                candidate_position = np.random.rand(2) * 10
                if all(np.linalg.norm(candidate_position - self.positions[j]) >= 2.0 * self.safety_distance
                    for j in range(i)):
                    self.positions[i] = candidate_position
                    break

        # Generate goal positions at least 2.5x safety_distance away from starting positions and each other
        for i in range(self.num_agents):
            while True:
                candidate_goal = np.random.rand(2) * 10
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


    '''
    Centralized Legacy Control Barrier functions
    '''
    def compute_hij(self, delta_pij, delta_vij):

        # Calculate h_ij using the provided formula
        # Change 4xcontrol limit to 2x(alpha_i + alpha_j)
        h_ij = np.sqrt(4*(self.control_limit)*(np.linalg.norm(delta_pij, ord=2)-self.safety_distance)) + (delta_pij.T @delta_vij)/np.linalg.norm(delta_pij, ord=2)
        return h_ij

    def compute_Aij_bij(self, i, j):

        # Fetch necessary data for agents i and j
        p_i = self.positions[i]
        v_i = self.velocities[i]
        p_j = self.positions[j]
        v_j = self.velocities[j]
        alpha_i = self.control_limit
        alpha_j = self.control_limit

        # Calculate delta_pij and delta_vij
        delta_pij = p_i - p_j
        delta_vij = v_i - v_j

        # Compute h_ij
        h_ij = self.compute_hij(delta_pij, delta_vij)

        # Compute A_ij and b_ij using the provided formulas
        norm_delta_pij = np.linalg.norm(delta_pij)
        
        A_ij = np.zeros((1, 2*self.num_agents))  # Assuming 2D control inputs for each agent
        b_ij=0.0

        # Returns 0 values for A_ij and b_ij if the agents are not within the neighbourhood distance
        if(np.linalg.norm(delta_pij, ord=2) >=self.neighbourhood_distance):
            return A_ij, b_ij
        
        A_ij[0][2*i] = -delta_pij[0]
        A_ij[0][2*i+1] = -delta_pij[1]
        A_ij[0][2*j] = delta_pij[0]
        A_ij[0][2*j+1] = delta_pij[1]
        
        b_ij = (self.gamma* (h_ij**3)*np.linalg.norm(delta_pij, ord=2) 
                                        - (delta_vij.T@delta_pij)**2/(np.linalg.norm(delta_pij, ord=2)**2)
                                        + (alpha_i+alpha_j)*(delta_vij.T@delta_pij)/np.sqrt(2*(alpha_i+alpha_j)*(np.linalg.norm(delta_pij, ord=2)-self.safety_distance))
                                        + np.linalg.norm(delta_vij, ord=2)**2
                                        )

        print(f"A_ij: {A_ij} for i: {i} and j: {j}")
        print(f"b_ij: {b_ij}")
        return A_ij, b_ij

    def centralized_control_barrier_qp(self, nominal_controls):
        """
        Solves the centralized control barrier QP for a multi-agent system using scipy.optimize.

        Parameters:
            nominal_controls (numpy array): The nominal controls for each agent (shape: (N, u_dim)).

        Returns:
            optimal_controls (numpy array): The optimal control inputs for each agent (shape: (N, u_dim)).
        """
        # Number of agents and control input dimensions
        N, u_dim = nominal_controls.shape

        # Flatten nominal controls for easier indexing in the optimization
        nominal_controls_flat = nominal_controls.ravel()

        # Objective function: minimize the squared error between nominal and actual controls
        def objective(u):
            return np.sum((u - nominal_controls_flat) ** 2)

        # Constraints: Control limits and safety distance constraints
        constraints = []

        # Control limits for each control variable
        for i in range(N * u_dim):
            constraints.append({'type': 'ineq', 'fun': lambda u, i=i: u[i] + self.control_limit})
            constraints.append({'type': 'ineq', 'fun': lambda u, i=i: self.control_limit - u[i]})

        # Safety distance constraints for each pair of agents
        for i in range(N):
            for j in range(i + 1, N):
                A_ij, b_ij = self.compute_Aij_bij(i, j)

                # Define the constraint as a lambda function
                constraints.append({'type': 'ineq', 'fun': lambda u, A=A_ij, b=b_ij: b - np.dot(A, u)})

        # Initial guess for the optimization (start from the nominal controls)
        initial_guess = nominal_controls_flat

        # Solve the optimization problem
        result = minimize(
            fun=objective,
            x0=initial_guess,
            constraints=constraints,
            method='SLSQP'
        )

        if result.success:
            # Reshape the optimized controls to match the input dimensions
            optimal_controls = result.x.reshape((N, u_dim))
            return optimal_controls
        else:
            print("No feasible solution found.")
            return nominal_controls



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

    def simulate(self, steps):
        """Run the simulation for a given number of steps."""
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.jet(np.linspace(0, 1, self.num_agents))

        for step in range(steps):
            # Get nominal controls from PID controller
            nominal_controls = self.pid_controller()
            # Get optimized controls using centralized control barrier QP
            controls = self.centralized_control_barrier_qp(nominal_controls)

            # Check for collisions before updating dynamics
            if self.check_collision():
                print("Simulation stopped due to safety violation.")
                break

            # Run dynamics with the optimized controls
            self.dynamics(controls)

            # # Run dynamics with the nominal controls
            # self.dynamics(nominal_controls)
            
            # Visualization of agent positions and dynamics
            ax.clear()
            ax.set_xlim(-2, 12)
            ax.set_ylim(-2, 12)
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

        plt.show()  # Show the figure and wait until it is  closed

        # Plot final trajectories
        self.plot_trajectories(colors)




    def plot_trajectories(self, colors):
        """Plot the trajectories of all agents."""
        plt.figure(figsize=(8, 8))
        plt.title("Trajectories of All Agents")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)

        for i, color in enumerate(colors):
            trajectory = np.array(self.trajectories[i])
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, label=f"Agent {i+1}")
            plt.plot(self.goal_positions[i, 0], self.goal_positions[i, 1], 'x', color=color, markersize=10)

        plt.legend(loc="upper right")
        plt.show()
        plt.close('all')  # Close all figures after plotting





