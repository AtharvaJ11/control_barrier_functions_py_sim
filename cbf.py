import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from scipy.optimize import minimize
import cvxpy as cp
import time

class CentralizedControlBarrierFunction:
    def __init__(self, num_agents, control_limit, velocity_limit, safety_distance, dt):
        print("Running CentralizedControlBarrierFunction")
        self.num_agents = num_agents
        self.control_limit = control_limit
        self.safety_distance = safety_distance
        self.gamma =1.0  # Barrier function parameter
        self.dt = dt
        self.velocity_limit = velocity_limit
        self.neighbourhood_distance = self.safety_distance + (np.power(4*self.control_limit/self.gamma, 1/3)+ 2*self.velocity_limit)**2/(4*self.control_limit)

        self.positions = np.zeros((num_agents, 2))
        self.velocities = np.zeros((num_agents, 2))


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

    def centralized_control_barrier_qp(self, nominal_controls, positions, velocities):
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
        
        self.positions =positions
        self.velocities = velocities

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


