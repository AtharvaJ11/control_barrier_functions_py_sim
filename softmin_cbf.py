import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from scipy.optimize import minimize
import time

class SoftminCentralizedControlBarrierFunction:
    def __init__(self, num_agents, control_limit, velocity_limit, safety_distance, dt):
        self.num_agents = num_agents
        self.control_limit = control_limit
        self.safety_distance = safety_distance
        self.gamma =1.0  # Barrier function parameter
        self.dt = dt
        self.velocity_limit = velocity_limit
        self.neighbourhood_distance = self.safety_distance + (np.power(4*self.control_limit/self.gamma, 1/3)+ 2*self.velocity_limit)**2/(4*self.control_limit)
        
        self.tau = .10 * np.log(self.num_agents)  # Softmin parameter
        # self.tau = 0.3/(self.num_agents)
        self.positions = np.zeros((num_agents, 2))
        self.velocities = np.zeros((num_agents, 2))
        self.eps = 0
        
        # Initialize h_ij matrix
        self.h_ij_matrix = np.zeros((self.num_agents, self.num_agents))

    '''
    Centralized Legacy Control Barrier functions
    '''
    def compute_hij(self, delta_pij, delta_vij):

        # Calculate h_ij using the provided formula
        # Change 4xcontrol limit to 2x(alpha_i + alpha_j)
        h_ij = np.sqrt(4*(self.control_limit)*(np.linalg.norm(delta_pij, ord=2)-self.safety_distance)) + (delta_pij.T @delta_vij)/np.linalg.norm(delta_pij, ord=2)
        return h_ij
    
    def compute_softmin_h(self):
        
        # Compute self.h and self.S 
        self.h = 0.0
        self.S = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                delta_pij = self.positions[i] - self.positions[j]
                delta_vij = self.velocities[i] - self.velocities[j]
                h_ij = self.compute_hij(delta_pij, delta_vij)
                self.h_ij_matrix[i, j] = h_ij
                self.h_ij_matrix[j, i] = h_ij  # Since h_ij is symmetric
                self.S += np.exp(-h_ij / self.tau)
        
        self.h = -self.tau * np.log(self.S)

    def get_coeff(self, i):
        A_uxi = 0.0
        A_uyi = 0.0
        A_vxi = 0.0
        A_vyi = 0.0

        for j in range(self.num_agents):
            if i != j:
                delta_pij = self.positions[i] - self.positions[j]
                delta_vij = self.velocities[i] - self.velocities[j]
                h_ij = self.h_ij_matrix[i, j]  # Get h_ij value from the matrix
                delta_pij_norm = np.linalg.norm(delta_pij, ord=2)
                
                exp_term = np.exp(-h_ij / self.tau)
                
                A_uxi += exp_term * delta_pij[0] / delta_pij_norm
                A_uyi += exp_term * delta_pij[1] / delta_pij_norm
                
                delhij_delxi = (2 * self.control_limit * delta_pij[0]) / (delta_pij_norm * np.sqrt(4 * self.control_limit * (delta_pij_norm - self.safety_distance)))
                delhij_delxi += delta_vij[0] / delta_pij_norm - (delta_pij[0] * (delta_pij.T @ delta_vij)) / (delta_pij_norm ** 3)
                
                delhij_delyi = (2 * self.control_limit * delta_pij[1]) / (delta_pij_norm * np.sqrt(4 * self.control_limit * (delta_pij_norm - self.safety_distance)))
                delhij_delyi += delta_vij[1] / delta_pij_norm - (delta_pij[1] * (delta_pij.T @ delta_vij)) / (delta_pij_norm ** 3)
                
                A_vxi += exp_term * delhij_delxi
                A_vyi += exp_term * delhij_delyi

        return -A_uxi, -A_uyi, A_vxi, A_vyi


    def compute_Ai_bi(self, i):
        
        A_i = np.zeros((1, 2*self.num_agents))  # Assuming 2D control inputs for each agent
        b_i=0.0

        Axi, Ayi, Avxi, Avyi = self.get_coeff(i)
        

        A_i[0, 2*i] = Axi
        A_i[0, 2*i+1] = Ayi
        # print(f"Axi: {Axi}, Ayi: {Ayi}")

        b_i = self.S*self.gamma*(self.h**3) + Avxi*self.velocities[i, 0] + Avyi*self.velocities[i, 1]
        
        return A_i, b_i

    def centralized_control_barrier_qp(self, nominal_controls, positions, velocities):
        """
        Solves the centralized control barrier QP for a multi-agent system using scipy.optimize. The optimization is based on softmin of the barrier functions.

        Parameters:
            nominal_controls (numpy array): The nominal controls for each agent (shape: (N, u_dim)).

        Returns:
            optimal_controls (numpy array): The optimal control inputs for each agent (shape: (N, u_dim)).
        """
        # Number of agents and control input dimensions
        N, u_dim = nominal_controls.shape

        
        # Create a copy of nominal controls
        left_turn_control = np.copy(nominal_controls)
        # Rotating left or CCW
        left_turn_control[:, 0] = nominal_controls[:, 1]
        left_turn_control[:, 1] = -1 * nominal_controls[:, 0]
        
        # Flatten nominal controls for easier indexing in the optimization
        nominal_controls_flat = nominal_controls.ravel()
        
        left_turn_control_flat = left_turn_control.ravel()
        
        
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

        A = np.zeros((N, 2*N))
        b = np.zeros(N) 
        
        self.compute_softmin_h()
        # Safety distance constraints for each pair of agents
        for i in range(N):
            # A_i of shape (1, 2N) and b_i of shape (1,)
            A_i, b_i = self.compute_Ai_bi(i)
            A[i] = A_i
            b[i] = b_i    

        # Define the constraint as a lambda function
        constraints.append({'type': 'ineq', 'fun': lambda u, A=A, b=b: b - np.dot(A, u)})

        # Initial guess for the optimization (start from the nominal controls)
        initial_guess = nominal_controls_flat

        # Solve the optimization problem
        result = minimize(
            fun=objective,
            x0=initial_guess,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-6*self.num_agents*self.num_agents}
        )


        if result.success:
            # Reshape the optimized controls to match the input dimensions
            optimal_controls = result.x.reshape((N, u_dim))
            return optimal_controls, True
        else:
            print(f"Norm of velocities {np.linalg.norm(velocities)}")
            print(f"Message: {result.message}")
            print(f"Final Cost Function Value: {result.fun}")
            print(f"Gradient Norm (KKT Residual): {np.linalg.norm(result.jac)}")
            print(f"Status Code: {result.status}")
            print(f"Result: {result.x}")
            return result.x.reshape((N, u_dim)), False


