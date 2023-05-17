from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets


class StackelbergDuopoly:
    def __init__(self, c, d):
        """Create Model"""
        self.c = c  # marginal cost
        self.d = d  # y-intercept of demand function / total demand when price is zero

    # Profit_f is the follower firm
    def profit_f(self, q1, q2):
        return (self.d - (q1 + q2) - self.c) * q1

    # Profit_l is the leader firm
    def profit_l(self, q1, q2):
        return (self.d - (q2 + q1) - self.c) * q2

    # Reaction of follower
    def R_f(self, q2):
        res = optimize.minimize(lambda q1: -self.profit_f(q1, q2), x0=0, bounds=[(0, np.inf)], method='Nelder-Mead')
        return res.x[0]

    # Reaction of leader
    def R_l(self, q1):
        res = optimize.minimize(lambda q2: -self.profit_l(q1, q2), x0=0, bounds=[(0, np.inf)], method='Nelder-Mead')
        return res.x[0]

    # Get optimal quantities using backwards induction
    def get_optimal_quantities(self):
        # Step 1: Solve follower's reaction given any quantity produced by the leader
        def follower_reaction(q2):
            return self.R_f(q2)

        # Step 2: Maximize leader's profit function using follower's reaction
        def leader_profit(q2):
            q1 = follower_reaction(q2)
            return self.profit_l(q1, q2)

        res = optimize.minimize(lambda q2: -leader_profit(q2), x0=0, bounds=[(0, np.inf)], method='Nelder-Mead')
        q2_opt = res.x[0]

        # Step 3: Find optimal quantity for the follower
        q1_opt = follower_reaction(q2_opt)

        return q1_opt, q2_opt


def plot_optimal_quantities(c=0, d=20, c_min = 1, c_max = 20, num_points=500):
    """
    Plots the Stackelberg duopoly quantities as a function of marginal costs.

        Parameters:
            d (int): Total demand when price is zero
            c (int): Marginal cost, this can really be anything and will not change the output, only needed to make it run. 
            c_min (int): Where to start plot 
            c_max (int): Where to end plot
            num_points (int): How many points to create in the plot 

        Returns:
            2d plot
    """
    # Define the range of c values to consider
    c_values = np.linspace(c_min, c_max, num_points)

    # Initialize arrays to store the optimal quantities
    qf_values = np.zeros_like(c_values)
    ql_values = np.zeros_like(c_values)

    # Create a StackelbergDuopoly object with fixed parameters
    model = StackelbergDuopoly(c=c, d=d)

    # Compute the optimal quantities for each value of c
    for i, c in enumerate(c_values):
        model.c = c  # Update the value of c in the duopoly object
        qf_opt, ql_opt = model.get_optimal_quantities()
        qf_values[i] = qf_opt
        ql_values[i] = ql_opt

    # Plot the optimal quantities against the cost parameter c
    plt.plot(c_values, qf_values, label='Quantity of follower')
    plt.plot(c_values, ql_values, label='Quantity of leader')
    plt.xlabel('Marginal Cost')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()