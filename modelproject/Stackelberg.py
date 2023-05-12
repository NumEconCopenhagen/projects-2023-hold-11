from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets


class StackelbergEquilibriumSolver:
    def __init__(self):
        """ setup model """

        par = self.par = SimpleNamespace()

        par.a = 1 # coefficiant for good 1
        par.b = 1 # coefficiant for good 2
        par.d = 20 # demand for p=0
        par.c  = [2,2] # marginal cost
        
    def demand_function(self, q1, q2):
        par = self.par
        demand = par.d-par.a*q1-par.b*q2 # inverted demand function
        return demand

    def cost_function(self, q, c):
        return c*q # marginal cost times quantity

    def profits(self, c, q1, q2): 
        # income - expenditures
        return self.demand_function(q1,q2)*q1-self.cost_function(q1,c)
    
    def reaction(self, q2,c1):
        # Maaaaaaax profit
        responce =  optimize.minimize(lambda x: - self.profits(c1,x,q2), x0=0, method = 'SLSQP')
        return responce.x # best responce

  

    def solve_eq(self):
        par = self.par
        initial_guess = np.array([0])
        # solve system of equations.
        obj = lambda q: - self.profits( par.c[0],q, self.reaction(q, par.c[1]))
        res = optimize.minimize(obj, initial_guess)
        q1 = res.x
        q2 = self.reaction(q1, par.c[1])
        return print(f'q1 = {q1} q2 = {q2}')
        
    
def sim(a = 1, b = 1, d = 20): # work on this This is another attept at trying to iterate over a costs list and output values of q1 anf q2. Does not work at the moment. 
    N = 100
    q1_values = np.empty(N)
    q2_values = np.empty(N)
    model = StackelbergEquilibriumSolver()
    # Update model parameters
    model.par.a = a
    model.par.b = b
    model.par.d = d
    # Solve Equilibrium for different costs
    # x axis grid
    grid = np.linspace(0,10,N)
    # y axis grid
    range_q = np.zeros((grid.size,2))
    for it, i in enumerate(grid):
        
    #return q1_values,q2_values
        pass 
    pass 


def plotting_function(x, y, x_label =None, y_label =None, x_lim = None, y_lim = None, labels = None, title = None, label_size = None, title_size = None):
    '''
    Plots y values of a numpy matrix by columns
    Inputs
        x (list / numpy array): x values
        y (numpy array): y values
        x_label (string): label for x-axis
        y_label (string): label for y-axis
        x_lim (tuble): limits for x axis
        y_limm (tuple): limits for y axis
        labels (list): labels for y-axis
        title (string): figure title
        label_size (int): size of y and x label
        title_size (int): size of the figure title

    '''
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot the function
    for i in range(y.shape[1]):
        ax.plot(x, y[:,i], label = labels[i])

    # titles and labels    
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
    ax.set_title(title, fontsize=title_size, fontweight=title_size)
    plt.legend()
    # Remove the border around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # axis limits
    if y_lim != None:
        ax.set_ylim(y_lim)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.show()

def plot(a = 1, b = 1, d = 20):
    model = StackelbergEquilibriumSolver()
    # Update model parameters
    model.par.a = a
    model.par.b = b
    model.par.d = d
    # x axis grid
    range_c = np.arange(0,1,0.1)
    # y axis grid
    range_q = np.zeros((range_c.size,2))
    for it, i in enumerate(range_c):
        # update cost function for company 1
        model.par.c[0] = i
        # find equilibrium quantities
        range_q[it] = model.solve_eq()  
        
    plotting_function(range_c,
                  range_q,
                  x_label = '$c_1$',
                  y_label = '$q$',
                  labels = ['Company 1','Company 2'], 
                  title = 'Stackelberg equilibrium quantities for differences in production costs', 
                  label_size = None, 
                  title_size = 16)
    
def plot_interact():
    widgets.interact(plot,
                
                 a=widgets.FloatSlider(
                     description="a", min=1, max=5, step=0.25, value=1),
                 b=widgets.FloatSlider(
                     description="b", min=1, max=5, step=0.25, value=1),
                 x0=widgets.FloatSlider(
                     description="d", min=1, max=50, step=0.5, value=20),
                 c2=widgets.FloatSlider(
                     description="c2", min=0, max=5, step=0.1, value=0)

    );


def plot2(a = 1, b = 1, d = 20):
    model = StackelbergEquilibriumSolver()
    # Update model parameters
    model.par.a = a
    model.par.b = b
    model.par.d = d
    # Solve Equilibrium
    model.solve_eq()
    # x axis grid
    grid = np.linspace(0,10,100)
    # y axis grid
    range_q = np.zeros((grid.size,2))
    range_qi = np.zeros((grid.size))
    range_qj = np.zeros((grid.size))
    for it, i in enumerate(grid):
        # update cost function for company 1
        # find equilibrium quantities
        range_q[it,0] = model.reaction(i,model.par.c[0])
        range_q[it,1] = model.reaction(range_q[it,0],model.par.c[1])

    
    plotting_function(grid,
                range_q,
                x_lim = (0,10),
                y_lim = (0,10),
                labels = ['Company 1','Company 2'], 
                title = 'Stackelberg equilibrium quantities', 
                label_size = None, 
                title_size = 16)
def plot2_interact():
    widgets.interact(plot2,
                    
                    a=widgets.FloatSlider(
                        description="a", min=1, max=5, step=0.25, value=1),
                    b=widgets.FloatSlider(
                        description="b", min=1, max=5, step=0.25, value=1),
                    x0=widgets.FloatSlider(
                        description="d", min=1, max=50, step=0.5, value=20)

    );

