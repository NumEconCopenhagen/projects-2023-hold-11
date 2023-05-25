# import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as sm
from scipy import optimize
from scipy.optimize import minimize



# Define parameters
bounds = [-600, 600]
tol = 1e-8
K_underline = 10
K = 1000

def algorithm(K_underline, K, bounds, tol, figure_number=1):
    # set seed
    np.random.seed(19)

    # Initialize x_star and empty list for storing x_k0
    x_star = None
    x_k0_all = [] 

    # Implement algorithm
    for k in range(K):
        # Step 3.A: Draw a random initial guess x_k
        x_k = np.random.uniform(bounds[0], bounds[1], 2)
        
        # Step 3.C and 3.D
        if k >= K_underline:
            chi_k = 0.50 * (2 / (1 + np.exp((k - K_underline) / 100))) # step 3.C
            x_k0 = chi_k * x_k + (1 - chi_k) * x_star # step 3.D
        else:
            x_k0 = x_k 

        x_k0_all.append(x_k0) # store results in list
        
        # Step 3.E: Run the optimizer
        res = minimize(griewank, x_k0, method='BFGS', tol=tol)
        
        # Step 3.F: Update x_star if this is the first iteration or if the new result is better than the best so far
        if x_star is None or griewank(res.x) < griewank(x_star):
            x_star = res.x
            
        # Step 3.G: Check if we've reached the specified tolerance
        if griewank(x_star) < tol:
            break

    # Plot how the effective initial guesses vary with the iteration counter
    x_k0_all = np.array(x_k0_all)
    plt.figure(figsize=(10, 5))
    plt.plot(x_k0_all[:, 0], label='x1')
    plt.plot(x_k0_all[:, 1], label='x2')
    plt.xlabel('Iteration')
    plt.ylabel('Initial guess')
    plt.title(f'Figure: {figure_number} Guesses by iteration. Converged by {k} iterations')
    plt.legend()
    plt.show()

    print(f"The optimal x is {x_star}")

    # test results are close to zero
    assert np.allclose(x_star, np.zeros(2), atol=1e-4)