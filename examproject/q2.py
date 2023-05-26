import numpy as np
def H(rho, iota, sigma_epsilon, R, eta, w, K):
    """
    Function to calculate ex ante value of the salon

    Args:
    -----
        rho (float): AR(1) parameter
        iota (float): Adjustment cost
        sigma_epsilon (float): Standard deviation of demand shock
        R (float): Discount factor
        eta (float): Elasticity of demand
        w (float): Wage
        K (int): Number of shock series

    Returns:
    --------
        float: Ex ante value of the salon

    """
    # set seed
    np.random.seed(1986)

    # Function to calculate profit at each time period
    def calculate_profit(ell_t, kappa_t, ell_previous, w, iota):
        adjustment_cost = iota if ell_t != ell_previous else 0
        profit = kappa_t*ell_t**(1-eta) - w * ell_t - adjustment_cost
        return profit

    # Function to calculate ex post value of the salon for a given shock series
    def calculate_ex_post_value(shock_series):
        ex_post_value = 0 # starting with no value
        ell_previous = 0 # starting with no employees
        kappa_previous = 1 # initial demand shock

        for t in range(len(shock_series)):
            kappa_t = np.exp(rho * np.log(kappa_previous) + shock_series[t]) # isolating demand shock

            ell_t = ((1 - eta) * kappa_t / w) ** (1 / eta)

            profit = calculate_profit(ell_t, kappa_t, ell_previous, w, iota)
            ex_post_value += R ** (-t) * profit

            ell_previous = ell_t # updating ell_previous

        return ex_post_value 

    # Generate K random shock series
    shock_series = np.random.normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=(K-1, 120)) 

    # Calculate ex post value for each shock series
    ex_post_values = [calculate_ex_post_value(shock) for shock in shock_series] 

    H = np.mean(ex_post_values) # return mean of ex post values

    
    return print(f'Discounted sum of profit: {round(H, 3)}')

    