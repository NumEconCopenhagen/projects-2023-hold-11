import numpy as np

def H2(rho, iota, sigma_epsilon, R, eta, w, K):
    # Parameters
    delta = 0.05  # Threshold value for policy change
    # set seed
    np.random.seed(1986)

    # Function to calculate profit at each time period
    def calculate_profit(ell_t, kappa_t, ell_previous, w, iota):
        adjustment_cost = iota if ell_t != ell_previous else 0
        profit = kappa_t*ell_t**(1-eta) - w * ell_t - adjustment_cost
        return profit

    # Function to calculate ex post value of the salon for a given shock series and policy
    def calculate_ex_post_value(shock_series, delta):
        ex_post_value = 0
        ell_previous = 0
        kappa_previous = 1

        for t in range(len(shock_series)):
            kappa_t = np.exp(rho * np.log(kappa_previous) + shock_series[t])

            ell_star = ((1 - eta) * kappa_t / w)**(1 / eta)

            if abs(ell_previous - ell_star) > delta:
                ell_t = ell_star
            else:
                ell_t = ell_previous

            profit = calculate_profit(ell_t, kappa_t, ell_previous, w, iota)
            ex_post_value += R ** (-t) * profit

            ell_previous = ell_t

        return ex_post_value

    # Generate K random shock series
    shock_series = np.random.normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=(K-1, 120))

    # Calculate ex post value for each shock series with Delta = 0
    ex_post_values_delta_0 = [calculate_ex_post_value(shock, delta=0) for shock in shock_series]

    # Calculate ex ante expected value H with Delta = 0
    H_delta_0 = np.mean(ex_post_values_delta_0)

    # Calculate ex post value for each shock series with Delta = 0.05
    ex_post_values_delta_0_05 = [calculate_ex_post_value(shock, delta) for shock in shock_series]

    # Calculate ex ante expected value H with Delta = 0.05
    H_delta_0_05 = np.mean(ex_post_values_delta_0_05)

    print(f"Approximated ex ante expected value H (Delta = 0): {H_delta_0:.3f}")
    print(f"Approximated ex ante expected value H (Delta = 0.05): {H_delta_0_05:.3f}")

    # Compare profitability
    if H_delta_0_05 > H_delta_0:
        print("The policy with Delta = 0.05 improves profitability.")
    elif H_delta_0_05 < H_delta_0:
        print("The policy with Delta = 0.05 does not improve profitability.")
    else:
        print("The policy with Delta = 0.05 has the same profitability as the policy with Delta = 0.")