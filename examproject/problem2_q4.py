import numpy as np
import matplotlib.pyplot as plt

def H3(rho, iota, sigma_epsilon, R, eta, w, K):
    # set seed
    np.random.seed(1986)
    # Parameters
    delta_values = np.linspace(0.01, 1, 100)  # Range of delta values to test

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

    # Calculate ex ante expected value H for different Delta values
    ex_ante_values = []
    for delta in delta_values:
        ex_post_values = [calculate_ex_post_value(shock, delta) for shock in shock_series]
        ex_ante_value = np.mean(ex_post_values)
        ex_ante_values.append(ex_ante_value)

    # Find optimal Delta maximizing H
    optimal_delta = delta_values[np.argmax(ex_ante_values)]
    max_ex_ante_value = np.max(ex_ante_values)

    # plotting H as a function of delta
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(delta_values, ex_ante_values, label="Figure 4: Ex Ante Expected Value (H)")
    ax.set_xlabel("Delta")
    ax.set_ylabel("Ex Ante Expected Value (H)")
    ax.set_title("Ex Ante Expected Value (H) as a Function of Delta")
    ax.scatter(optimal_delta, max_ex_ante_value, color="red", label="Optimal Delta")
    ax.legend()
    ax.grid()
    plt.show()

    print(f"Optimal Delta: {optimal_delta:.5f}")
    print(f"Maximum Ex Ante Expected Value (H): {max_ex_ante_value:.5f}")