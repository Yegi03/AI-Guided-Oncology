"""
src/estimation/mcmc.py

This script performs Bayesian parameter estimation (via emcee) on the PDE+ODE model
for Qi100 (75% training, 25% test). It directly loads Qi100 data from your
datasets folder, splits into training/testing, and runs MCMC. If you prefer Qi70,
uncomment the relevant section.

"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import emcee
import corner
import matplotlib.pyplot as plt

def system_of_equations(t, variables, params):
    """
    PDE system with radial discretization plus ODE terms for binding, etc.
    variables are sliced for (C, C_b, C_bs, C_i, T).
    params = (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K, alpha_1, alpha_2,
              beta_1, beta_2, k_1, k_2, c, K_T).
    Returns d/dt of all states as a flattened array.
    """
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K, alpha_1, alpha_2,
     beta_1, beta_2, k_1, k_2, c, K_T) = params

    n = len(variables) // 5
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]

    C    = variables[0:n]
    C_b  = variables[n:2*n]
    C_bs = variables[2*n:3*n]
    C_i  = variables[3*n:4*n]
    T    = variables[4*n:5*n]

    dCdt = np.zeros_like(C)
    dC_bdt = np.zeros_like(C_b)
    dC_bsdot = np.zeros_like(C_bs)
    dC_idt = np.zeros_like(C_i)
    dTdt = np.zeros_like(T)

    # PDE for C (diffusion + reactions)
    dCdt[1:-1] = (1 / (r[1:-1]**2)) * (
        D * epsilon * r[1:-1]**2 * ((C[2:] - 2*C[1:-1] + C[:-2]) / dr**2)
    ) - (k_a * C_bs[1:-1] * C[1:-1] / epsilon) + (k_d * C_b[1:-1])
    dCdt[0] = 0
    dCdt[-1] = 0

    # C_b (bound drug)
    dC_bdt[1:-1] = (
        (k_a * C_bs[1:-1] * C[1:-1] / epsilon)
        - (k_d * C_b[1:-1])
        - (k_i * C_b[1:-1])
    )
    dC_bdt[0] = 0
    dC_bdt[-1] = 0

    # C_bs (available binding sites)
    dC_bsdot[1:-1] = (
        -(k_a * C_bs[1:-1] * C[1:-1] / epsilon)
        + (k_d * C_b[1:-1])
        + (k_i * C_b[1:-1])
        + a * C_bs[1:-1] * (1 - C_bs[1:-1]/K)
        - alpha_1*C_bs[1:-1]*T[1:-1]
    )
    dC_bsdot[0] = 0
    dC_bsdot[-1] = 0

    # C_i (internalized)
    dC_idt[1:-1] = (
        (k_i * C_b[1:-1])
        + (r[1:-1]*C_bs[1:-1]*T[1:-1])
        - (beta_1*C_i[1:-1]*T[1:-1])
        - (k_2*C[1:-1]*C_i[1:-1])
    )
    dC_idt[0] = 0
    dC_idt[-1] = 0

    # T
    dTdt[1:-1] = (
        c*T[1:-1]*(1 - T[1:-1]/K_T)
        - alpha_2*C_bs[1:-1]*T[1:-1]
        - beta_2*C_i[1:-1]*T[1:-1]
        - k_1*C[1:-1]*T[1:-1]
    )
    dTdt[0] = 0
    dTdt[-1] = 0

    return np.concatenate([dCdt, dC_bdt, dC_bsdot, dC_idt, dTdt])


def log_likelihood(params, time, volume):
    """
    Solves PDE with the given params, compares solution to 'volume' data
    with a Gaussian likelihood (sigma=0.1). Returns the log-likelihood.
    """
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K, alpha_1, alpha_2,
     beta_1, beta_2, k_1, k_2, c, K_T) = params

    n = 100
    init_conditions = np.concatenate([
        np.ones(n),    # C
        np.zeros(n),   # C_b
        np.ones(n),    # C_bs
        np.zeros(n),   # C_i
        np.ones(n)     # T
    ])

    # Ensure time is sorted, remove duplicates
    time_sorted, indices = np.unique(time, return_index=True)
    volume_sorted = volume[indices]

    try:
        sol = solve_ivp(
            system_of_equations,
            [time_sorted[0], time_sorted[-1]],
            init_conditions,
            t_eval=time_sorted,
            args=(params,),
            method='BDF'
        )
    except Exception as e:
        print("[log_likelihood] ODE solver error:", e)
        return -np.inf

    if sol.status != 0:
        print("[log_likelihood] ODE solver did not converge:", sol.message)
        return -np.inf

    # Example: interpret model_volume as the second slice of n points at the final time
    # Adjust if the "volume" is computed differently in your PDE
    model_volume = sol.y[n:2*n, -1]  # e.g., C_b or whichever slice is relevant
    if len(model_volume) != len(volume_sorted):
        return -np.inf

    sigma = 0.1
    ll = -0.5 * np.sum(((volume_sorted - model_volume)/sigma)**2)
    return ll


def log_prior(params):
    """
    Uniform prior: (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
                   alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T).
    """
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K, alpha_1, alpha_2,
     beta_1, beta_2, k_1, k_2, c, K_T) = params

    if (0 < D < 0.01 and
        0 < epsilon < 1 and
        0 < k_a < 0.1 and
        0 < C_bs_init < 1 and
        0 < k_d < 0.1 and
        0 < k_i < 0.1 and
        0 < a < 1 and
        0 < K < 1 and
        0 < alpha_1 < 1 and
        0 < alpha_2 < 1 and
        0 < beta_1 < 1 and
        0 < beta_2 < 1 and
        0 < k_1 < 0.1 and
        0 < k_2 < 0.1 and
        0 < c < 1 and
        0 < K_T < 1):
        return 0.0
    return -np.inf


def log_posterior(params, time, volume):
    """
    Posterior = log_prior + log_likelihood.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, time, volume)
    return lp + ll


def analyze_results(sampler):
    """
    Prints corner plot and final parameter means after removing burn-in.
    """
    labels = [
        "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
        "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T"
    ]
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    # Plot corner
    param_ranges = [(flat_samples[:, i].min(), flat_samples[:, i].max())
                    for i in range(flat_samples.shape[1])]
    fig = corner.corner(flat_samples, labels=labels, range=param_ranges)
    plt.show()

    # Print mean parameter values
    print("Final extracted parameters (mean):")
    params_mean = np.mean(flat_samples, axis=0)
    for param_name, val in zip(labels, params_mean):
        print(f"{param_name}: {val}")


def run_mcmc(time, volume):
    """
    Sets up initial guesses, runs MCMC sampling with emcee, and returns the sampler.
    """
    init_guess = [
        0.005, 0.1, 0.01, 0.5, 0.01, 0.01, 0.01, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ]
    ndim = len(init_guess)
    nwalkers = 36  # e.g., 2 * 18

    # Perturb the init guess slightly for each walker
    p0 = [
        np.array(init_guess) + 1e-4 * np.random.randn(ndim)
        for _ in range(nwalkers)
    ]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(time, volume)
    )
    print("[run_mcmc] Running MCMC with 1000 steps...")
    sampler.run_mcmc(p0, 1000, progress=True)
    return sampler


def main():
    """
    Main function that:
      1. Loads Qi100 data from local datasets folder.
      2. Splits into 75% training and 25% test.
      3. Runs MCMC on the training portion.
      4. Analyzes results (prints final parameters, corner plot).
      5. Saves training/test CSV for reference if desired.
    """

    # Adjust paths for Qi100 or Qi70
    # Qi100 paths:
    data_path = "../../datasets/Qi100/Qi100.xlsx"
    training_out = "qi100_training.csv"
    testing_out = "qi100_testing.csv"

    # If you want Qi70, comment out the above and uncomment these:
    # data_path = "../../datasets/Qi70/Qi70.xlsx"
    # training_out = "qi70_training.csv"
    # testing_out = "qi70_testing.csv"

    print("[main] Loading data from:", data_path)
    df = pd.read_excel(data_path)
    time = df['var1'].values
    volume = df['var2'].values

    # Sort by time and remove duplicates
    sorted_idx = np.argsort(time)
    time = time[sorted_idx]
    volume = volume[sorted_idx]
    time, unique_idx = np.unique(time, return_index=True)
    volume = volume[unique_idx]

    # Split into 75%-25%
    total_indices = np.arange(len(time))
    np.random.shuffle(total_indices)
    train_size = int(0.75 * len(time))
    train_idx = total_indices[:train_size]
    test_idx = total_indices[train_size:]

    time_train = time[train_idx]
    vol_train = volume[train_idx]
    time_test = time[test_idx]
    vol_test = volume[test_idx]

    # (Optional) save to CSV
    pd.DataFrame({'time': time_train, 'volume': vol_train}).to_csv(training_out, index=False)
    pd.DataFrame({'time': time_test,  'volume': vol_test}).to_csv(testing_out,  index=False)
    print(f"[main] Saved training to {training_out}, testing to {testing_out}")

    print("[main] Running MCMC on training set (75%)...")
    sampler = run_mcmc(time_train, vol_train)

    print("[main] MCMC completed. Analyzing results...")
    analyze_results(sampler)

    print("[main] Done.")

if __name__ == "__main__":
    main()