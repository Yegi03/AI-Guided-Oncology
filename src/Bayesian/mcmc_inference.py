"""
src/estimation/bayesian-inference.py

This script demonstrates a PDE-based Bayesian parameter estimation workflow
for modeling data from Qi100 (and optionally Qi70, see commented lines).
We discretize a radial PDE into ODEs, solve with solve_ivp, and perform
Bayesian inference via MCMC (emcee) to estimate parameters. If an Excel file
storing the parameters already exists, the code will load from there; otherwise,
it will run MCMC, save the parameters, simulate the full dataset, and finally
plot the results.


Note:
- By default, this script focuses on Qi100. If you want to run Qi70, you can
  uncomment the relevant lines and swap paths accordingly.
- The PDE system_of_equations is defined for 5 state variables: C, C_b, C_bs,
  C_i, and T, each discretized radially.

="""

import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import emcee
import corner
import matplotlib.pyplot as plt
import openpyxl  # used for reading/writing Excel files

def system_of_equations(t, variables, params):
    """
    Defines the PDE + ODE system in radial coordinates.
    variables: A concatenation of 5 state vectors (C, C_b, C_bs, C_i, T).
    params: Tuple of model parameters, in the order:
            (D, epsilon, k_a, C_bs_initial, k_d, k_i, a, K, alpha_1, alpha_2,
             beta_1, beta_2, k_1, k_2, c, K_T).

    Returns: A flattened array of the time derivatives for all spatial points
             of each variable.
    """

    # Unpack parameter tuple
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T) = params

    # Number of radial points is total variables / 5
    n = len(variables) // 5
    r = np.linspace(0, 1, n)     # radial domain from 0 to 1
    dr = r[1] - r[0]

    # Split the single 'variables' array into separate slices
    C    = variables[0:n]
    C_b  = variables[n:2*n]
    C_bs = variables[2*n:3*n]
    C_i  = variables[3*n:4*n]
    T    = variables[4*n:5*n]

    # Create derivative arrays (same shape as the state arrays)
    dCdt = np.zeros_like(C)
    dC_bdt = np.zeros_like(C_b)
    dC_bsdot = np.zeros_like(C_bs)
    dC_idt = np.zeros_like(C_i)
    dTdt = np.zeros_like(T)

    # PDE portion for C includes a radial diffusion term and binding/unbinding
    C_over_epsilon = C / epsilon
    dCdt[1:-1] = (1 / (r[1:-1]**2)) * (
        D * epsilon * (r[1:-1]**2) * (
            C_over_epsilon[2:] - 2*C_over_epsilon[1:-1] + C_over_epsilon[:-2]
        ) / (dr**2)
    ) - (k_a * C_bs[1:-1] * C[1:-1] / epsilon) + (k_d * C_b[1:-1])

    # Enforce no-flux boundary conditions at r=0 and r=1 by setting derivatives to 0
    dCdt[0] = 0
    dCdt[-1] = 0

    # Bound drug (C_b) ODE terms
    dC_bdt[1:-1] = (
        (k_a * C_bs[1:-1] * C[1:-1] / epsilon)
        - k_d*C_b[1:-1]
        - k_i*C_b[1:-1]
    )
    dC_bdt[0] = 0
    dC_bdt[-1] = 0

    # Available binding sites (C_bs) ODE
    dC_bsdot[1:-1] = (
        -k_a * C_bs[1:-1]*C[1:-1]/epsilon
        + k_d*C_b[1:-1]
        + k_i*C_b[1:-1]
        + a*C_bs[1:-1]*(1 - C_bs[1:-1]/K)
        - alpha_1*C_bs[1:-1]*T[1:-1]
    )
    dC_bsdot[0] = 0
    dC_bsdot[-1] = 0

    # Internalized drug (C_i) ODE
    dC_idt[1:-1] = (
        k_i*C_b[1:-1]
        + r[1:-1]*C_bs[1:-1]*T[1:-1]
        - beta_1*C_i[1:-1]*T[1:-1]
        - k_2*C[1:-1]*C_i[1:-1]
    )
    dC_idt[0] = 0
    dC_idt[-1] = 0

    # Growing entity (T) ODE, includes logistic growth and drug/treatment terms
    dTdt[1:-1] = (
        c*T[1:-1]*(1 - T[1:-1]/K_T)
        - alpha_2*C_bs[1:-1]*T[1:-1]
        - beta_2*C_i[1:-1]*T[1:-1]
        - k_1*C[1:-1]*T[1:-1]
    )
    dTdt[0] = 0
    dTdt[-1] = 0

    # Return a single concatenated array of all derivatives
    return np.concatenate([dCdt, dC_bdt, dC_bsdot, dC_idt, dTdt])

def compute_model_volume(sol, r):
    """
    Integrates T (the final variable) over a spherical domain from r=0 to 1.
    This yields a single 'volume-like' scalar per time point.
    """
    n = len(r)
    T = sol.y[4*n:5*n, :]
    model_volume = 4*np.pi*np.trapz(T*(r**2)[:,None], x=r, axis=0)
    return model_volume

def log_likelihood(params, time, volume):
    """
    Given a set of parameters, solves the PDE/ODE system over 'time' and
    compares the integrated T to the experimental 'volume' data. Returns
    the log-likelihood, assuming a Gaussian error model with sigma=0.1.
    """

    print("[log_likelihood] Evaluating parameters for PDE solver...")
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T) = params

    # Radial grid resolution
    n = 100
    r = np.linspace(0, 1, n)

    # Set initial conditions for C, C_b, C_bs, C_i, T
    C0 = np.ones(n)
    C_b0 = np.zeros(n)
    C_bs0 = np.ones(n)*C_bs_init
    C_i0 = np.zeros(n)
    T0 = np.ones(n)
    initial_conditions = np.concatenate([C0, C_b0, C_bs0, C_i0, T0])

    # Solve PDE/ODE system
    try:
        sol = solve_ivp(
            system_of_equations,
            [time[0], time[-1]],
            initial_conditions,
            t_eval=time,
            args=(params,),
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
    except Exception as e:
        print("[log_likelihood] ODE solver error:", e)
        return -np.inf

    # Check if solver ended abnormally
    if sol.status != 0:
        print("[log_likelihood] ODE solver did not converge.")
        return -np.inf

    # Integrate T to obtain model's volume prediction at each time point
    model_vol = compute_model_volume(sol, r)

    # Ensure the simulation and data have same length
    if len(model_vol) != len(volume):
        print("[log_likelihood] Length mismatch between model output and data.")
        return -np.inf

    # Simple Gaussian likelihood with sigma=0.1
    sigma = 0.1
    ll = -0.5 * np.sum(((volume - model_vol)/sigma)**2)
    return ll

def log_prior(params):
    """
    Uniform prior: each parameter must lie in a specified range.
    If outside range, return -inf (excluded). Otherwise return 0.
    """
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T) = params

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
        0 < K_T < 2000):
        return 0.0
    else:
        return -np.inf

def log_posterior(params, time, volume):
    """
    Posterior = log_prior + log_likelihood.
    Returns -inf if either prior or likelihood is invalid for these params.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, time, volume)
    return lp + ll

def print_summary_statistics(flat_samples, labels):
    """
    Prints out mean, std, and 95% credible interval for each parameter.
    """
    mean_params = np.mean(flat_samples, axis=0)
    std_params = np.std(flat_samples, axis=0)
    lower_95 = np.percentile(flat_samples, 2.5, axis=0)
    upper_95 = np.percentile(flat_samples, 97.5, axis=0)

    print("Parameter summary:")
    for i, label in enumerate(labels):
        print(f"{label}: mean={mean_params[i]:.4g}, std={std_params[i]:.4g}, "
              f"95% CI=[{lower_95[i]:.4g}, {upper_95[i]:.4g}]")

def analyze_results(sampler):
    """
    After MCMC finishes, this function extracts the final chain (dropping burn-in),
    prints statistics, and shows a corner plot.
    Returns the mean parameter vector.
    """
    labels = [
        "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
        "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T"
    ]
    print("[analyze_results] Processing MCMC chain.")
    flat_samples = sampler.get_chain(discard=200, thin=10, flat=True)

    # Print statistics
    print_summary_statistics(flat_samples, labels)

    # Show corner plot
    fig = corner.corner(
        flat_samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975]
    )
    plt.show()

    # Return mean of posterior samples
    return np.mean(flat_samples, axis=0)

def run_mcmc(time, volume):
    """
    Sets up the parameter space, initializes MCMC walkers, performs a burn-in,
    then a main sampling phase. Returns the emcee sampler.
    """
    print("[run_mcmc] Setting up parameter ranges and initial guesses.")

    # Parameter range dictionary for uniform priors
    param_ranges = {
        'D': (1e-4, 0.009),
        'epsilon': (0.01, 0.9),
        'k_a': (0.001, 0.09),
        'C_bs_init': (0.1, 0.9),
        'k_d': (0.001, 0.09),
        'k_i': (0.001, 0.09),
        'a': (0.001, 0.9),
        'K': (0.001, 0.9),
        'alpha_1': (0.001, 0.9),
        'alpha_2': (0.001, 0.9),
        'beta_1': (0.001, 0.9),
        'beta_2': (0.001, 0.9),
        'k_1': (0.001, 0.09),
        'k_2': (0.001, 0.09),
        'c': (0.001, 0.9),
        'K_T': (1, 2000)
    }

    labels = list(param_ranges.keys())
    bounds = [param_ranges[l] for l in labels]
    ndim = len(bounds)
    nwalkers = 64

    # Initialize walkers within the prior ranges
    p0 = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        for j, (low, high) in enumerate(bounds):
            p0[i, j] = np.random.uniform(low, high)

    print("[run_mcmc] Initializing emcee EnsembleSampler.")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(time, volume)
    )

    print("[run_mcmc] Running burn-in (500 steps).")
    state = sampler.run_mcmc(p0, 500, progress=True)
    sampler.reset()  # clears the chain so burn-in samples are discarded

    print("[run_mcmc] Running main MCMC (2000 steps).")
    sampler.run_mcmc(state, 2000, progress=True)

    print("Acceptance fractions:", sampler.acceptance_fraction)
    return sampler

def read_or_generate_params(param_file, time_train, volume_train):
    """
    Checks if the param_file already exists in the datasets folder.
    If it exists, load the parameters from Excel. If not, run MCMC,
    save the new parameters to Excel, and return them.
    """
    if os.path.exists(param_file):
        print(f"[read_or_generate_params] Found existing parameter file: {param_file}")
        df_params = pd.read_excel(param_file)
        param_values = df_params.iloc[0].values
        print("[read_or_generate_params] Parameters loaded from Excel.")
    else:
        print("[read_or_generate_params] No parameter file found. Running MCMC now...")
        sampler = run_mcmc(time_train, volume_train)
        param_values = analyze_results(sampler)

        print("[read_or_generate_params] Saving new parameter file.")
        columns = [
            "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
            "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T"
        ]
        df_to_save = pd.DataFrame([param_values], columns=columns)
        df_to_save.to_excel(param_file, index=False)
        print(f"[read_or_generate_params] Parameters saved to {param_file}")

    return param_values

def main():
    """
    Main workflow:
      - Load full dataset (100%).
      - Load training dataset (75%).
      - Check or run MCMC to get parameters.
      - Simulate PDE with those parameters.
      - Compare results to test data (25%), compute MSE, and plot.
      - Save results to the results folder.
    """

    print("[main] Starting script.")

    # # -----------------
    # # Data paths for Qi100
    # # -----------------
    # file_100 = "../../datasets/Qi100/Qi100.xlsx"       # full data file
    # file_75  = "../../datasets/Qi100/75_qi100.xlsx"    # 75% training data
    # param_file = "../../datasets/Qi100/estimated_params_qi100.xlsx"  # for saving/loading Qi100 parameters

    # -----------------
    # Data paths for Qi70 (COMMENTED OUT for now)
    # -----------------
    file_100 = "../../datasets/Qi70/Qi70.xlsx"
    file_75  = "../../datasets/Qi70/75_qi70.xlsx"
    param_file = "../../datasets/Qi70/estimated_params_qi70.xlsx"

    # -----------------
    # Load full data (Qi100)
    # -----------------
    print("[main] Loading full data for Qi100.")
    full_df = pd.read_excel(file_100)
    time_full = full_df['var1'].values
    volume_full = full_df['var2'].values
    # Sort by time in ascending order
    sorted_indices = np.argsort(time_full)
    time_full = time_full[sorted_indices]
    volume_full = volume_full[sorted_indices]
    # Ensure no duplicate time points
    time_full, unique_indices = np.unique(time_full, return_index=True)
    volume_full = volume_full[unique_indices]

    # -----------------
    # Load 75% training data (Qi100)
    # -----------------
    print("[main] Loading 75% training data.")
    train_df = pd.read_excel(file_75)
    time_train = train_df['var1'].values
    volume_train = train_df['var2'].values
    # Sort + remove duplicates
    sorted_indices = np.argsort(time_train)
    time_train = time_train[sorted_indices]
    volume_train = volume_train[sorted_indices]
    time_train, unique_indices = np.unique(time_train, return_index=True)
    volume_train = volume_train[unique_indices]

    print("[main] Checking or generating parameters for Qi100.")
    params_mean = read_or_generate_params(param_file, time_train, volume_train)

    # Print final parameter estimates
    labels = [
        "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
        "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T"
    ]
    print("\n[main] Final estimated parameters (used for simulation):")
    for param, val in zip(labels, params_mean):
        print(f"{param}: {val}")

    # Solve PDE with final parameters over the entire Qi100 dataset
    print("[main] Solving PDE/ODE system with final parameters on Qi100 dataset.")
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T) = params_mean

    # Discretization for PDE solver
    n = 100
    r = np.linspace(0, 1, n)

    # Initial conditions
    C0 = np.ones(n)
    C_b0 = np.zeros(n)
    C_bs0 = np.ones(n)*C_bs_init
    C_i0 = np.zeros(n)
    T0 = np.ones(n)
    initial_conditions = np.concatenate([C0, C_b0, C_bs0, C_i0, T0])

    # Solve PDE
    sol_full = solve_ivp(
        system_of_equations,
        [time_full[0], time_full[-1]],
        initial_conditions,
        t_eval=time_full,
        args=(params_mean,),
        method='BDF',
        rtol=1e-8,
        atol=1e-10
    )
    model_full = compute_model_volume(sol_full, r)

    # Identify which points are test data (the other 25%)
    print("[main] Identifying test data (25%).")
    train_times_set = set(time_train)
    test_mask = [t not in train_times_set for t in time_full]

    # -----------------
    # Plot results
    # -----------------
    print("[main] Generating plot for Qi100 results.")
    plt.figure(figsize=(10,6))
    plt.plot(time_train, volume_train, 'o', color='orange', label='Training data (75%)')
    plt.plot(time_full[test_mask], volume_full[test_mask], 'ro', label='Testing data (25%)')
    plt.plot(time_full, model_full, 'x-', color='blue', label='Simulation (Qi100)')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Model Fit and Prediction (Qi100)')
    plt.legend()

    # -----------------
    # Calculate MSE on test set
    # -----------------
    print("[main] Calculating MSE on Qi100 test data.")
    test_time = time_full[test_mask]
    test_volume = volume_full[test_mask]
    test_model = model_full[test_mask]
    mse = np.mean((test_volume - test_model)**2)
    print(f"[main] MSE on test data: {mse:.4f}")

    # -----------------
    # Save outputs to 'result' folder
    # -----------------
    results_dir = "../result"  # make sure this folder exists in the project
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plot_path = os.path.join(results_dir, "model_fit_qi100.png")
    txt_path  = os.path.join(results_dir, "test_accuracy_qi100.txt")

    plt.savefig(plot_path, dpi=150)
    print(f"[main] Plot saved to {plot_path}")

    with open(txt_path, 'w') as f:
        f.write(f"MSE on Qi100 test set: {mse:.4f}\n")
    print(f"[main] MSE saved to {txt_path}")

    plt.show()
    print("[main] Done. Results are in the 'result' folder.")

if __name__ == "__main__":
    main()