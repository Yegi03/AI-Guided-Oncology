"""
src/estimation/model_fit_all_treatments.py

This script performs Bayesian parameter estimation (via MCMC) on the PDE+ODE model
for all treatment datasets: Saline, Untreated, MNPS, and MNFDG.
It adapts the workflow from bayesian-inference.py to handle multiple datasets
with consistent parameter estimation, model fitting, and result visualization.

"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import emcee
import corner
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path

# 1. Add a dictionary of best-fit mechanistic parameters for each dataset (from previous fits)
# Now includes S (scaling) and T0 (initial tumor volume)
MECH_BEST_PARAMS = {
    'Saline': [0.002, 0.5, 0.03, 0.5, 0.03, 0.03, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.03, 0.5, 1000, 1.0, 120],
    'Untreated': [0.002, 0.5, 0.03, 0.5, 0.03, 0.03, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.03, 0.5, 1000, 1.0, 120],
    'MNPS': [0.002, 0.5, 0.03, 0.5, 0.03, 0.03, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.03, 0.5, 1000, 1.0, 100],
    'MNFDG': [0.002, 0.5, 0.03, 0.5, 0.03, 0.03, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.03, 0.5, 1000, 1.0, 120],
}

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
    the log-likelihood, assuming a Gaussian error model with sigma=2.0.
    """
    # Unpack parameters (now includes S and T0)
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T, S, T0) = params
    n = 100
    r = np.linspace(0, 1, n)
    # Set initial conditions for C, C_b, C_bs, C_i, T (T0 is now a parameter)
    C0 = np.ones(n)
    C_b0 = np.zeros(n)
    C_bs0 = np.ones(n)*C_bs_init
    C_i0 = np.zeros(n)
    T0_vec = np.ones(n)*T0
    initial_conditions = np.concatenate([C0, C_b0, C_bs0, C_i0, T0_vec])
    try:
        sol = solve_ivp(
            system_of_equations,
            [time[0], time[-1]],
            initial_conditions,
            t_eval=time,
            args=(params[:-2],),  # pass only the mechanistic params
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
    except Exception as e:
        return -np.inf
    if sol.status != 0:
        return -np.inf
    model_vol = compute_model_volume(sol, r)
    # Apply scaling parameter S
    model_vol_scaled = S * model_vol
    if len(model_vol_scaled) != len(volume):
        return -np.inf
    sigma = 2.0
    ll = -0.5 * np.sum(((volume - model_vol_scaled)/sigma)**2)
    return ll

def log_prior(params):
    """
    Uniform prior: each parameter must lie in a specified range.
    If outside range, return -inf (excluded). Otherwise return 0.
    """
    # Unpack parameters (now includes S and T0)
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T, S, T0) = params
    # 2. Loosened bounds for S and T0
    if (1e-5 < D < 0.02 and
        0.001 < epsilon < 1.5 and
        0.0001 < k_a < 0.2 and
        0.01 < C_bs_init < 1.2 and
        0.0001 < k_d < 0.2 and
        0.0001 < k_i < 0.2 and
        0.0001 < a < 1.2 and
        0.0001 < K < 1.2 and
        0.0001 < alpha_1 < 1.2 and
        0.0001 < alpha_2 < 1.2 and
        0.0001 < beta_1 < 1.2 and
        0.0001 < beta_2 < 1.2 and
        0.0001 < k_1 < 0.2 and
        0.0001 < k_2 < 0.2 and
        0.0001 < c < 1.2 and
        10 < K_T < 3000 and
        0.5 < S < 2.0 and
        50 < T0 < 300):
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

def analyze_results(sampler, treatment_name):
    """
    After MCMC finishes, this function extracts the final chain (dropping burn-in),
    prints statistics, and shows a corner plot.
    Returns the mean parameter vector.
    """
    labels = [
        "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
        "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T","S","T0"
    ]
    print(f"[analyze_results] Processing MCMC chain for {treatment_name}.")
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
    plt.suptitle(f'Parameter Posterior Distributions - {treatment_name}', y=1.02)
    plt.tight_layout()
    plt.show()

    # Return mean of posterior samples
    return np.mean(flat_samples, axis=0)

def run_mcmc(time, volume, treatment_name):
    """
    Sets up the parameter space, initializes MCMC walkers, performs a burn-in,
    then a main sampling phase. Returns the emcee sampler.
    """
    print(f"[run_mcmc] Setting up parameter ranges and initial guesses for {treatment_name}.")
    # 3. Loosen prior bounds and add S, T0
    param_ranges = {
        'D': (1e-5, 0.02),
        'epsilon': (0.001, 1.5),
        'k_a': (0.0001, 0.2),
        'C_bs_init': (0.01, 1.2),
        'k_d': (0.0001, 0.2),
        'k_i': (0.0001, 0.2),
        'a': (0.0001, 1.2),
        'K': (0.0001, 1.2),
        'alpha_1': (0.0001, 1.2),
        'alpha_2': (0.0001, 1.2),
        'beta_1': (0.0001, 1.2),
        'beta_2': (0.0001, 1.2),
        'k_1': (0.0001, 0.2),
        'k_2': (0.0001, 0.2),
        'c': (0.0001, 1.2),
        'K_T': (10, 3000),
        'S': (0.5, 2.0),
        'T0': (50, 300)
    }
    labels = list(param_ranges.keys())
    bounds = [param_ranges[l] for l in labels]
    ndim = len(bounds)
    nwalkers = 128
    base_guess = np.array(MECH_BEST_PARAMS[treatment_name])
    p0 = [base_guess + 0.05 * (np.random.rand(ndim) - 0.5) for _ in range(nwalkers)]
    print(f"[run_mcmc] Initializing emcee EnsembleSampler for {treatment_name}.")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(time, volume)
    )
    print(f"[run_mcmc] Running burn-in (2000 steps) for {treatment_name}.")
    state = sampler.run_mcmc(p0, 2000, progress=True)
    sampler.reset()
    print(f"[run_mcmc] Running main MCMC (5000 steps) for {treatment_name}.")
    sampler.run_mcmc(state, 5000, progress=True)
    print(f"Acceptance fractions for {treatment_name}:", sampler.acceptance_fraction)
    return sampler

def get_T0_for_initial_volume(target_volume):
    # For model: initial volume = 4*pi*T0*int_0^1 r^2 dr = 4*pi*T0*(1/3)
    # So T0 = target_volume / (4*pi/3)
    return target_volume / (4 * np.pi / 3)

def read_or_generate_params(param_file, time_train, volume_train, treatment_name, T0_init):
    """
    Checks if the param_file already exists in the datasets folder.
    If it exists, load the parameters from Excel. If not, run MCMC,
    save the new parameters to Excel, and return them.
    """
    if os.path.exists(param_file):
        print(f"[read_or_generate_params] Found existing parameter file: {param_file}")
        df_params = pd.read_excel(param_file)
        param_values = df_params.iloc[0].values
        print(f"[read_or_generate_params] Parameters loaded from Excel for {treatment_name}.")
    else:
        print(f"[read_or_generate_params] No parameter file found for {treatment_name}. Running MCMC now...")
        # Use T0_init for initial guess
        base_guess = np.array(MECH_BEST_PARAMS[treatment_name])
        base_guess[-1] = T0_init
        p0 = [base_guess + 0.05 * (np.random.rand(len(base_guess)) - 0.5) for _ in range(128)]
        sampler = emcee.EnsembleSampler(
            128, len(base_guess), log_posterior, args=(time_train, volume_train)
        )
        print(f"[run_mcmc] Running burn-in (2000 steps) for {treatment_name}.")
        state = sampler.run_mcmc(p0, 2000, progress=True)
        sampler.reset()
        print(f"[run_mcmc] Running main MCMC (5000 steps) for {treatment_name}.")
        sampler.run_mcmc(state, 5000, progress=True)
        param_values = analyze_results(sampler, treatment_name)

        print(f"[read_or_generate_params] Saving new parameter file for {treatment_name}.")
        columns = [
            "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
            "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T","S","T0"
        ]
        df_to_save = pd.DataFrame([param_values], columns=columns)
        df_to_save.to_excel(param_file, index=False)
        print(f"[read_or_generate_params] Parameters saved to {param_file}")

    return param_values

def process_dataset(dataset_name, data_file, param_file):
    """
    Process a single dataset: load data, run MCMC, simulate model, and plot results.
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load full dataset
    print(f"[process_dataset] Loading full data for {dataset_name}.")
    full_df = pd.read_excel(data_file)
    time_full = full_df['var1'].values
    volume_full = full_df['var2'].values
    
    # Sort by time in ascending order
    sorted_indices = np.argsort(time_full)
    time_full = time_full[sorted_indices]
    volume_full = volume_full[sorted_indices]
    
    # Ensure no duplicate time points
    time_full, unique_indices = np.unique(time_full, return_index=True)
    volume_full = volume_full[unique_indices]

    # Split into 75% training and 25% testing
    total_indices = np.arange(len(time_full))
    np.random.shuffle(total_indices)
    train_size = int(0.75 * len(time_full))
    train_idx = total_indices[:train_size]
    test_idx = total_indices[train_size:]

    time_train = time_full[train_idx]
    volume_train = volume_full[train_idx]
    time_test = time_full[test_idx]
    volume_test = volume_full[test_idx]

    print(f"[process_dataset] Data split: {len(time_train)} training, {len(time_test)} testing points")

    # Set T0 so model initial volume matches first training data point
    T0_init = get_T0_for_initial_volume(volume_train[0])
    # Update MECH_BEST_PARAMS for this dataset
    base_params = MECH_BEST_PARAMS[dataset_name][:]
    base_params[-1] = T0_init
    # Check or run MCMC to get parameters
    params_mean = read_or_generate_params(param_file, time_train, volume_train, dataset_name, T0_init)

    # Print final parameter estimates
    labels = [
        "D","epsilon","k_a","C_bs_init","k_d","k_i","a","K",
        "alpha_1","alpha_2","beta_1","beta_2","k_1","k_2","c","K_T","S","T0"
    ]
    print(f"\n[process_dataset] Final estimated parameters for {dataset_name}:")
    for param, val in zip(labels, params_mean):
        print(f"{param}: {val}")

    # Solve PDE with final parameters over the entire dataset
    print(f"[process_dataset] Solving PDE/ODE system with final parameters on {dataset_name} dataset.")
    (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
     alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T, S, T0) = params_mean

    # Discretization for PDE solver
    n = 100
    r = np.linspace(0, 1, n)

    # Initial conditions
    C0 = np.ones(n)
    C_b0 = np.zeros(n)
    C_bs0 = np.ones(n)*C_bs_init
    C_i0 = np.zeros(n)
    T0_vec = np.ones(n)*T0
    initial_conditions = np.concatenate([C0, C_b0, C_bs0, C_i0, T0_vec])

    # Solve PDE
    sol_full = solve_ivp(
        system_of_equations,
        [time_full[0], time_full[-1]],
        initial_conditions,
        t_eval=time_full,
        args=(params_mean[:-2],),
        method='BDF',
        rtol=1e-8,
        atol=1e-10
    )
    model_full = compute_model_volume(sol_full, r)

    # Calculate MSE on test set
    print(f"[process_dataset] Calculating MSE on {dataset_name} test data.")
    test_times_set = set(time_test)
    test_mask = [t in test_times_set for t in time_full]
    test_model = model_full[test_mask]
    mse = np.mean((volume_test - test_model)**2)
    print(f"[process_dataset] MSE on test data: {mse:.4f}")

    # Plot results
    print(f"[process_dataset] Generating plot for {dataset_name} results.")
    plt.figure(figsize=(10,6))
    plt.plot(time_train, volume_train, 'o', color='orange', label='Training data (75%)')
    plt.plot(time_test, volume_test, 'ro', label='Testing data (25%)')
    plt.plot(time_full, model_full, 'x-', color='blue', label='Model simulation')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title(f'Model Fit and Prediction - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    results_dir = "../result"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plot_path = os.path.join(results_dir, f"model_fit_{dataset_name.lower()}.png")
    txt_path = os.path.join(results_dir, f"test_accuracy_{dataset_name.lower()}.txt")

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[process_dataset] Plot saved to {plot_path}")

    with open(txt_path, 'w') as f:
        f.write(f"MSE on {dataset_name} test set: {mse:.4f}\n")
    print(f"[process_dataset] MSE saved to {txt_path}")

    plt.show()
    
    return mse, params_mean

def main():
    """
    Main workflow that processes all treatment datasets:
    - Saline (saline85)
    - Untreated (untreated85) 
    - MNPS (mnps85)
    - MNFDG (mnfdg85)
    """
    print("[main] Starting comprehensive MCMC parameter estimation for all treatments.")
    
    # Define dataset configurations
    datasets = {
        'Saline': {
            'data_file': '../../datasets/saline85/saline85.xlsx',
            'param_file': '../../datasets/saline85/estimated_params_saline85.xlsx'
        },
        'Untreated': {
            'data_file': '../../datasets/untreated85/untreated85.xlsx',
            'param_file': '../../datasets/untreated85/estimated_params_untreated85.xlsx'
        },
        'MNPS': {
            'data_file': '../../datasets/mnps85/mnps85.xlsx',
            'param_file': '../../datasets/mnps85/estimated_params_mnps85.xlsx'
        },
        'MNFDG': {
            'data_file': '../../datasets/mnfdg85/mnfdg85.xlsx',
            'param_file': '../../datasets/mnfdg85/estimated_params_mnfdg85.xlsx'
        }
    }
    
    # Process each dataset
    results = {}
    for dataset_name, config in datasets.items():
        try:
            mse, params = process_dataset(dataset_name, config['data_file'], config['param_file'])
            results[dataset_name] = {'mse': mse, 'params': params}
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            results[dataset_name] = {'mse': None, 'params': None}
    
    # Print summary of all results
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*60}")
    
    for dataset_name, result in results.items():
        if result['mse'] is not None:
            print(f"{dataset_name}: MSE = {result['mse']:.4f}")
        else:
            print(f"{dataset_name}: Failed to process")
    
    print(f"\n[main] All processing complete. Results saved to ../result/ folder.")

if __name__ == "__main__":
    main() 