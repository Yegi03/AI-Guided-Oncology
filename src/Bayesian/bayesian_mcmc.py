"""
Bayesian Markov Chain Monte Carlo (MCMC) Implementation for Tumor-Nanoparticle Model

This module implements the Bayesian inference framework described in the paper:
"A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology"

The implementation follows the methodology outlined in Appendix I of the paper, using:
1. MCMC sampling with emcee ensemble sampler
2. Proper likelihood and prior definitions
3. Convergence diagnostics and uncertainty quantification
4. Parameter estimation for all treatment groups

Author: Yeganeh Abdollahinejad, Amit K Chattopadhyay, et al.
Date: 2024-2025
"""

import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from typing import Tuple, List, Dict, Optional
import os
import pickle
from datetime import datetime

from .mechanistic_model import TumorNanoparticleModel

class BayesianMCMCInference:
    """
    Bayesian MCMC inference for the tumor-nanoparticle model.
    
    This class implements the complete Bayesian inference pipeline including:
    - Likelihood function based on Gaussian noise model
    - Prior distributions for all parameters
    - MCMC sampling with emcee
    - Convergence diagnostics
    - Posterior analysis and visualization
    """
    
    def __init__(self, model: TumorNanoparticleModel, treatment_name: str):
        """
        Initialize the Bayesian MCMC inference.
        
        Parameters:
        -----------
        model : TumorNanoparticleModel
            The mechanistic model instance
        treatment_name : str
            Name of the treatment group (e.g., 'Untreated', 'Saline', 'MNP', 'MNPFDG')
        """
        self.model = model
        self.treatment_name = treatment_name
        self.n_params = len(model.param_names)
        
        # MCMC parameters (as specified in the paper)
        self.n_walkers = 128  # Number of walkers in ensemble
        self.n_steps = 7000   # Total number of steps (2000 burn-in + 5000 main)
        self.burn_in_steps = 2000
        
        # Results storage
        self.sampler = None
        self.flat_samples = None
        self.parameter_estimates = None
        
    def log_likelihood(self, params: np.ndarray, time_data: np.ndarray, 
                      volume_data: np.ndarray, sigma: float = 0.1) -> float:
        """
        Compute the log-likelihood function.
        
        Following the methodology in Appendix I, we assume the observed data
        follows a normal distribution with mean given by the model prediction
        and standard deviation sigma.
        
        Parameters:
        -----------
        params : np.ndarray
            Model parameters
        time_data : np.ndarray
            Time points of observations
        volume_data : np.ndarray
            Observed tumor volumes
        sigma : float
            Standard deviation of the noise model (default: 0.1)
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        try:
            # Solve the model with current parameters
            solution = self.model.solve_model(
                params=tuple(params),
                time_span=(time_data.min(), time_data.max())
            )
            
            # Compute predicted tumor volumes
            predicted_volumes = self.model.compute_tumor_volume(solution)
            
            # Interpolate predictions to match data time points
            from scipy.interpolate import interp1d
            f_interp = interp1d(solution.t, predicted_volumes, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
            predicted_at_data_times = f_interp(time_data)
            
            # Compute log-likelihood (Gaussian noise model)
            # log L = -0.5 * sum((y_obs - y_pred)² / σ²) - n/2 * log(2πσ²)
            residuals = volume_data - predicted_at_data_times
            log_likelihood = -0.5 * np.sum(residuals**2 / sigma**2) - len(volume_data)/2 * np.log(2*np.pi*sigma**2)
            
            return log_likelihood
            
        except Exception as e:
            # Return very low likelihood for invalid parameters
            return -np.inf
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute the log-prior probability.
        
        Following the methodology in Appendix I, we use normal priors for parameters
        based on literature values from Goodman et al. (2008) and Graff & Wittrup (2003).
        
        Parameters:
        -----------
        params : np.ndarray
            Model parameters
            
        Returns:
        --------
        float
            Log-prior probability
        """
        # Get parameter bounds
        bounds = self.model.get_parameter_bounds()
        
        # Check if parameters are within bounds
        for i, (param, (min_val, max_val)) in enumerate(zip(params, bounds)):
            if param < min_val or param > max_val:
                return -np.inf
        
        # Define prior means and standard deviations based on literature
        # These are based on Goodman et al. (2008) and Graff & Wittrup (2003)
        prior_means = np.array([
            0.003,   # D: Diffusion coefficient [mm²/s]
            0.4,     # epsilon: Porosity
            0.03,    # k_a: Association rate [s⁻¹]
            0.5,     # C_bs_init: Initial binding sites
            0.03,    # k_d: Dissociation rate [s⁻¹]
            0.03,    # k_i: Internalization rate [s⁻¹]
            0.5,     # a: Binding site growth rate
            0.5,     # K: Carrying capacity
            0.5,     # alpha_1: Tumor inhibition by binding sites
            0.5,     # alpha_2: Tumor inhibition by binding sites (alt)
            0.5,     # beta_1: Tumor inhibition by internalized drug
            0.5,     # beta_2: Tumor inhibition by internalized drug (alt)
            0.03,    # k_1: Direct nanoparticle-tumor interaction
            0.03,    # k_2: Nanoparticle-internalized interaction
            0.5,     # c: Tumor growth rate [day⁻¹]
            1000     # K_T: Tumor carrying capacity [cells/mm³]
        ])
        
        prior_stds = np.array([
            0.001,   # D
            0.1,     # epsilon
            0.01,    # k_a
            0.1,     # C_bs_init
            0.01,    # k_d
            0.01,    # k_i
            0.1,     # a
            0.1,     # K
            0.1,     # alpha_1
            0.1,     # alpha_2
            0.1,     # beta_1
            0.1,     # beta_2
            0.01,    # k_1
            0.01,    # k_2
            0.1,     # c
            100      # K_T
        ])
        
        # Compute log-prior (sum of log-normal densities)
        log_prior = 0.0
        for i, (param, mean, std) in enumerate(zip(params, prior_means, prior_stds)):
            log_prior += norm.logpdf(param, mean, std)
        
        return log_prior
    
    def log_posterior(self, params: np.ndarray, time_data: np.ndarray, 
                     volume_data: np.ndarray) -> float:
        """
        Compute the log-posterior probability.
        
        Following Bayes' theorem: log P(θ|data) = log P(data|θ) + log P(θ) - log P(data)
        Since P(data) is a constant, we compute: log P(θ|data) ∝ log P(data|θ) + log P(θ)
        
        Parameters:
        -----------
        params : np.ndarray
            Model parameters
        time_data : np.ndarray
            Time points of observations
        volume_data : np.ndarray
            Observed tumor volumes
            
        Returns:
        --------
        float
            Log-posterior probability
        """
        log_prior_val = self.log_prior(params)
        
        # If prior is -inf, posterior is also -inf
        if not np.isfinite(log_prior_val):
            return -np.inf
        
        log_likelihood_val = self.log_likelihood(params, time_data, volume_data)
        
        # Return sum of log-likelihood and log-prior
        return log_likelihood_val + log_prior_val
    
    def initialize_walkers(self) -> np.ndarray:
        """
        Initialize MCMC walkers around the prior means.
        
        Returns:
        --------
        np.ndarray
            Initial positions of walkers (n_walkers × n_params)
        """
        # Get prior means and standard deviations
        prior_means = np.array([
            0.003, 0.4, 0.03, 0.5, 0.03, 0.03, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.03, 0.03, 0.5, 1000
        ])
        
        prior_stds = np.array([
            0.001, 0.1, 0.01, 0.1, 0.01, 0.01, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 100
        ])
        
        # Initialize walkers with small random perturbations around prior means
        initial_positions = np.zeros((self.n_walkers, self.n_params))
        
        for i in range(self.n_walkers):
            # Add small random perturbation to prior means
            perturbation = np.random.normal(0, 0.1 * prior_stds)
            initial_positions[i] = prior_means + perturbation
        
        return initial_positions
    
    def run_mcmc(self, time_data: np.ndarray, volume_data: np.ndarray, 
                save_results: bool = True) -> Dict:
        """
        Run the MCMC sampling procedure.
        
        This implements the complete MCMC workflow as described in Appendix I:
        1. Initialize walkers
        2. Run burn-in phase
        3. Run main sampling phase
        4. Analyze convergence
        5. Extract posterior samples
        
        Parameters:
        -----------
        time_data : np.ndarray
            Time points of observations
        volume_data : np.ndarray
            Observed tumor volumes
        save_results : bool
            Whether to save results to file
            
        Returns:
        --------
        Dict
            Dictionary containing MCMC results and diagnostics
        """
        print(f"Starting MCMC inference for {self.treatment_name} treatment...")
        
        # Initialize walkers
        initial_positions = self.initialize_walkers()
        
        # Create emcee sampler
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.n_walkers,
            ndim=self.n_params,
            log_prob_fn=lambda params: self.log_posterior(params, time_data, volume_data),
            backend=None  # We'll handle storage manually
        )
        
        # Run burn-in phase
        print(f"Running burn-in phase ({self.burn_in_steps} steps)...")
        burn_in_state = self.sampler.run_mcmc(
            initial_positions, 
            self.burn_in_steps, 
            progress=True
        )
        
        # Reset sampler and run main sampling phase
        print(f"Running main sampling phase ({self.n_steps - self.burn_in_steps} steps)...")
        self.sampler.reset()
        self.sampler.run_mcmc(
            burn_in_state.coords, 
            self.n_steps - self.burn_in_steps, 
            progress=True
        )
        
        # Extract flat samples (all walkers, all steps after burn-in)
        self.flat_samples = self.sampler.get_chain(flat=True)
        
        # Analyze results
        results = self.analyze_results()
        
        # Save results if requested
        if save_results:
            self.save_results(results)
        
        return results
    
    def analyze_results(self) -> Dict:
        """
        Analyze MCMC results and compute summary statistics.
        
        Returns:
        --------
        Dict
            Dictionary containing analysis results
        """
        if self.flat_samples is None:
            raise ValueError("No samples available. Run MCMC first.")
        
        # Compute parameter estimates (mean and standard deviation)
        param_means = np.mean(self.flat_samples, axis=0)
        param_stds = np.std(self.flat_samples, axis=0)
        
        # Create parameter estimates dictionary
        self.parameter_estimates = {}
        for i, param_name in enumerate(self.model.param_names):
            self.parameter_estimates[param_name] = {
                'mean': param_means[i],
                'std': param_stds[i],
                'median': np.median(self.flat_samples[:, i]),
                'q16': np.percentile(self.flat_samples[:, i], 16),
                'q84': np.percentile(self.flat_samples[:, i], 84)
            }
        
        # Compute convergence diagnostics
        convergence_diagnostics = self.compute_convergence_diagnostics()
        
        # Compute acceptance fraction
        acceptance_fraction = np.mean(self.sampler.acceptance_fraction)
        
        results = {
            'treatment_name': self.treatment_name,
            'parameter_estimates': self.parameter_estimates,
            'convergence_diagnostics': convergence_diagnostics,
            'acceptance_fraction': acceptance_fraction,
            'n_samples': len(self.flat_samples),
            'n_walkers': self.n_walkers,
            'n_steps': self.n_steps,
            'burn_in_steps': self.burn_in_steps
        }
        
        return results
    
    def compute_convergence_diagnostics(self) -> Dict:
        """
        Compute convergence diagnostics for MCMC chains.
        
        Returns:
        --------
        Dict
            Dictionary containing convergence diagnostics
        """
        # Get chains for each walker
        chains = self.sampler.get_chain()
        
        # Compute Gelman-Rubin diagnostic (R-hat) for each parameter
        r_hat_values = []
        for i in range(self.n_params):
            # Extract chains for this parameter
            param_chains = chains[:, :, i]  # (n_steps, n_walkers)
            
            # Compute within-chain variance
            within_var = np.var(param_chains, axis=0, ddof=1)
            mean_within_var = np.mean(within_var)
            
            # Compute between-chain variance
            chain_means = np.mean(param_chains, axis=0)
            overall_mean = np.mean(chain_means)
            between_var = np.var(chain_means, ddof=1)
            
            # Compute R-hat
            n_steps = param_chains.shape[0]
            r_hat = np.sqrt((between_var + mean_within_var) / mean_within_var)
            r_hat_values.append(r_hat)
        
        # Check if all R-hat values are < 1.01 (convergence criterion)
        converged = all(r_hat < 1.01 for r_hat in r_hat_values)
        
        return {
            'r_hat_values': r_hat_values,
            'converged': converged,
            'max_r_hat': max(r_hat_values)
        }
    
    def save_results(self, results: Dict):
        """
        Save MCMC results to file.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary to save
        """
        # Create results directory if it doesn't exist
        os.makedirs('results/models', exist_ok=True)
        
        # Save results as pickle file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/models/mcmc_{self.treatment_name}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': results,
                'flat_samples': self.flat_samples,
                'sampler_state': self.sampler.get_chain()
            }, f)
        
        print(f"MCMC results saved to {filename}")
        
        # Save parameter estimates as CSV
        param_df = pd.DataFrame([
            {
                'parameter': param_name,
                'mean': estimates['mean'],
                'std': estimates['std'],
                'median': estimates['median'],
                'q16': estimates['q16'],
                'q84': estimates['q84']
            }
            for param_name, estimates in self.parameter_estimates.items()
        ])
        
        csv_filename = f"results/models/mcmc_parameters_{self.treatment_name}_{timestamp}.csv"
        param_df.to_csv(csv_filename, index=False)
        print(f"Parameter estimates saved to {csv_filename}")
    
    def plot_corner(self, save_plot: bool = True):
        """
        Create corner plot of posterior distributions.
        
        Parameters:
        -----------
        save_plot : bool
            Whether to save the plot
        """
        if self.flat_samples is None:
            raise ValueError("No samples available. Run MCMC first.")
        
        # Create corner plot
        fig = corner.corner(
            self.flat_samples,
            labels=self.model.param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        # Add title
        fig.suptitle(f'MCMC Posterior Distributions - {self.treatment_name}', 
                    fontsize=16, y=0.98)
        
        if save_plot:
            os.makedirs('results/visualization', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/visualization/mcmc_corner_{self.treatment_name}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Corner plot saved to {filename}")
        
        plt.show()
    
    def plot_chains(self, save_plot: bool = True):
        """
        Plot MCMC chains for visual convergence assessment.
        
        Parameters:
        -----------
        save_plot : bool
            Whether to save the plot
        """
        if self.sampler is None:
            raise ValueError("No sampler available. Run MCMC first.")
        
        # Get chains
        chains = self.sampler.get_chain()
        
        # Create subplots
        n_params = chains.shape[2]
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten()
        
        for i in range(n_params):
            ax = axes[i]
            
            # Plot chains for this parameter
            for j in range(min(10, self.n_walkers)):  # Plot first 10 walkers
                ax.plot(chains[:, j, i], alpha=0.3)
            
            ax.set_title(f'{self.model.param_names[i]}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Parameter Value')
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('results/visualization', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/visualization/mcmc_chains_{self.treatment_name}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Chain plot saved to {filename}")
        
        plt.show() 