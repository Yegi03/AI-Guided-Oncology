"""
Mechanistic Model Implementation for Nanoparticle Transport in Tumors

This module implements the continuum kinetic model described in the paper:
"A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology"

The model consists of a system of coupled ordinary differential equations (ODEs) that describe:
1. Nanoparticle diffusion and transport (C)
2. Binding/unbinding kinetics (C_b, C_bs) 
3. Internalization processes (C_i)
4. Tumor cell population dynamics (T)

Author: Yeganeh Abdollahinejad, Amit K Chattopadhyay, et al.
Date: 2024-2025
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class TumorNanoparticleModel:
    """
    Continuum kinetic model for nanoparticle transport in tumors.
    
    This class implements the mathematical model described in the paper, which captures
    the key biochemical processes governing drug transport within cancerous tumors.
    
    The model includes:
    - Radial diffusion of nanoparticles
    - Binding/unbinding kinetics with tumor cell receptors
    - Internalization of bound nanoparticles
    - Tumor cell population dynamics under treatment
    """
    
    def __init__(self, n_radial_points: int = 50):
        """
        Initialize the tumor-nanoparticle model.
        
        Parameters:
        -----------
        n_radial_points : int
            Number of radial discretization points (default: 50)
        """
        self.n_radial_points = n_radial_points
        self.r = np.linspace(0, 1, n_radial_points)  # Radial domain [0, 1]
        self.dr = self.r[1] - self.r[0]
        
        # Parameter names matching the paper
        self.param_names = [
            'D',        # Diffusion coefficient [mm²/s]
            'epsilon',  # Volumetric porosity
            'k_a',      # Association rate coefficient [s⁻¹]
            'C_bs_init', # Initial binding sites concentration
            'k_d',      # Dissociation rate coefficient [s⁻¹]
            'k_i',      # Internalization rate coefficient [s⁻¹]
            'a',        # Binding site growth rate
            'K',        # Binding site carrying capacity
            'alpha_1',  # Tumor growth inhibition by binding sites
            'alpha_2',  # Tumor growth inhibition by binding sites (alternative)
            'beta_1',   # Tumor growth inhibition by internalized drug
            'beta_2',   # Tumor growth inhibition by internalized drug (alternative)
            'k_1',      # Direct nanoparticle-tumor interaction rate
            'k_2',      # Nanoparticle-internalized drug interaction rate
            'c',        # Tumor growth rate [day⁻¹]
            'K_T'       # Tumor carrying capacity [cells/mm³]
        ]
        
    def system_of_equations(self, t: float, variables: np.ndarray, params: Tuple) -> np.ndarray:
        """
        Define the coupled ODE system for nanoparticle transport in tumors.
        
        This implements the exact equations from the paper:
        
        ∂C/∂t = (1/r²) ∂/∂r[D·ε·r² ∂/∂r(C/ε)] - k_a·C_bs·C/ε + k_d·C_b
        ∂C_b/∂t = k_a·C_bs·C/ε - k_d·C_b - k_i·C_b
        ∂C_bs/∂t = -k_a·C_bs·C/ε + k_d·C_b + k_i·C_b + a·C_bs(1-C_bs/K) - α·C_bs·T
        ∂C_i/∂t = k_i·C_b + r·C_bs·T - β·C_i·T - k_2·C·C_i
        ∂T/∂t = c·T(1-T/K_T) - α·C_bs·T - β·C_i·T - k_1·C·T
        
        Parameters:
        -----------
        t : float
            Current time point
        variables : np.ndarray
            Flattened array of state variables [C, C_b, C_bs, C_i, T]
        params : Tuple
            Model parameters in the order specified in param_names
            
        Returns:
        --------
        np.ndarray
            Time derivatives of all state variables
        """
        # Unpack parameters
        (D, epsilon, k_a, C_bs_init, k_d, k_i, a, K,
         alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T) = params
        
        # Split variables into individual state vectors
        n = self.n_radial_points
        C = variables[0:n]           # Nanoparticle concentration
        C_b = variables[n:2*n]       # Bound nanoparticle concentration
        C_bs = variables[2*n:3*n]    # Available binding sites
        C_i = variables[3*n:4*n]     # Internalized nanoparticle concentration
        T = variables[4*n:5*n]       # Tumor cell population
        
        # Initialize derivative arrays
        dCdt = np.zeros_like(C)
        dC_bdt = np.zeros_like(C_b)
        dC_bsdt = np.zeros_like(C_bs)
        dC_idt = np.zeros_like(C_i)
        dTdt = np.zeros_like(T)
        
        # Equation 1: Nanoparticle diffusion and binding (C)
        # ∂C/∂t = (1/r²) ∂/∂r[D·ε·r² ∂/∂r(C/ε)] - k_a·C_bs·C/ε + k_d·C_b
        C_over_epsilon = C / epsilon
        for i in range(1, n-1):
            # Radial diffusion term with proper discretization
            diffusion_term = (1 / (self.r[i]**2)) * (
                D * epsilon * self.r[i]**2 * (
                    C_over_epsilon[i+1] - 2*C_over_epsilon[i] + C_over_epsilon[i-1]
                ) / (self.dr**2)
            )
            # Binding/unbinding terms
            binding_term = k_a * C_bs[i] * C[i] / epsilon
            unbinding_term = k_d * C_b[i]
            
            dCdt[i] = diffusion_term - binding_term + unbinding_term
        
        # No-flux boundary conditions at r=0 and r=1
        dCdt[0] = 0
        dCdt[-1] = 0
        
        # Equation 2: Bound nanoparticle dynamics (C_b)
        # ∂C_b/∂t = k_a·C_bs·C/ε - k_d·C_b - k_i·C_b
        for i in range(1, n-1):
            binding_term = k_a * C_bs[i] * C[i] / epsilon
            unbinding_term = k_d * C_b[i]
            internalization_term = k_i * C_b[i]
            
            dC_bdt[i] = binding_term - unbinding_term - internalization_term
        
        # Boundary conditions
        dC_bdt[0] = 0
        dC_bdt[-1] = 0
        
        # Equation 3: Available binding sites dynamics (C_bs)
        # ∂C_bs/∂t = -k_a·C_bs·C/ε + k_d·C_b + k_i·C_b + a·C_bs(1-C_bs/K) - α·C_bs·T
        for i in range(1, n-1):
            binding_loss = k_a * C_bs[i] * C[i] / epsilon
            unbinding_gain = k_d * C_b[i]
            internalization_gain = k_i * C_b[i]
            growth_term = a * C_bs[i] * (1 - C_bs[i] / K)
            tumor_inhibition = alpha_1 * C_bs[i] * T[i]
            
            dC_bsdt[i] = -binding_loss + unbinding_gain + internalization_gain + growth_term - tumor_inhibition
        
        # Boundary conditions
        dC_bsdt[0] = 0
        dC_bsdt[-1] = 0
        
        # Equation 4: Internalized nanoparticle dynamics (C_i)
        # ∂C_i/∂t = k_i·C_b + r·C_bs·T - β·C_i·T - k_2·C·C_i
        for i in range(1, n-1):
            internalization_gain = k_i * C_b[i]
            spatial_term = self.r[i] * C_bs[i] * T[i]
            tumor_inhibition = beta_1 * C_i[i] * T[i]
            interaction_loss = k_2 * C[i] * C_i[i]
            
            dC_idt[i] = internalization_gain + spatial_term - tumor_inhibition - interaction_loss
        
        # Boundary conditions
        dC_idt[0] = 0
        dC_idt[-1] = 0
        
        # Equation 5: Tumor cell population dynamics (T)
        # ∂T/∂t = c·T(1-T/K_T) - α·C_bs·T - β·C_i·T - k_1·C·T
        for i in range(1, n-1):
            logistic_growth = c * T[i] * (1 - T[i] / K_T)
            binding_inhibition = alpha_2 * C_bs[i] * T[i]
            internalized_inhibition = beta_2 * C_i[i] * T[i]
            direct_inhibition = k_1 * C[i] * T[i]
            
            dTdt[i] = logistic_growth - binding_inhibition - internalized_inhibition - direct_inhibition
        
        # Boundary conditions
        dTdt[0] = 0
        dTdt[-1] = 0
        
        # Return flattened array of all derivatives
        return np.concatenate([dCdt, dC_bdt, dC_bsdt, dC_idt, dTdt])
    
    def compute_tumor_volume(self, solution: object) -> np.ndarray:
        """
        Compute tumor volume from the model solution.
        
        Tumor volume is calculated by integrating the tumor cell population
        over the radial domain, accounting for the spherical geometry.
        
        Parameters:
        -----------
        solution : scipy.integrate.OdeSolution
            Solution object from solve_ivp
            
        Returns:
        --------
        np.ndarray
            Tumor volume at each time point
        """
        volumes = []
        
        for t_idx in range(len(solution.t)):
            # Extract tumor cell population at this time point
            T = solution.y[4*self.n_radial_points:5*self.n_radial_points, t_idx]
            
            # Integrate over radial domain (spherical geometry)
            # Volume = ∫₀¹ 4πr² T(r) dr
            volume = 0
            for i in range(self.n_radial_points):
                r = self.r[i]
                volume += 4 * np.pi * r**2 * T[i] * self.dr
            
            volumes.append(volume)
        
        return np.array(volumes)
    
    def set_initial_conditions(self, params: Tuple) -> np.ndarray:
        """
        Set initial conditions for the model variables.
        
        Parameters:
        -----------
        params : Tuple
            Model parameters including C_bs_init
            
        Returns:
        --------
        np.ndarray
            Initial conditions for all state variables
        """
        _, _, _, C_bs_init, _, _, _, _, _, _, _, _, _, _, _, _ = params
        
        # Initial conditions (all zeros except binding sites)
        C_init = np.zeros(self.n_radial_points)
        C_b_init = np.zeros(self.n_radial_points)
        C_bs_init_array = C_bs_init * np.ones(self.n_radial_points)
        C_i_init = np.zeros(self.n_radial_points)
        T_init = 100 * np.ones(self.n_radial_points)  # Initial tumor size
        
        return np.concatenate([C_init, C_b_init, C_bs_init_array, C_i_init, T_init])
    
    def solve_model(self, params: Tuple, time_span: Tuple, 
                   initial_conditions: Optional[np.ndarray] = None) -> object:
        """
        Solve the model equations using scipy.integrate.solve_ivp.
        
        Parameters:
        -----------
        params : Tuple
            Model parameters
        time_span : Tuple
            Time span (t_start, t_end)
        initial_conditions : np.ndarray, optional
            Initial conditions (if None, will be set automatically)
            
        Returns:
        --------
        scipy.integrate.OdeSolution
            Solution object containing time points and state variables
        """
        if initial_conditions is None:
            initial_conditions = self.set_initial_conditions(params)
        
        # Solve the ODE system
        solution = solve_ivp(
            fun=lambda t, y: self.system_of_equations(t, y, params),
            t_span=time_span,
            y0=initial_conditions,
            method='RK45',
            t_eval=np.linspace(time_span[0], time_span[1], 100),
            rtol=1e-8,
            atol=1e-10
        )
        
        return solution
    
    def get_parameter_bounds(self) -> List[Tuple]:
        """
        Get parameter bounds for Bayesian inference.
        
        These bounds are based on biological plausibility and literature values
        from Goodman et al. (2008) and Graff & Wittrup (2003).
        
        Returns:
        --------
        List[Tuple]
            List of (min, max) bounds for each parameter
        """
        bounds = [
            (1e-6, 1e-2),    # D: Diffusion coefficient [mm²/s]
            (0.1, 0.9),      # epsilon: Porosity
            (1e-3, 1e-1),    # k_a: Association rate [s⁻¹]
            (0.1, 1.0),      # C_bs_init: Initial binding sites
            (1e-3, 1e-1),    # k_d: Dissociation rate [s⁻¹]
            (1e-3, 1e-1),    # k_i: Internalization rate [s⁻¹]
            (0.1, 1.0),      # a: Binding site growth rate
            (0.1, 1.0),      # K: Carrying capacity
            (0.1, 1.0),      # alpha_1: Tumor inhibition by binding sites
            (0.1, 1.0),      # alpha_2: Tumor inhibition by binding sites (alt)
            (0.1, 1.0),      # beta_1: Tumor inhibition by internalized drug
            (0.1, 1.0),      # beta_2: Tumor inhibition by internalized drug (alt)
            (1e-3, 1e-1),    # k_1: Direct nanoparticle-tumor interaction
            (1e-3, 1e-1),    # k_2: Nanoparticle-internalized interaction
            (0.1, 1.0),      # c: Tumor growth rate [day⁻¹]
            (100, 1000)      # K_T: Tumor carrying capacity [cells/mm³]
        ]
        
        return bounds 