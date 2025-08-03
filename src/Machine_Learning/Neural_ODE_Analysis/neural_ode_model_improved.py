"""
Neural Ordinary Differential Equations (Neural ODE) Implementation for Tumor Growth Prediction

This module implements the Neural ODE-based time series prediction model described in the paper:
"A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology"

The Neural ODE model serves as a complementary machine learning validation tool that explicitly
models continuous-time dynamics, providing a bridge between traditional ODE modeling and
deep learning approaches.

Key Features:
- Continuous-time dynamics modeling
- Multi-treatment analysis (Saline, Untreated, MNP, MNPFDG)
- 75/25 train-test split as specified in the methodology
- Learning rate sensitivity analysis
- Convergence diagnostics and uncertainty quantification

Author: Yeganeh Abdollahinejad, Amit K Chattopadhyay, et al.
Date: 2024-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import warnings
import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import json

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ODEFunc(nn.Module):
    """
    Neural network function for the ODE system.
    
    This class defines the neural network that parameterizes the continuous-time
    dynamics in the Neural ODE framework. The network takes the current state
    and outputs the time derivative.
    """
    
    def __init__(self, dim: int, depth: int = 2, hidden_dims: List[int] = [32, 16], 
                 dropout: float = 0.1, activation: str = 'tanh'):
        """
        Initialize the ODE function network.
        
        Parameters:
        -----------
        dim : int
            Input/output dimension
        depth : int
            Number of hidden layers
        hidden_dims : List[int]
            Dimensions of hidden layers
        dropout : float
            Dropout rate for regularization
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid')
        """
        super().__init__()
        
        # Set activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        input_dim = dim
        
        for i in range(depth):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(self.activation)
            
            # Dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dims[i]
        
        # Output layer
        layers.append(nn.Linear(input_dim, dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE function.
        
        Parameters:
        -----------
        t : torch.Tensor
            Time point (not used in this implementation)
        x : torch.Tensor
            Current state vector
            
        Returns:
        --------
        torch.Tensor
            Time derivative of the state
        """
        return self.net(x)

class ODEBlock(nn.Module):
    """
    Neural ODE block that integrates the ODE function.
    
    This class wraps the ODE function with an ODE solver, allowing the network
    to model continuous-time dynamics.
    """
    
    def __init__(self, odefunc: ODEFunc, solver: str = 'rk4', rtol: float = 1e-5, 
                 atol: float = 1e-7, integration_time: float = 1.0):
        """
        Initialize the ODE block.
        
        Parameters:
        -----------
        odefunc : ODEFunc
            The ODE function network
        solver : str
            ODE solver method ('rk4', 'dopri5', 'euler')
        rtol : float
            Relative tolerance for adaptive solvers
        atol : float
            Absolute tolerance for adaptive solvers
        integration_time : float
            Integration time step
        """
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.integration_time = integration_time
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ODE block.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input state vector
            
        Returns:
        --------
        torch.Tensor
            Evolved state vector
        """
        # Define time points for integration
        t = torch.tensor([0.0, self.integration_time], dtype=torch.float32).to(x.device)
        
        # Solve ODE
        out = odeint(
            self.odefunc, 
            x, 
            t, 
            method=self.solver, 
            rtol=self.rtol, 
            atol=self.atol
        )
        
        return out[1]  # Return final state

class NeuralODERegressor(nn.Module):
    """
    Complete Neural ODE regressor for tumor growth prediction.
    
    This class combines the ODE block with a readout layer to create a complete
    regression model for time series prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [32, 16], 
                 depth: int = 2, dropout: float = 0.1, solver: str = 'rk4',
                 rtol: float = 1e-5, atol: float = 1e-7, activation: str = 'tanh'):
        """
        Initialize the Neural ODE regressor.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension
        hidden_dims : List[int]
            Hidden layer dimensions
        depth : int
            Number of hidden layers
        dropout : float
            Dropout rate
        solver : str
            ODE solver method
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        activation : str
            Activation function
        """
        super().__init__()
        
        # Create ODE function
        odefunc = ODEFunc(
            dim=input_dim,
            depth=depth,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation
        )
        
        # Create ODE block
        self.odeblock = ODEBlock(
            odefunc=odefunc,
            solver=solver,
            rtol=rtol,
            atol=atol
        )
        
        # Readout layer
        self.readout = nn.Linear(input_dim, 1)
        
        # Initialize readout weights
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Neural ODE regressor.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features
            
        Returns:
        --------
        torch.Tensor
            Predicted output
        """
        # Pass through ODE block
        x_evolved = self.odeblock(x)
        
        # Readout layer
        output = self.readout(x_evolved)
        
        return output.squeeze(-1)

class TumorGrowthNeuralODE:
    """
    Neural ODE-based model for tumor growth prediction.
    
    This class implements the Neural ODE validation approach described in the paper,
    providing a continuous-time modeling alternative to the mechanistic model.
    """
    
    def __init__(self, lag: int = 6, hidden_dims: List[int] = [32, 16], 
                 depth: int = 2, dropout: float = 0.1):
        """
        Initialize the Neural ODE model.
        
        Parameters:
        -----------
        lag : int
            Number of lagged time steps to use as input
        hidden_dims : List[int]
            Hidden layer dimensions
        depth : int
            Number of hidden layers
        dropout : float
            Dropout rate
        """
        self.lag = lag
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.dropout = dropout
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
        # Treatment configurations as specified in the paper
        self.treatments = {
            'Saline': {
                'file': 'data/raw/saline85/saline85.xlsx',
                'color': 'tab:blue',
                'marker': 'o'
            },
            'Untreated': {
                'file': 'data/raw/untreated85/untreated85.xlsx',
                'color': 'darkorange',
                'marker': 's'
            },
            'MNP': {
                'file': 'data/raw/mnps85/mnps85.xlsx',
                'color': 'forestgreen',
                'marker': '^'
            },
            'MNPFDG': {
                'file': 'data/raw/mnfdg85/mnfdg85.xlsx',
                'color': 'firebrick',
                'marker': 'D'
            }
        }
    
    def create_lagged_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged features for time series prediction.
        
        This method creates input features by concatenating lagged values of the
        target variable, allowing the model to capture temporal dependencies.
        
        Parameters:
        -----------
        X : np.ndarray
            Time points
        y : np.ndarray
            Target values (tumor volumes)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Lagged features and corresponding targets
        """
        X_lagged, y_lagged = [], []
        
        for i in range(self.lag, len(X)):
            # Create lagged feature vector
            lagged_feature = np.concatenate([
                y[i-self.lag:i].flatten(),  # Lagged target values
                X[i].flatten()              # Current time point
            ])
            
            X_lagged.append(lagged_feature)
            y_lagged.append(y[i])
        
        return np.array(X_lagged), np.array(y_lagged)
    
    def prepare_data(self, treatment_name: str, test_size: float = 0.25) -> Dict:
        """
        Prepare data for a specific treatment group.
        
        This method implements the 75/25 train-test split as specified in the
        methodology and creates lagged features suitable for Neural ODE training.
        
        Parameters:
        -----------
        treatment_name : str
            Name of the treatment group
        test_size : float
            Proportion of data for testing (default: 0.25)
            
        Returns:
        --------
        Dict
            Dictionary containing prepared data
        """
        # Load data
        treatment_config = self.treatments[treatment_name]
        df = pd.read_excel(treatment_config['file'])
        
        # Extract features and targets
        X = df['var1'].values.reshape(-1, 1)  # Time points
        y = df['var2'].values  # Tumor volumes
        
        # Split data (75/25 as specified in methodology)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Create lagged features
        X_train_lagged, y_train_lagged = self.create_lagged_features(X_train, y_train)
        X_test_lagged, y_test_lagged = self.create_lagged_features(X_test, y_test)
        
        # Scale the data
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train_lagged)
        X_test_scaled = self.scaler_X.transform(X_test_lagged)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train_lagged.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test_lagged.reshape(-1, 1)).flatten()
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'X_train_orig': X_train[self.lag:],
            'y_train_orig': y_train[self.lag:],
            'X_test_orig': X_test[self.lag:],
            'y_test_orig': y_test[self.lag:],
            'treatment_name': treatment_name,
            'color': treatment_config['color'],
            'marker': treatment_config['marker']
        }
    
    def train_model(self, data: Dict, learning_rate: float = 0.001, epochs: int = 1000,
                   batch_size: int = 32, patience: int = 50, validation_split: float = 0.2) -> Dict:
        """
        Train the Neural ODE model.
        
        This method implements the training procedure with early stopping and
        learning rate scheduling to ensure convergence and prevent overfitting.
        
        Parameters:
        -----------
        data : Dict
            Prepared data dictionary
        learning_rate : float
            Learning rate for optimization
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
        validation_split : float
            Proportion of training data for validation
            
        Returns:
        --------
        Dict
            Training results and model performance
        """
        # Convert data to PyTorch tensors
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.FloatTensor(data['y_train'])
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.FloatTensor(data['y_test'])
        
        # Create validation split
        n_val = int(len(X_train) * validation_split)
        X_val = X_train[:n_val]
        y_val = y_train[:n_val]
        X_train = X_train[n_val:]
        y_train = y_train[n_val:]
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = NeuralODERegressor(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            dropout=self.dropout
        )
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Record losses
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
            
            # Check early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_train_pred_scaled = self.model(X_train).numpy()
            y_test_pred_scaled = self.model(X_test).numpy()
        
        # Inverse transform predictions
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        train_mse = mean_squared_error(data['y_train_orig'], y_train_pred)
        test_mse = mean_squared_error(data['y_test_orig'], y_test_pred)
        train_mae = mean_absolute_error(data['y_train_orig'], y_train_pred)
        test_mae = mean_absolute_error(data['y_test_orig'], y_test_pred)
        train_r2 = r2_score(data['y_train_orig'], y_train_pred)
        test_r2 = r2_score(data['y_test_orig'], y_test_pred)
        
        results = {
            'treatment_name': data['treatment_name'],
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_train_orig': data['y_train_orig'],
            'y_test_orig': data['y_test_orig'],
            'X_train_orig': data['X_train_orig'],
            'X_test_orig': data['X_test_orig'],
            'color': data['color'],
            'marker': data['marker'],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rate': learning_rate
        }
        
        return results
    
    def run_learning_rate_analysis(self, treatment_name: str, 
                                  learning_rates: List[float] = None) -> Dict:
        """
        Run learning rate sensitivity analysis.
        
        This method implements the learning rate analysis described in Appendix II
        of the paper, evaluating model performance across different learning rates
        to identify optimal training parameters.
        
        Parameters:
        -----------
        treatment_name : str
            Name of the treatment group
        learning_rates : List[float]
            List of learning rates to test
            
        Returns:
        --------
        Dict
            Learning rate analysis results
        """
        if learning_rates is None:
            learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        
        # Prepare data
        data = self.prepare_data(treatment_name)
        
        results = {}
        
        for lr in learning_rates:
            print(f"Testing learning rate: {lr}")
            
            try:
                # Train model with current learning rate
                model_results = self.train_model(data, learning_rate=lr, epochs=500)
                
                results[lr] = {
                    'test_mse': model_results['test_mse'],
                    'test_r2': model_results['test_r2'],
                    'final_train_loss': model_results['train_losses'][-1] if model_results['train_losses'] else np.nan,
                    'final_val_loss': model_results['val_losses'][-1] if model_results['val_losses'] else np.nan
                }
                
            except Exception as e:
                print(f"Error with learning rate {lr}: {e}")
                results[lr] = {
                    'test_mse': np.nan,
                    'test_r2': np.nan,
                    'final_train_loss': np.nan,
                    'final_val_loss': np.nan
                }
        
        return results
    
    def plot_learning_rate_analysis(self, lr_results: Dict, save_plot: bool = True):
        """
        Plot learning rate analysis results.
        
        Parameters:
        -----------
        lr_results : Dict
            Learning rate analysis results
        save_plot : bool
            Whether to save the plot
        """
        lrs = list(lr_results.keys())
        mse_values = [lr_results[lr]['test_mse'] for lr in lrs]
        r2_values = [lr_results[lr]['test_r2'] for lr in lrs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot MSE vs learning rate
        ax1.semilogx(lrs, mse_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Test MSE')
        ax1.set_title('Test MSE vs Learning Rate')
        ax1.grid(True, alpha=0.3)
        
        # Plot R² vs learning rate
        ax2.semilogx(lrs, r2_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Test R² vs Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('results/visualization', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/visualization/neural_ode_lr_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Learning rate analysis plot saved to {filename}")
        
        plt.show()
    
    def plot_predictions(self, results: Dict, save_plot: bool = True):
        """
        Plot model predictions vs actual values.
        
        Parameters:
        -----------
        results : Dict
            Training results dictionary
        save_plot : bool
            Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training predictions
        ax1.scatter(results['X_train_orig'], results['y_train_orig'], 
                   color=results['color'], marker=results['marker'], 
                   alpha=0.7, s=50, label='Actual (Train)')
        ax1.scatter(results['X_train_orig'], results['y_train_pred'], 
                   color='black', marker='x', s=50, label='Predicted (Train)')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Tumor Volume (mm³)')
        ax1.set_title(f'{results["treatment_name"]} - Training Set\nMSE: {results["train_mse"]:.4f}, R²: {results["train_r2"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test predictions
        ax2.scatter(results['X_test_orig'], results['y_test_orig'], 
                   color=results['color'], marker=results['marker'], 
                   alpha=0.7, s=50, label='Actual (Test)')
        ax2.scatter(results['X_test_orig'], results['y_test_pred'], 
                   color='black', marker='x', s=50, label='Predicted (Test)')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Tumor Volume (mm³)')
        ax2.set_title(f'{results["treatment_name"]} - Test Set\nMSE: {results["test_mse"]:.4f}, R²: {results["test_r2"]:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('results/visualization', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/visualization/neural_ode_predictions_{results['treatment_name']}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {filename}")
        
        plt.show()
    
    def save_results(self, results: Dict, treatment_name: str):
        """
        Save model results to file.
        
        Parameters:
        -----------
        results : Dict
            Model results
        treatment_name : str
            Treatment name
        """
        # Create results directory
        os.makedirs('results/predictions', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'time': np.concatenate([results['X_train_orig'], results['X_test_orig']]),
            'actual': np.concatenate([results['y_train_orig'], results['y_test_orig']]),
            'predicted': np.concatenate([results['y_train_pred'], results['y_test_pred']]),
            'set': ['train'] * len(results['y_train_orig']) + ['test'] * len(results['y_test_orig'])
        })
        
        predictions_filename = f"results/predictions/neural_ode_predictions_{treatment_name}_{timestamp}.csv"
        predictions_df.to_csv(predictions_filename, index=False)
        
        # Save model
        model_filename = f"results/models/neural_ode_model_{treatment_name}_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_filename)
        
        # Save metrics
        metrics = {
            'treatment_name': treatment_name,
            'train_mse': results['train_mse'],
            'test_mse': results['test_mse'],
            'train_mae': results['train_mae'],
            'test_mae': results['test_mae'],
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2'],
            'lag': self.lag,
            'hidden_dims': self.hidden_dims,
            'depth': self.depth,
            'dropout': self.dropout,
            'learning_rate': results['learning_rate'],
            'timestamp': timestamp
        }
        
        metrics_filename = f"results/models/neural_ode_metrics_{treatment_name}_{timestamp}.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results saved for {treatment_name}:")
        print(f"  Predictions: {predictions_filename}")
        print(f"  Model: {model_filename}")
        print(f"  Metrics: {metrics_filename}")

def run_neural_ode_analysis_all_treatments():
    """
    Run Neural ODE analysis for all treatment groups.
    
    This function implements the complete Neural ODE analysis pipeline as described
    in the paper, training models for all treatment groups and generating
    comprehensive results and visualizations.
    """
    print("Starting Neural ODE analysis for all treatment groups...")
    
    # Initialize Neural ODE model
    neural_ode_model = TumorGrowthNeuralODE(
        lag=6,
        hidden_dims=[32, 16],
        depth=2,
        dropout=0.1
    )
    
    # Treatment groups to analyze
    treatments = ['Saline', 'Untreated', 'MNP', 'MNPFDG']
    
    all_results = {}
    
    for treatment in treatments:
        print(f"\n{'='*50}")
        print(f"Analyzing {treatment} treatment...")
        print(f"{'='*50}")
        
        try:
            # Prepare data
            data = neural_ode_model.prepare_data(treatment)
            
            # Train model
            results = neural_ode_model.train_model(data, learning_rate=1e-3)
            all_results[treatment] = results
            
            # Plot results
            neural_ode_model.plot_predictions(results)
            
            # Save results
            neural_ode_model.save_results(results, treatment)
            
            # Print summary
            print(f"\n{treatment} Results:")
            print(f"  Train MSE: {results['train_mse']:.4f}")
            print(f"  Test MSE: {results['test_mse']:.4f}")
            print(f"  Train R²: {results['train_r2']:.4f}")
            print(f"  Test R²: {results['test_r2']:.4f}")
            
        except Exception as e:
            print(f"Error analyzing {treatment}: {e}")
            continue
    
    # Create comparison plot
    create_comparison_plot(all_results)
    
    return all_results

def create_comparison_plot(all_results: Dict):
    """
    Create comparison plot for all treatments.
    
    Parameters:
    -----------
    all_results : Dict
        Results for all treatment groups
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (treatment, results) in enumerate(all_results.items()):
        ax = axes[i]
        
        # Plot training data
        ax.scatter(results['X_train_orig'], results['y_train_orig'], 
                  color=results['color'], marker=results['marker'], 
                  alpha=0.7, s=30, label='Train (Actual)')
        ax.scatter(results['X_train_orig'], results['y_train_pred'], 
                  color='black', marker='x', s=30, label='Train (Pred)')
        
        # Plot test data
        ax.scatter(results['X_test_orig'], results['y_test_orig'], 
                  color=results['color'], marker=results['marker'], 
                  alpha=0.7, s=30, label='Test (Actual)')
        ax.scatter(results['X_test_orig'], results['y_test_pred'], 
                  color='black', marker='x', s=30, label='Test (Pred)')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_title(f'{treatment}\nTest MSE: {results["test_mse"]:.4f}, R²: {results["test_r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results/visualization', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/visualization/neural_ode_all_treatments_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {filename}")
    
    plt.show()

if __name__ == "__main__":
    # Run the complete Neural ODE analysis
    results = run_neural_ode_analysis_all_treatments() 