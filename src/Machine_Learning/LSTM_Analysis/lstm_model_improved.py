"""
Long Short-Term Memory (LSTM) Network Implementation for Tumor Growth Prediction

This module implements the LSTM-based time series prediction model described in the paper:
"A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology"

The LSTM model serves as a machine learning validation tool for the mechanistic model,
providing complementary predictive capabilities for tumor growth dynamics across different
treatment regimens.

Key Features:
- Multi-treatment analysis (Saline, Untreated, MNP, MNPFDG)
- 75/25 train-test split as specified in the methodology
- Comprehensive hyperparameter optimization
- Convergence analysis with multiple random seeds
- Performance metrics and visualization

Author: Yeganeh Abdollahinejad, Amit K Chattopadhyay, et al.
Date: 2024-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
import os
import csv
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TumorGrowthLSTM:
    """
    LSTM-based model for tumor growth prediction across multiple treatment regimens.
    
    This class implements the LSTM validation approach described in the paper,
    providing a data-driven alternative to the mechanistic model for tumor growth
    prediction. The model is trained on 75% of the data and validated on the
    remaining 25%, following the methodology outlined in the paper.
    """
    
    def __init__(self, sequence_length: int = 10, n_features: int = 1):
        """
        Initialize the LSTM model for tumor growth prediction.
        
        Parameters:
        -----------
        sequence_length : int
            Number of time steps to use for prediction (default: 10)
        n_features : int
            Number of input features (default: 1 for time series)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        
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
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        This method transforms the time series data into sequences suitable for
        LSTM training, where each input sequence contains 'sequence_length' time
        steps and the target is the next value.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (time points)
        y : np.ndarray
            Target values (tumor volumes)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Input sequences and corresponding targets
        """
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            # Create input sequence
            X_seq = X[i-self.sequence_length:i]
            y_seq = y[i]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_lstm_model(self, n_lstm_layers: int = 2, lstm_units: List[int] = [128, 64],
                        dropout_rate: float = 0.2, learning_rate: float = 0.001) -> Sequential:
        """
        Build the LSTM model architecture.
        
        The model architecture follows the specifications in the paper:
        - Multiple LSTM layers with decreasing units
        - Dropout for regularization
        - Batch normalization for training stability
        - Dense output layer for regression
        
        Parameters:
        -----------
        n_lstm_layers : int
            Number of LSTM layers (default: 2)
        lstm_units : List[int]
            Number of units in each LSTM layer (default: [128, 64])
        dropout_rate : float
            Dropout rate for regularization (default: 0.2)
        learning_rate : float
            Learning rate for optimizer (default: 0.001)
            
        Returns:
        --------
        Sequential
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=(n_lstm_layers > 1),
            input_shape=(self.sequence_length, self.n_features),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, n_lstm_layers):
            return_sequences = (i < n_lstm_layers - 1)
            model.add(LSTM(
                units=lstm_units[i],
                return_sequences=return_sequences,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, treatment_name: str, test_size: float = 0.25) -> Dict:
        """
        Prepare data for a specific treatment group.
        
        This method loads the data, performs the 75/25 train-test split as specified
        in the methodology, and creates sequences suitable for LSTM training.
        
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
        
        # Scale the data
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'X_train_orig': X_train,
            'y_train_orig': y_train,
            'X_test_orig': X_test,
            'y_test_orig': y_test,
            'treatment_name': treatment_name,
            'color': treatment_config['color'],
            'marker': treatment_config['marker']
        }
    
    def train_model(self, data: Dict, epochs: int = 500, batch_size: int = 32,
                   validation_split: float = 0.2, patience: int = 50) -> Dict:
        """
        Train the LSTM model on the prepared data.
        
        This method implements the training procedure with early stopping and
        learning rate reduction to prevent overfitting and ensure convergence.
        
        Parameters:
        -----------
        data : Dict
            Prepared data dictionary
        epochs : int
            Maximum number of training epochs (default: 500)
        batch_size : int
            Batch size for training (default: 32)
        validation_split : float
            Proportion of training data for validation (default: 0.2)
        patience : int
            Patience for early stopping (default: 50)
            
        Returns:
        --------
        Dict
            Training results and model performance
        """
        # Build model
        self.model = self.build_lstm_model()
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            data['X_train'],
            data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(data['X_train'])
        y_test_pred_scaled = self.model.predict(data['X_test'])
        
        # Inverse transform predictions
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled).flatten()
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled).flatten()
        
        # Calculate metrics
        train_mse = mean_squared_error(data['y_train_orig'][self.sequence_length:], y_train_pred)
        test_mse = mean_squared_error(data['y_test_orig'][self.sequence_length:], y_test_pred)
        train_mae = mean_absolute_error(data['y_train_orig'][self.sequence_length:], y_train_pred)
        test_mae = mean_absolute_error(data['y_test_orig'][self.sequence_length:], y_test_pred)
        train_r2 = r2_score(data['y_train_orig'][self.sequence_length:], y_train_pred)
        test_r2 = r2_score(data['y_test_orig'][self.sequence_length:], y_test_pred)
        
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
            'y_train_orig': data['y_train_orig'][self.sequence_length:],
            'y_test_orig': data['y_test_orig'][self.sequence_length:],
            'X_train_orig': data['X_train_orig'][self.sequence_length:],
            'X_test_orig': data['X_test_orig'][self.sequence_length:],
            'color': data['color'],
            'marker': data['marker']
        }
        
        return results
    
    def plot_training_history(self, save_plot: bool = True):
        """
        Plot training history (loss and metrics).
        
        Parameters:
        -----------
        save_plot : bool
            Whether to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('results/visualization', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/visualization/lstm_training_history_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {filename}")
        
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
            filename = f"results/visualization/lstm_predictions_{results['treatment_name']}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {filename}")
        
        plt.show()
    
    def run_convergence_analysis(self, treatment_name: str, sample_sizes: List[int] = None,
                               n_seeds: int = 5) -> Dict:
        """
        Run convergence analysis with different sample sizes and random seeds.
        
        This method implements the convergence analysis described in Appendix II
        of the paper, evaluating model performance across different training
        dataset sizes and random initializations.
        
        Parameters:
        -----------
        treatment_name : str
            Name of the treatment group
        sample_sizes : List[int]
            List of sample sizes to test (default: [20, 40, 60, 80, 90])
        n_seeds : int
            Number of random seeds to test (default: 5)
            
        Returns:
        --------
        Dict
            Convergence analysis results
        """
        if sample_sizes is None:
            sample_sizes = [20, 40, 60, 80, 90]
        
        # Load full dataset
        treatment_config = self.treatments[treatment_name]
        df = pd.read_excel(treatment_config['file'])
        X_full = df['var1'].values.reshape(-1, 1)
        y_full = df['var2'].values
        
        results = {}
        
        for sample_size in sample_sizes:
            print(f"Testing sample size: {sample_size}")
            mse_values = []
            
            for seed in range(n_seeds):
                # Set random seed
                np.random.seed(seed)
                tf.random.set_seed(seed)
                
                # Sample data
                indices = np.random.choice(len(X_full), sample_size, replace=False)
                indices = np.sort(indices)
                X_sample = X_full[indices]
                y_sample = y_full[indices]
                
                # Prepare data
                data = self.prepare_data_from_arrays(X_sample, y_sample, treatment_name)
                
                # Train model
                try:
                    model_results = self.train_model(data, epochs=200)
                    mse_values.append(model_results['test_mse'])
                except Exception as e:
                    print(f"Error with seed {seed}: {e}")
                    mse_values.append(np.nan)
            
            results[sample_size] = {
                'mean_mse': np.nanmean(mse_values),
                'median_mse': np.nanmedian(mse_values),
                'std_mse': np.nanstd(mse_values),
                'mse_values': mse_values
            }
        
        return results
    
    def prepare_data_from_arrays(self, X: np.ndarray, y: np.ndarray, 
                                treatment_name: str) -> Dict:
        """
        Prepare data from numpy arrays (for convergence analysis).
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        treatment_name : str
            Treatment name
            
        Returns:
        --------
        Dict
            Prepared data dictionary
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, shuffle=False
        )
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'X_train_orig': X_train,
            'y_train_orig': y_train,
            'X_test_orig': X_test,
            'y_test_orig': y_test,
            'treatment_name': treatment_name,
            'color': self.treatments[treatment_name]['color'],
            'marker': self.treatments[treatment_name]['marker']
        }
    
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
        
        predictions_filename = f"results/predictions/lstm_predictions_{treatment_name}_{timestamp}.csv"
        predictions_df.to_csv(predictions_filename, index=False)
        
        # Save model
        model_filename = f"results/models/lstm_model_{treatment_name}_{timestamp}.h5"
        self.model.save(model_filename)
        
        # Save metrics
        metrics = {
            'treatment_name': treatment_name,
            'train_mse': results['train_mse'],
            'test_mse': results['test_mse'],
            'train_mae': results['train_mae'],
            'test_mae': results['test_mae'],
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2'],
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'timestamp': timestamp
        }
        
        metrics_filename = f"results/models/lstm_metrics_{treatment_name}_{timestamp}.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results saved for {treatment_name}:")
        print(f"  Predictions: {predictions_filename}")
        print(f"  Model: {model_filename}")
        print(f"  Metrics: {metrics_filename}")

def run_lstm_analysis_all_treatments():
    """
    Run LSTM analysis for all treatment groups.
    
    This function implements the complete LSTM analysis pipeline as described
    in the paper, training models for all treatment groups and generating
    comprehensive results and visualizations.
    """
    print("Starting LSTM analysis for all treatment groups...")
    
    # Initialize LSTM model
    lstm_model = TumorGrowthLSTM(sequence_length=10)
    
    # Treatment groups to analyze
    treatments = ['Saline', 'Untreated', 'MNP', 'MNPFDG']
    
    all_results = {}
    
    for treatment in treatments:
        print(f"\n{'='*50}")
        print(f"Analyzing {treatment} treatment...")
        print(f"{'='*50}")
        
        try:
            # Prepare data
            data = lstm_model.prepare_data(treatment)
            
            # Train model
            results = lstm_model.train_model(data)
            all_results[treatment] = results
            
            # Plot results
            lstm_model.plot_predictions(results)
            
            # Save results
            lstm_model.save_results(results, treatment)
            
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
    filename = f"results/visualization/lstm_all_treatments_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {filename}")
    
    plt.show()

if __name__ == "__main__":
    # Run the complete LSTM analysis
    results = run_lstm_analysis_all_treatments() 