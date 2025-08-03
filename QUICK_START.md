# Quick Start Guide

## Get Started in 5 Minutes

This guide will get you up and running with the tumor-nanoparticle model analysis quickly.

## Prerequisites

- Python 3.8 or higher
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU (optional, but recommended for faster processing)

## Step 1: Setup Environment

```bash
# Clone the repository
git clone git@github.com:Yegi03/AI-Guided-Oncology.git
cd AI-Guided-Oncology

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify Data Structure

Ensure your data files are in the correct location:

```
data/raw/
├── untreated85/untreated85.xlsx
├── saline85/saline85.xlsx
├── mnps85/mnps85.xlsx
└── mnfdg85/mnfdg85.xlsx
```

Each Excel file should contain:
- `var1`: Time points (days)
- `var2`: Tumor volume (mm³)

## Step 3: Run Complete Analysis

```bash
# Run the entire analysis pipeline
python scripts/run_complete_analysis.py
```

This will:
- Run MCMC analysis for all treatments
- Train LSTM models
- Train Neural ODE models
- Generate comparison plots
- Save all results

**Expected Runtime**: ~42 hours (can be reduced with GPU)

## Step 4: View Results

Results will be saved in the `results/` directory:

```
results/
├── models/                    # Trained models and parameters
├── predictions/              # Model predictions
├── visualization/            # Plots and figures
└── metrics/                  # Performance metrics
```

## Quick Examples

### Run Just MCMC Analysis

```python
from src.Bayesian.mechanistic_model import TumorNanoparticleModel
from src.Bayesian.bayesian_mcmc import BayesianMCMCInference
import pandas as pd

# Load data
df = pd.read_excel('data/raw/mnfdg85/mnfdg85.xlsx')
time_data = df['var1'].values
volume_data = df['var2'].values

# Initialize model and MCMC
model = TumorNanoparticleModel()
mcmc = BayesianMCMCInference(model, "MNPFDG")

# Run MCMC
results = mcmc.run_mcmc(time_data, volume_data)

# View results
print(f"Converged: {results['convergence_diagnostics']['converged']}")
print(f"Acceptance fraction: {results['acceptance_fraction']:.3f}")
```

### Run Just LSTM Analysis

```python
from src.Machine_Learning.LSTM_Analysis.lstm_model_improved import TumorGrowthLSTM

# Initialize LSTM
lstm = TumorGrowthLSTM(sequence_length=10)

# Prepare data
data = lstm.prepare_data("MNPFDG")

# Train model
results = lstm.train_model(data)

# View results
print(f"Test MSE: {results['test_mse']:.4f}")
print(f"Test R²: {results['test_r2']:.4f}")
```

### Run Just Neural ODE Analysis

```python
from src.Machine_Learning.Neural_ODE_Analysis.neural_ode_model_improved import TumorGrowthNeuralODE

# Initialize Neural ODE
node = TumorGrowthNeuralODE(lag=6)

# Prepare data
data = node.prepare_data("MNPFDG")

# Train model
results = node.train_model(data)

# View results
print(f"Test MSE: {results['test_mse']:.4f}")
print(f"Test R²: {results['test_r2']:.4f}")
```

## Expected Results

### Model Performance
- **MCMC**: MSE ≈ 0.023 ± 0.008 (Best)
- **LSTM**: MSE ≈ 0.038 ± 0.011
- **Neural ODE**: MSE ≈ 0.049 ± 0.016

### Treatment Efficacy
1. **MNPFDG**: Most effective
2. **MNP**: Moderate effectiveness
3. **Saline**: Limited effectiveness
4. **Untreated**: Rapid progression

## Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError`
```bash
# Ensure you're in the correct directory
cd Submission
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Memory Error**: `MemoryError`
```bash
# Reduce batch sizes or use smaller models
# For MCMC: Reduce n_walkers from 128 to 64
# For LSTM: Reduce sequence_length from 10 to 5
```

**Data Not Found**: `FileNotFoundError`
```bash
# Check data file paths
ls data/raw/*/*.xlsx
# Ensure file names match exactly
```

### Performance Optimization

**GPU Acceleration**:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Parallel Processing**:
```python
# Run treatments in parallel
import multiprocessing as mp
# See README_METHODOLOGY.md for details
```

## Next Steps

1. **Read README_METHODOLOGY.md** for detailed methodology explanation
2. **Explore individual components** using the examples above
3. **Customize parameters** for your specific needs
4. **Generate custom visualizations** using the plotting functions

## Support

- **Documentation**: README_METHODOLOGY.md
- **Legacy Code**: LEGACY_CODE_README.md
- **Issues**: Check troubleshooting section above

---

**Note**: This quick start guide uses the improved implementation. For legacy code, see LEGACY_CODE_README.md. 