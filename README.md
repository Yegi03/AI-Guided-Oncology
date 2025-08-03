# A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology

**Authors:** Yeganeh Abdollahinejad, Angelo Sajan Kalathil, Subhagata Chattopadhyay, Gillian Pearce, and Amit K. Chattopadhyay*


---

## Overview

This repository contains the complete implementation of the hybrid modeling framework described in the paper:

> **A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology**

We present a comprehensive computational framework that integrates:
- **Mechanistic mathematical modeling** (coupled ODEs) of nanoparticle-tumor interactions
- **Bayesian parameter estimation** using Markov Chain Monte Carlo (MCMC)
- **Machine learning validation** (LSTM and Neural ODE) for predictive tumor growth analysis
- **Multi-treatment comparison** (Untreated, Saline, MNP, MNPFDG)

All code and data are provided for full reproducibility of the results and figures in the manuscript.

---

## Key Features

### **Mechanistic Model**
- **5-State ODE System**: C (nanoparticles), C_b (bound), C_bs (binding sites), C_i (internalized), T (tumor)
- **Radial Discretization**: Proper boundary conditions and spherical geometry
- **16 Parameters**: Estimated via Bayesian inference with uncertainty quantification
- **Biological Interpretability**: All parameters have clear biological meaning

### **Bayesian MCMC Implementation**
- **128 Walkers**: Ensemble sampling for robust convergence
- **7000 Steps**: 2000 burn-in + 5000 main sampling
- **Convergence Diagnostics**: Gelman-Rubin (R-hat < 1.01)
- **Literature-Informed Priors**: Based on Goodman et al. (2008) and Graff & Wittrup (2003)

### **Machine Learning Validation**
- **LSTM Networks**: Sequence modeling with temporal dependencies
- **Neural ODE**: Continuous-time dynamics modeling
- **75/25 Split**: Consistent train-test methodology
- **Convergence Analysis**: Multiple random seeds and hyperparameter optimization

### **Treatment Analysis**
- **4 Treatment Groups**: Untreated, Saline, MNP, MNPFDG
- **Performance Ranking**: MNPFDG > MNP > Saline > Untreated
- **Clinical Interpretation**: Tumor growth rates and suppression efficacy

---

## Directory Structure

```
Submission/
├── README.md                           # Main documentation
├── README_METHODOLOGY.md               # Detailed methodology guide
├── requirements.txt                    # Python dependencies
├── project_structure.md                # Recommended structure
│
├── src/                                # Source code
│   ├── Bayesian/
│   │   ├── mechanistic_model.py        # Core ODE implementation
│   │   ├── bayesian_mcmc.py            # Complete MCMC framework
│   │   ├── mcmc_inference.py           # Legacy MCMC code
│   │   ├── mcmc.py                     # Legacy simplified MCMC
│   │   └── model_fit_all_treatments.py # Legacy multi-treatment
│   │
│   ├── Machine_Learning/
│   │   ├── LSTM_Analysis/
│   │   │   ├── lstm_model_improved.py  # Enhanced LSTM implementation
│   │   │   └── lstm_model.py           # Legacy LSTM code
│   │   └── Neural_ODE_Analysis/
│   │       ├── neural_ode_model_improved.py  # Enhanced Neural ODE
│   │       └── neural_ode_model.py           # Legacy Neural ODE
│   │
│   └── __init__.py
│
├── scripts/
│   ├── run_complete_analysis.py        # Main analysis pipeline
│   └── run_full_analysis.py            # Legacy analysis script
│
├── data/                               # All datasets
│   └── raw/                            # Original data files
│       ├── untreated85/untreated85.xlsx
│       ├── saline85/saline85.xlsx
│       ├── mnps85/mnps85.xlsx
│       └── mnfdg85/mnfdg85.xlsx
│
├── results/                            # Generated results
│   ├── models/                         # Trained models and parameters
│   ├── predictions/                    # Model predictions
│   ├── visualization/                  # Plots and figures
│   └── metrics/                        # Performance metrics
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:Yegi03/AI-Guided-Oncology.git
cd AI-Guided-Oncology
```

### 2. Set Up the Python Environment

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Files

All required data files are in the `data/raw/` directory, organized by treatment group.

---

## Reproducing the Results

### **Complete Analysis Pipeline (Recommended)**

Run the entire analysis pipeline with one command:

```bash
python scripts/run_complete_analysis.py
```

This will execute:
1. **Mechanistic Model Analysis**: ODE system with MCMC parameter estimation
2. **LSTM Validation**: Time series prediction with sequence modeling
3. **Neural ODE Validation**: Continuous-time dynamics modeling
4. **Comprehensive Comparison**: Method and treatment comparisons
5. **Results Generation**: All plots, metrics, and reports

### **Individual Components**

#### **A. Mechanistic Model & Bayesian Inference**

```bash
# Run MCMC for all treatments
python -c "
from src.Bayesian.mechanistic_model import TumorNanoparticleModel
from src.Bayesian.bayesian_mcmc import BayesianMCMCInference
# See README_METHODology.md for detailed examples
"
```

#### **B. Machine Learning Validation**

```bash
# Run LSTM analysis
python -c "
from src.Machine_Learning.LSTM_Analysis.lstm_model_improved import run_lstm_analysis_all_treatments
run_lstm_analysis_all_treatments()
"

# Run Neural ODE analysis
python -c "
from src.Machine_Learning.Neural_ODE_Analysis.neural_ode_model_improved import run_neural_ode_analysis_all_treatments
run_neural_ode_analysis_all_treatments()
"
```

---

## Key Results

### **Model Performance Ranking**
1. **MCMC Mechanistic Model**: MSE = 0.023 ± 0.008 (Best)
2. **LSTM Network**: MSE = 0.038 ± 0.011
3. **Neural ODE**: MSE = 0.049 ± 0.016

### **Treatment Efficacy Ranking**
1. **MNPFDG**: Most effective (slowest tumor growth, best long-term suppression)
2. **MNP**: Moderate effectiveness (moderate tumor growth inhibition)
3. **Saline**: Limited effectiveness (effective only for first 3 days)
4. **Untreated**: Rapid progression (no treatment, fastest tumor growth)

### **Methodology Compliance**
- **75/25 Train-Test Split**: Consistent across all methods
- **MCMC Parameters**: 128 walkers, 7000 steps, convergence diagnostics
- **Data Standardization**: Box-Cox normalization
- **Uncertainty Quantification**: Posterior distributions and credible intervals
- **Convergence Analysis**: Multiple random seeds and hyperparameter optimization

---

## Requirements

All dependencies are listed in `requirements.txt`. Key packages include:
- **Scientific Computing**: numpy, scipy, pandas
- **Machine Learning**: scikit-learn, tensorflow, torch, torchdiffeq
- **Bayesian Inference**: emcee, corner
- **Visualization**: matplotlib, seaborn
- **Data Handling**: openpyxl

Install with:
```bash
pip install -r requirements.txt
```

---

## Computational Requirements

### **Hardware Specifications**
- **Recommended**: NVIDIA A100 GPU (80GB HBM2e)
- **Minimum**: 16GB RAM, multi-core CPU
- **Storage**: 10GB for complete analysis

### **Computation Time**
- **MCMC**: ~8.5 hours per treatment (4 treatments = 34 hours)
- **LSTM**: ~3.8 hours total
- **Neural ODE**: ~4.2 hours total
- **Total**: ~42 hours for complete pipeline

---

## Documentation

- **README_METHODOLOGY.md**: Detailed methodology explanation and usage examples
- **project_structure.md**: Recommended project organization
- **Code Comments**: Comprehensive inline documentation in all source files

---

## Data and Code Availability

All data and modeling scripts are available in this repository. The implementation ensures complete methodology compliance with the paper specifications.

---

## Citation

If you use this code or data, please cite our paper:

```bibtex
@article{abdollahinejad2024kinetic,
  title={A Kinetic Model of Nanoparticle Transport in Tumors: Artificial Intelligence Guided Digital Oncology},
  author={Abdollahinejad, Yeganeh and Chattopadhyay, Amit K and Pearce, Gillian and Kalathil, Angelo Sajan},
  journal={Under Review},
  year={2024}
}
```

---

## Contact

For questions about the methodology or implementation please reach out to Yeganeh (yza5171@psu.edu) or Dr. Amit Chattopadhyay (amit.chattopadhyay@ncirl.ie)
We would be delighted to collaborate—if you have suggestions, ideas, or would like to contribute, please feel free to reach out, open an issue, or submit a pull request.
