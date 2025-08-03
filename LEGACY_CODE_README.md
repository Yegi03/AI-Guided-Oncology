# Legacy Code Documentation

## Overview

This document explains the legacy code files in the project and their relationship to the new improved implementation.

## Legacy vs. Improved Implementation

### New Improved Implementation (Recommended)

The following files represent the **complete, methodology-compliant implementation**:

- `src/Bayesian/mechanistic_model.py` - Core ODE system implementation
- `src/Bayesian/bayesian_mcmc.py` - Complete MCMC framework
- `src/Machine_Learning/LSTM_Analysis/lstm_model_improved.py` - Enhanced LSTM
- `src/Machine_Learning/Neural_ODE_Analysis/neural_ode_model_improved.py` - Enhanced Neural ODE
- `scripts/run_complete_analysis.py` - Complete analysis pipeline

### Legacy Code Files

The following files are **legacy implementations** that were developed earlier:

#### Bayesian Analysis
- `src/Bayesian/mcmc_inference.py` - Original MCMC implementation (495 lines)
- `src/Bayesian/mcmc.py` - Simplified MCMC implementation (291 lines)
- `src/Bayesian/model_fit_all_treatments.py` - Multi-treatment analysis (522 lines)

#### Machine Learning
- `src/Machine_Learning/LSTM_Analysis/lstm_model.py` - Original LSTM implementation (88 lines)
- `src/Machine_Learning/Neural_ODE_Analysis/neural_ode_model.py` - Original Neural ODE implementation (216 lines)

#### Scripts
- `scripts/run_full_analysis.py` - Legacy analysis script (94 lines)

## Key Differences

### Legacy Implementation
- **Working Code**: All files are functional and produce results
- **Limited Documentation**: Minimal commenting and documentation
- **Inconsistent Structure**: Different approaches across files
- **Basic Error Handling**: Limited robustness
- **No Methodology Compliance**: Doesn't follow paper specifications exactly

### Improved Implementation
- **Complete Documentation**: Comprehensive commenting and docstrings
- **Methodology Compliance**: Follows paper specifications exactly
- **Consistent Structure**: Unified approach across all components
- **Robust Error Handling**: Comprehensive error checking
- **Professional Quality**: Production-ready code

## Migration Guide

### If You're Using Legacy Code

1. **For MCMC Analysis**: Replace `mcmc_inference.py` with `bayesian_mcmc.py`
2. **For LSTM Analysis**: Replace `lstm_model.py` with `lstm_model_improved.py`
3. **For Neural ODE**: Replace `neural_ode_model.py` with `neural_ode_model_improved.py`
4. **For Complete Pipeline**: Use `run_complete_analysis.py` instead of `run_full_analysis.py`

### Example Migration

**Legacy Code:**
```python
# Old way
from src.Bayesian.mcmc_inference import run_mcmc
results = run_mcmc(time_data, volume_data)
```

**Improved Code:**
```python
# New way
from src.Bayesian.mechanistic_model import TumorNanoparticleModel
from src.Bayesian.bayesian_mcmc import BayesianMCMCInference

model = TumorNanoparticleModel()
mcmc = BayesianMCMCInference(model, "MNPFDG")
results = mcmc.run_mcmc(time_data, volume_data)
```

## File Path Differences

### Legacy Code
- Uses `datasets/` directory
- References `Qi70`, `Qi100` datasets
- Inconsistent file naming

### Improved Code
- Uses `data/raw/` directory
- References `untreated85`, `saline85`, `mnps85`, `mnfdg85`
- Consistent naming convention

## Results Compatibility

### Legacy Results
- Basic CSV outputs
- Limited visualization
- No comprehensive reporting

### Improved Results
- Structured output directories
- Comprehensive visualizations
- Detailed analysis reports
- Performance metrics and comparisons

## Recommendations

### For New Users
- **Use the improved implementation** (`*_improved.py` files)
- **Follow README_METHODOLOGY.md** for detailed instructions
- **Run `scripts/run_complete_analysis.py`** for complete analysis

### For Existing Users
- **Migrate to improved implementation** gradually
- **Review README_METHODOLOGY.md** for new features
- **Update file paths** from `datasets/` to `data/raw/`

### For Development
- **Don't modify legacy files** unless necessary
- **Add new features** to improved implementation
- **Update documentation** for any changes

## Legacy Code Preservation

The legacy code is preserved for:
- **Historical Reference**: Understanding the development process
- **Backward Compatibility**: Supporting existing workflows
- **Educational Purposes**: Learning from different implementation approaches

## Support

- **For Improved Implementation**: Use README_METHODOLOGY.md
- **For Legacy Code Issues**: Check inline comments in legacy files
- **For Migration Help**: Refer to this document

---

**Note**: The improved implementation is the recommended approach for all new work and ensures complete methodology compliance with the paper. 