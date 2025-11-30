# DeepPLIV: Deep Partially Linear Instrumental Variable

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A neural network-based two-stage package for causal inference in partially-linear instrumental variable settings.

## Overview

DeepPLIV is a Python package that implements a deep learning approach to causal inference in partially linear models with instrumental variables. The package relaxes the linearity constraint between instrumental variables and the exposure of interest, while still providing consistent estimates of causal effects.

**Key Features:**
- Two-stage neural network architecture for flexible modeling
- Handles high-dimensional settings
- Supports non-linear relationships between instruments and endogenous variables
- Maintains linear treatment effects for interpretability
- Built on PyTorch for GPU acceleration

This package implements the methodology from the paper:
**"Causal Inference in Partially Linear Models with Instrumental Variables"** by Tomer Weiss & Malka Gorfine, 2024.

## Installation

### From PyPI (recommended)

```bash
pip install deeppliv
```

### For Development

```bash
git clone https://github.com/tomerweiss/deeppliv.git
cd deeppliv
pip install -e .
```

### With Optional Dependencies

For running the examples:
```bash
pip install deeppliv[examples]
```

For AWS functionality:
```bash
pip install deeppliv[aws]
```

For development with testing:
```bash
pip install deeppliv[dev]
```

## Quick Start

```python
from deeppliv import DeepPLIV
import numpy as np

# Prepare your data
# v_train: endogenous variable (training)
# z_train: instrumental variables (training)
# z_predict: instrumental variables (prediction)
# x: exogenous variables
# y: outcome variable

# Initialize and fit the model
model = DeepPLIV()
first_stage, second_stage = model.fit(
    v_1=v_train,
    z_1=z_train,
    z_2=z_predict,
    x=x,
    y=y,
    first_stage_epochs=500,
    second_stage_epochs=500
)

# Get the causal effect estimate
causal_effect = model.get_v_predicted_coefficient()
print(f"Estimated causal effect: {causal_effect}")

# Make predictions
predictions = model.predict(z=z_new, x=x_new)
```

## Model Architecture

DeepPLIV uses a two-stage approach:

### First Stage
- **Input**: Instrumental variables + exogenous variables
- **Architecture**: Deep neural network (64→64→16→output) with dropout regularization
- **Purpose**: Predict the endogenous variable from instruments

### Second Stage
- **Input**: Predicted endogenous variable + exogenous variables
- **Architecture**: Hybrid model with:
  - Deep network (32→16→16) for modeling confounding through exogenous variables
  - Linear layer for the treatment effect
- **Purpose**: Estimate the causal effect while controlling for confounding

## Examples

The package includes three main example scripts demonstrating different use cases:

### 1. Simple Usage (`example_usage_simple.py`)
Demonstrates basic usage with simulated data across 5 different scenarios:
```bash
python examples/example_usage_simple.py
```

### 2. Deep IV Setting (`example_deep_iv_setting.py`)
Compares DeepPLIV with Deep IV methods:
```bash
python examples/example_deep_iv_setting.py
```

### 3. Genetic IV Application (`example_genetic_iv.py`)
Shows application to genetic instrumental variables:
```bash
python examples/example_genetic_iv.py
```

## API Reference

### DeepPLIV Class

#### `fit(v_1, z_1, z_2, x, y, first_stage_epochs=500, second_stage_epochs=500, ...)`
Fit the two-stage model.

**Parameters:**
- `v_1` (np.array): Endogenous variable (training data)
- `z_1` (np.array): Instrumental variables for first stage
- `z_2` (np.array): Instrumental variables for prediction
- `x` (np.array): Exogenous variables
- `y` (np.array): Outcome variable
- `first_stage_epochs` (int): Number of training epochs for first stage (default: 500)
- `second_stage_epochs` (int): Number of training epochs for second stage (default: 500)
- `first_stage_learning_rate` (float): Learning rate for first stage (default: 0.01)
- `second_stage_learning_rate` (float): Learning rate for second stage (default: 0.01)
- `dropout` (float): Dropout rate for regularization (default: 0.6)

**Returns:**
- Tuple of (first_stage_model, second_stage_model)

#### `predict(z, x)`
Predict outcomes using the fitted model.

**Parameters:**
- `z` (np.array): Instrumental variables
- `x` (np.array): Exogenous variables

**Returns:**
- np.array: Predicted outcomes

#### `get_v_predicted_coefficient()`
Get the estimated causal effect coefficient.

**Returns:**
- float: The causal effect estimate

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.1
- NumPy >= 2.0.2
- pandas >= 2.2.3
- scikit-learn >= 1.5.2
- matplotlib >= 3.9.2
- seaborn >= 0.13.2
- scipy >= 1.13.1

## Citation

If you use this package in your research, please cite:

```bibtex
@article{weiss2024deeppliv,
  title={Causal Inference in Partially Linear Models with Instrumental Variables},
  author={Weiss, Tomer and Gorfine, Malka},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue on GitHub.

## Acknowledgments

This package was developed as part of a Master's thesis on causal inference in partially linear settings.
