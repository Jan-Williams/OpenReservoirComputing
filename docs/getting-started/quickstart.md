# Quick Start Guide

This guide will get you up and running with your first reservoir computer using ORC in just a few minutes.

## Your First Reservoir Computer

Let's create a simple Echo State Network (ESN) to forecast the famous Lorenz chaotic system:

```python
import orc
import matplotlib.pyplot as plt

# Generate Lorenz system data
U, t = orc.data.lorenz63(tN=100, dt=0.01)
print(f"Generated data shape: {U.shape}")  # (10000, 3)

# Split into training and testing sets
test_perc = 0.2
split_idx = int((1 - test_perc) * U.shape[0])
U_train = U[:split_idx, :]
U_test = U[split_idx:, :]

# Create ESN forecaster
forecaster = orc.ESNForecaster(
    res_dim=500,           # Reservoir size
    spectral_radius=0.9,   # Spectral radius of reservoir
    density=0.1,           # Sparsity of reservoir connections
    sigma=0.1,             # Input scaling
    ridge_param=1e-6       # Ridge regression parameter
)

# Train the forecaster
forecaster = orc.train_ESNForecaster(forecaster, U_train)

# Make predictions
forecast_steps = len(U_test)
U_pred = forecaster.forecast(U_test[0], steps=forecast_steps)

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
variables = ['X', 'Y', 'Z']

for i, (ax, var) in enumerate(zip(axes, variables)):
    ax.plot(t[split_idx:], U_test[:, i], 'b-', label=f'True {var}', alpha=0.7)
    ax.plot(t[split_idx:], U_pred[:, i], 'r--', label=f'Predicted {var}', alpha=0.7)
    ax.set_ylabel(var)
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()
```

## Understanding the Code

### 1. Data Generation
```python
U, t = orc.data.lorenz63(tN=100, dt=0.01)
```
This generates 100 time units of the Lorenz-63 system with a time step of 0.01, resulting in 10,000 data points with 3 variables (X, Y, Z).

### 2. ESN Configuration
```python
forecaster = orc.ESNForecaster(
    res_dim=500,           # Number of reservoir neurons
    spectral_radius=0.9,   # Controls reservoir dynamics stability
    density=0.1,           # Fraction of non-zero connections
    sigma=0.1,             # Input scaling factor
    ridge_param=1e-6       # Regularization for readout training
)
```

Key parameters:
- **`res_dim`**: Size of the reservoir (more neurons = more capacity but slower)
- **`spectral_radius`**: Should be ≤ 1.0 for stability; closer to 1.0 gives longer memory
- **`density`**: Lower values create sparser, more efficient reservoirs
- **`sigma`**: Controls input strength; tune based on your data scale

### 3. Training
```python
forecaster = orc.train_ESNForecaster(forecaster, U_train)
```
This trains only the output weights using ridge regression - the reservoir weights remain fixed.

### 4. Forecasting
```python
U_pred = forecaster.forecast(U_test[0], steps=forecast_steps)
```
Generate autonomous predictions starting from an initial condition.

## Next Steps

### Try Different Systems
```python
# Kuramoto-Sivashinsky equation
U, t = orc.data.ks(tN=50, dt=0.25, L=22)

# Custom data
# U should be shape (time_steps, features)
```

### Experiment with Parameters
```python
# Larger reservoir for complex dynamics
forecaster = orc.ESNForecaster(res_dim=1000, spectral_radius=0.95)

# Different input scaling
forecaster = orc.ESNForecaster(res_dim=500, sigma=0.5)
```

### Use Continuous-Time ESN
```python
# For continuous-time dynamics
forecaster = orc.CESNForecaster(
    res_dim=500,
    gamma=1.0,      # Leakage rate
    tau=1.0         # Time constant
)
forecaster = orc.train_CESNForecaster(forecaster, U_train, t_train)
```

### Hyperparameter Tuning
```python
import orc.ray_tune_integration as rt

# Define search space
search_space = {
    "res_dim": [300, 500, 800],
    "spectral_radius": [0.8, 0.9, 0.95, 0.99],
    "sigma": [0.1, 0.3, 0.5],
    "ridge_param": [1e-8, 1e-6, 1e-4]
}

# Run hyperparameter optimization
best_config = rt.run_hyperparameter_tuning(
    model_class=orc.ESNForecaster,
    train_data=U_train,
    search_space=search_space,
    num_samples=20
)
```

## Common Issues and Solutions

### Poor Forecasting Performance
- **Increase reservoir size**: Try `res_dim=1000` or higher
- **Adjust spectral radius**: Try values closer to 1.0 (e.g., 0.95, 0.99)
- **Scale your data**: Normalize inputs to have reasonable magnitude
- **More training data**: Ensure sufficient training samples

### Unstable Dynamics
- **Reduce spectral radius**: Try values < 0.9
- **Reduce input scaling**: Lower `sigma` parameter
- **Check data preprocessing**: Ensure data is properly scaled

### Slow Performance
- **Reduce reservoir size**: Start with `res_dim=300`
- **Increase sparsity**: Lower `density` parameter
- **Use GPU**: Install JAX with CUDA support

Ready to dive deeper? Check out the [User Guide](../user-guide/overview.md) for comprehensive documentation of all ORC features.
