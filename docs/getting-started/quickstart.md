# Quick Start Guide

This guide will get you up and running with your first reservoir computer in just a few minutes. We'll train an Echo State Network (ESN) to forecast the famous, chaotic Lorenz system.

### Step 1: Import ORC

```python
import orc
import jax.numpy as jnp
```

### Step 2: Generate Data

Let's create some data using the built-in Lorenz system:

```python
# Generate Lorenz trajectory
U, t = orc.data.lorenz63(tN=100, dt=0.01)
print(f"Data shape: {U.shape}")  # (10000, 3)

# Split into training and testing
test_perc = 0.2
split_idx = int((1 - test_perc) * U.shape[0])
U_train = U[:split_idx, :]
U_test = U[split_idx:, :]
```

### Step 3: Create and Train Your ESN

```python
# Create ESN forecaster
esn = orc.models.ESNForecaster(
    data_dim=3,      # Lorenz system is 3D
    res_dim=500,     # 500 reservoir neurons
    seed=42          # For reproducibility
)

# Train the model
esn, res_states = orc.models.train_ESNForecaster(esn, U_train)
print("Training complete!")
```

### Step 4: Make Predictions

```python
# Forecast from the last training state
forecast_len = U_test.shape[0]
U_pred = esn.forecast(fcast_len=forecast_len, res_state=res_states[-1])

print(f"Forecast shape: {U_pred.shape}")
```

### Step 5: Visualize Results

```python
# Plot the results
orc.utils.visualization.plot_time_series(
    [U_test, U_pred],
    t[split_idx:] - t[split_idx],  # Reset time to start from 0
    state_var_names=["$x$", "$y$", "$z$"],
    time_series_labels=["True", "Predicted"],
    line_formats=["-", "r--"],
    title="Lorenz System Forecast",
    x_label="Time"
)
```

## Complete Example Script

Here's the complete working example you can copy and run:

```python
import orc
import jax.numpy as jnp

# Generate Lorenz system data
U, t = orc.data.lorenz63(tN=100, dt=0.01)

# Train-test split
test_perc = 0.2
split_idx = int((1 - test_perc) * U.shape[0])
U_train = U[:split_idx, :]
U_test = U[split_idx:, :]

# Create and train ESN
esn = orc.models.ESNForecaster(data_dim=3, res_dim=500, seed=42)
esn, res_states = orc.models.train_ESNForecaster(esn, U_train)

# Make predictions
U_pred = esn.forecast(fcast_len=U_test.shape[0], res_state=res_states[-1])

# Visualize results
orc.utils.visualization.plot_time_series(
    [U_test, U_pred],
    t[split_idx:] - t[split_idx],
    state_var_names=["$x$", "$y$", "$z$"],
    time_series_labels=["True", "Predicted"],
    line_formats=["-", "r--"],
    title="Lorenz System Forecast"
)
```

## Understanding the Results

The ESN should successfully forecast the chaotic Lorenz system for several Lyapunov times (a characteristic time scale of the system). You'll see the predicted trajectory (red dashed line) closely following the true trajectory (solid line) before gradually diverging due to the chaotic nature of the system.

## Next Steps

Now that you have a working reservoir computer:

1. **Experiment with Parameters**: Try different `res_dim`, `spectral_radius`, or `leak_rate`
2. **Try Other Systems**: Use `orc.data.mackey_glass()` or bring your own data
3. **Explore Continuous RC**: Check out `CESNForecaster` for continuous-time systems
4. **Learn Advanced Features**: Read the [User Guide](../user-guide/overview.md) for parallel reservoirs, custom drivers, and more

## Common Patterns

### Working with Your Own Data

```python
# Your data should be shaped (time_steps, features)
your_data = jnp.array(...)  # Shape: (1000, 5) for 1000 timesteps, 5 features

# Create appropriately sized ESN
esn = orc.models.ESNForecaster(
    data_dim=your_data.shape[1],  # Number of features
    res_dim=200                   # Adjust based on complexity
)

# Train and forecast as before
esn, states = orc.models.train_ESNForecaster(esn, your_data[:-100])
predictions = esn.forecast(fcast_len=100, res_state=states[-1])
```

### Forecasting on a Different Trajectory

```python
# Create training and test trajectories
U_train, t_train = orc.data.lorenz63(tN=100, dt=0.01, u0=(-10.0, 1.0, 10.0))
U_test, t_test = orc.data.lorenz63(tN=100, dt=0.01, u0=(10.0, 1.0, -10.0))

# Train as before
res_dim = 500
data_dim = 3
esn = orc.models.ESNForecaster(data_dim=data_dim, res_dim=res_dim, seed=42)
esn, res_states_train = orc.models.train_ESNForecaster(esn, U_train)

# Spinup the reservoir on new trajectory
# Note the initial reservoir state is a row vector
spinup = 100
fcast_len = 200
test_spin_states = esn.force(
                            in_seq=U_test[:spinup],
                            res_state=jnp.zeros((1, res_dim))
                            )
fcast = esn.forecast(fcast_len=fcast_len, res_state=test_spin_states[-1])

# Equivalently, use the forecast_from_IC function
fcast = esn.forecast_from_IC(fcast_len=fcast_len, spinup_data=U_test[:spinup])
```

Ready to dive deeper? Check out our [Examples](../examples/lorenz.ipynb) and [User Guide](../user-guide/overview.md)!