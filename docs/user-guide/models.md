# Models

ORC provides two main forecasting models: ESNForecaster for discrete-time systems and CESNForecaster for continuous-time systems.

## ESNForecaster

The `ESNForecaster` implements the standard Echo State Network for discrete-time forecasting.

### Basic Usage

```python
import orc

# Create ESN forecaster
forecaster = orc.ESNForecaster(
    res_dim=500,
    spectral_radius=0.9,
    density=0.1,
    sigma=0.1,
    ridge_param=1e-6
)

# Train on data
forecaster = orc.train_ESNForecaster(forecaster, training_data)

# Make predictions
predictions = forecaster.forecast(initial_condition, steps=100)
```

### Key Parameters

- **`res_dim`**: Number of reservoir neurons
- **`spectral_radius`**: Spectral radius of reservoir matrix
- **`density`**: Sparsity of reservoir connections
- **`sigma`**: Input scaling factor
- **`ridge_param`**: Ridge regression regularization

## CESNForecaster

The `CESNForecaster` implements Continuous Echo State Networks for continuous-time dynamics.

### Basic Usage

```python
import orc

# Create CESN forecaster
forecaster = orc.CESNForecaster(
    res_dim=500,
    spectral_radius=0.9,
    gamma=1.0,  # Leakage rate
    tau=1.0     # Time constant
)

# Train on data (requires time information)
forecaster = orc.train_CESNForecaster(forecaster, training_data, time_data)

# Make predictions
predictions = forecaster.forecast_from_IC(initial_condition, time_points)
```

### Key Parameters

- **`gamma`**: Leakage rate controlling memory decay
- **`tau`**: Time constant for continuous dynamics
- **`spectral_radius`**: Same as ESN
- **`res_dim`**: Same as ESN

## Training Functions

### train_ESNForecaster

Trains an ESN using the standard approach:

```python
trained_forecaster = orc.train_ESNForecaster(
    forecaster,
    data,
    washout=100,  # Transient steps to discard
    **kwargs
)
```

### train_CESNForecaster

Trains a CESN with time-aware training:

```python
trained_forecaster = orc.train_CESNForecaster(
    forecaster,
    data,
    time,
    washout=100,
    **kwargs
)
```

## Advanced Usage

### Custom Components

You can customize models by providing your own components:

```python
from orc.embeddings import PolynomialEmbedding
from orc.readouts import RidgeReadout

forecaster = orc.ESNForecaster(
    res_dim=500,
    embedding=PolynomialEmbedding(degree=2),
    readout=RidgeReadout(ridge_param=1e-4)
)
```

### Hyperparameter Tuning

Use the Ray Tune integration for automated hyperparameter optimization:

```python
import orc.ray_tune_integration as rt

search_space = {
    "res_dim": [300, 500, 800],
    "spectral_radius": [0.8, 0.9, 0.95],
    "sigma": [0.1, 0.3, 0.5]
}

best_config = rt.run_hyperparameter_tuning(
    model_class=orc.ESNForecaster,
    train_data=training_data,
    search_space=search_space,
    num_samples=50
)
```

For detailed API documentation, see the [Models API Reference](../api/models.md).
