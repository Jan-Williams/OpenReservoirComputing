# Models

## Overview

The models module provides complete, ready-to-use reservoir computing architectures by combining the modular embedding, driver, and readout components. These models implement common RC paradigms found in the literature, allowing users to quickly apply proven architectures without building from scratch.

All models are built on the base classes `RCForecasterBase` (discrete-time) and `CRCForecasterBase` (continuous-time) defined in `src/orc/rc.py`.

## Echo State Networks (ESN)

OpenReservoirComputing provides both discrete and continuous time implementations of Echo State Networks, the most widely used reservoir computing architecture.

### ESNForecaster (Discrete-time)

The `ESNForecaster` implements standard discrete-time Echo State Networks with tanh nonlinearity.

#### Basic Usage

```python
from orc.models.esn import ESNForecaster, train_ESNForecaster
import jax.numpy as jnp

# Create ESN model
model = ESNForecaster(
    data_dim=3,           # Input/output dimension
    res_dim=500,          # Reservoir dimension
    leak_rate=0.6,        # Integration leak rate
    spectral_radius=0.8,  # Spectral radius of reservoir matrix
    seed=42
)

# Generate training data (e.g., Lorenz system)
# train_seq shape: (seq_len, data_dim)
# target_seq shape: (seq_len, data_dim)

# Train the model
trained_model, res_states = train_ESNForecaster(
    model, 
    train_seq, 
    target_seq=target_seq,
    spinup=100,           # Discard initial transient
    beta=8e-8            # Ridge regression regularization
)

# Forecast from final training state
forecast = trained_model.forecast(fcast_len=1000, res_state=res_states[-1])

# Or forecast from new initial condition
forecast = trained_model.forecast_from_IC(fcast_len=1000, spinup_data=new_ic)
```

#### Key Parameters

**Reservoir Architecture:**

- `data_dim`: Input/output dimension

- `res_dim`: Reservoir dimension (typically 100-1000)

- `leak_rate`: Memory retention (0 < leak_rate ≤ 1)

- `bias`: Shifts tanh activation (affects dynamics)

**Reservoir Matrix:**
- `Wr_spectral_radius`: Largest eigenvalue (typically < 1 for stability)

- `Wr_density`: Sparsity of connections (0 < density ≤ 1)

**Input Processing:**
- `embedding_scaling`: Scales input weight matrix (typically 0.01-0.1)

**Spatial Processing:**
- `chunks`: Number of parallel reservoirs for spatial data

- `locality`: Overlap between adjacent chunks

- `periodic`: Periodic boundary conditions for spatial decomposition

**Performance Options:**
- `use_sparse_eigs`: Use sparse eigensolvers for large reservoirs (default True)

#### Spatial/Parallel Processing

For spatiotemporal systems or high-dimensional data:

```python
# Parallel reservoirs for spatial data
model = ESNForecaster(
    data_dim=1000,        # Spatial dimension
    res_dim=200,          # Reservoir size per chunk
    chunks=50,            # Number of parallel reservoirs  
    locality=2,           # Overlap between chunks
    periodic=True,        # Periodic boundary conditions
    seed=42
)
```

#### Readout Options

```python
# Quadratic readout for richer feature space
model = ESNForecaster(
    data_dim=3,
    res_dim=500,
    quadratic=True,       # Use quadratic instead of linear readout
    seed=42
)
```

### CESNForecaster (Continuous-time)

The `CESNForecaster` implements continuous-time Echo State Networks using ODE solvers from the `diffrax` library.

#### Basic Usage

```python
from orc.models.esn import CESNForecaster, train_CESNForecaster
import jax.numpy as jnp

# Create continuous ESN model  
model = CESNForecaster(
    data_dim=3,
    res_dim=500,
    time_const=50.0,      # Time constant τ in continuous dynamics
    bias=1.6,
    seed=42
)

# Training requires time array
t_train = jnp.linspace(0, 10, train_seq.shape[0])

# Train the model
trained_model, res_states = train_CESNForecaster(
    model,
    train_seq,
    t_train,              # Time array for training
    target_seq=target_seq,
    spinup=100,
    beta=8e-8
)

# Forecasting with time array
t_forecast = jnp.linspace(0, 20, 1000)
forecast = trained_model.forecast(t_forecast, res_states[-1])
```

#### Key Differences from Discrete ESN

**Continuous Dynamics:**
- Uses `time_const` instead of `leak_rate` to control temporal scale

- Dynamics: `dx/dt = time_const * (-x + tanh(W*x + proj_vars + bias))`

- Requires time arrays for training and forecasting

**ODE Integration:**
- Default solver: `diffrax.Tsit5()` (adaptive Runge-Kutta)

- Default step size controller: `diffrax.PIDController()`

- Built-in cubic interpolation for input forcing

**Custom ODE Settings:**

```python
import diffrax

model = CESNForecaster(
    data_dim=3,
    res_dim=500,
    solver=diffrax.Dopri5(),                    # Different solver
    stepsize_controller=diffrax.PIDController(   # Custom tolerances
        rtol=1e-5, atol=1e-8
    ),
    seed=42
)
```

## Training Functions

Both ESN models use ridge regression for training the readout layer.

### Training Parameters

**Core Parameters:**
- `spinup`: Number of initial transient steps to discard

- `beta`: Tikhonov regularization parameter (typically 1e-8 to 1e-6)

- `initial_res_state`: Custom initial reservoir state (default: zeros)

**Memory Management:**
- `batch_size`: Process parallel reservoirs in batches to reduce memory usage

### Advanced Training

```python
# Batched training for memory efficiency with many parallel reservoirs
trained_model, res_states = train_ESNForecaster(
    model,
    train_seq,
    target_seq,
    spinup=100,
    beta=1e-7,
    batch_size=10    # Process 10 parallel reservoirs at a time
)
```

## Model Methods

All models inherit standard methods from the base classes `RCForecasterBase` and `CRCForecasterBase`:

### Core Methods

**Training Phase:**
- `force(in_seq, res_state)`: Teacher forcing with input sequence
- `set_readout(readout)`: Replace readout layer (e.g., after training)
- `set_embedding(embedding)`: Replace embedding layer

**Prediction Phase:**
- `forecast(fcast_len, res_state)`: Autonomous prediction from reservoir state
- `forecast_from_IC(fcast_len, spinup_data)`: Prediction from initial condition data

### Shape Conventions

**Discrete Models:**
- Input sequences: `(seq_len, data_dim)`
- Reservoir states: `(chunks, res_dim)` or `(res_dim,)` for single reservoir
- Reservoir sequences: `(seq_len, chunks, res_dim)`

**Continuous Models:**
- Time arrays: `(seq_len,)`
- All other shapes same as discrete models

## Design Philosophy

The models module demonstrates the power of ORC's modular design:

1. **Composition**: Each model combines specific embedding, driver, and readout layers

2. **Flexibility**: Users can modify individual components or create custom variants

3. **Performance**: Built-in optimizations like sparse eigensolvers and batched training

4. **Extensibility**: New model classes can be added following the same patterns

## Creating Custom Models

To create your own model, follow the ESN pattern:

```python
from orc.rc import RCForecasterBase
from orc.drivers import CustomDriver
from orc.embeddings import CustomEmbedding  
from orc.readouts import CustomReadout

class CustomForecaster(RCForecasterBase):
    def __init__(self, data_dim, res_dim, **kwargs):
        # Initialize components
        driver = CustomDriver(res_dim=res_dim, **kwargs)
        embedding = CustomEmbedding(in_dim=data_dim, res_dim=res_dim, **kwargs)
        readout = CustomReadout(out_dim=data_dim, res_dim=res_dim, **kwargs)
        
        # Initialize base class
        super().__init__(driver, readout, embedding, **kwargs)
```

## Performance Considerations

- Use `jnp.float64` for numerical precision (default)
- Enable `use_sparse_eigs=True` for large reservoirs (default)
- Use `batch_size` parameter for memory-constrained training with parallel reservoirs
- Consider `chunks > 1` for high-dimensional systems to reduce the required reservoir size

For detailed API documentation, see the [Models API Reference](../api/models.md).
