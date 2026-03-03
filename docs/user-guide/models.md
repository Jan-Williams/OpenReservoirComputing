# Models

## Overview

The models module provides complete, ready-to-use reservoir computing architectures by combining the modular embedding, driver, and readout components. These models implement common RC paradigms found in the literature, allowing users to quickly apply proven architectures without building from scratch.

All models are built on base classes defined in the corresponding submodules: `RCForecasterBase` and `CRCForecasterBase` in `src/orc/forecaster/base.py`, `RCClassifierBase` in `src/orc/classifier/base.py`, and `RCControllerBase` in `src/orc/control/base.py`.

## Echo State Networks (ESN)

OpenReservoirComputing provides both discrete and continuous time implementations of Echo State Networks, the most widely used reservoir computing architecture.

### ESNForecaster (Discrete-time)

The `ESNForecaster` implements standard discrete-time Echo State Networks with tanh nonlinearity.

#### Basic Usage

```python
from orc.forecaster import ESNForecaster, train_ESNForecaster
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
from orc.forecaster import CESNForecaster, train_CESNForecaster
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

## ESN Classifier

The `ESNClassifier` implements an Echo State Network for sequence classification tasks. Rather than forecasting, the readout is trained to predict class labels from the reservoir's representation of an input sequence.

### Basic Usage

```python
from orc.classifier import ESNClassifier, train_ESNClassifier
import jax.numpy as jnp

# Create ESN classifier
model = ESNClassifier(
    data_dim=3,       # Input dimension
    n_classes=4,      # Number of classes
    res_dim=500,      # Reservoir dimension
    seed=42
)

# Train with a list of sequences and corresponding labels
# train_seqs: list of arrays, each shape (seq_len, data_dim)
# labels: array of integer class indices, shape (n_sequences,)
trained_model = train_ESNClassifier(model, train_seqs, labels)

# Classify a new sequence
probs = trained_model.classify(
    in_seq=test_seq,                    # Shape: (seq_len, data_dim)
    res_state=jnp.zeros((model.res_dim,))
)
predicted_class = jnp.argmax(probs)
```

### Key Parameters

- `data_dim`: Dimension of the input sequences
- `n_classes`: Number of target classes
- `res_dim`: Reservoir dimension
- `leak_rate`: Memory retention (default `0.6`)
- `Wr_spectral_radius`: Spectral radius of reservoir matrix (default `0.8`)
- `state_repr`: How to summarize reservoir states for classification — `"final"` uses only the last state, `"mean"` averages all states after spinup (default `"final"`)
- `quadratic`: Use quadratic readout for a richer feature space (default `False`)

### State Representation

The `state_repr` parameter controls which reservoir states are passed to the readout:

```python
# Use mean reservoir state (often better for longer sequences)
model = ESNClassifier(
    data_dim=3,
    n_classes=4,
    res_dim=500,
    state_repr="mean",
    seed=42
)
```

For detailed API documentation, see the [Classifier API Reference](../api/classifier.md).

## ESN Controller

The `ESNController` implements an Echo State Network for control tasks, where the model tracks a reference trajectory by optimizing a control input sequence.

### Basic Usage

```python
from orc.control import ESNController, train_ESNController
import jax.numpy as jnp

# Create ESN controller
model = ESNController(
    data_dim=3,           # System input/output dimension
    control_dim=1,        # Control input dimension
    res_dim=500,          # Reservoir dimension
    seed=42
)

# Train with input and control sequences
trained_model, res_states = train_ESNController(
    model,
    train_seq,            # shape: (seq_len, data_dim)
    control_seq,          # shape: (seq_len, control_dim)
    target_seq=target_seq,
    spinup=100,
    beta=8e-8
)

# Compute optimal control to track a reference trajectory
control_opt = trained_model.compute_control(
    control_seq=initial_guess,   # shape: (fcast_len, control_dim)
    res_state=res_states[-1],
    ref_traj=reference           # shape: (fcast_len, data_dim)
)
```

### Control Penalty Weights

The controller optimizes a penalty function with three terms, each weighted by a tunable parameter:

- `alpha_1` (default 100): **Trajectory deviation** — penalizes squared error between the controlled forecast and the reference trajectory. Higher values enforce tighter tracking.

- `alpha_2` (default 1): **Control magnitude** — penalizes the squared norm of control inputs. Higher values favor smaller control effort.

- `alpha_3` (default 5): **Control smoothness** — penalizes the squared norm of consecutive control input differences. Higher values produce smoother control signals.

```python
# Custom penalty weights
model = ESNController(
    data_dim=3,
    control_dim=1,
    res_dim=500,
    alpha_1=200,    # Stricter trajectory tracking
    alpha_2=0.5,    # Allow larger control inputs
    alpha_3=10,     # Enforce smoother control
    seed=42
)
```

### Key Parameters

In addition to the standard ESN parameters (`leak_rate`, `bias`, `embedding_scaling`, `Wr_density`, `Wr_spectral_radius`), the controller accepts:

- `control_dim`: Dimension of the control input
- `alpha_1`, `alpha_2`, `alpha_3`: Control penalty weights (see above)

For detailed API documentation, see the [Control API Reference](../api/control.md).

## Training Functions

All models use ridge regression for training the readout layer. ORC provides unified training functions that work with any model inheriting from the base classes, as well as convenience wrappers for each specific model type.

### Unified Training Functions (Recommended)

The primary training API uses generic functions that work with any `RCForecasterBase`, `RCClassifierBase`, or `RCControllerBase` subclass, including custom models:

```python
from orc.forecaster import train_RCForecaster
from orc.classifier import train_RCClassifier
from orc.control import train_RCController

# Forecaster: works with ESNForecaster, CESNForecaster, EnsembleESNForecaster,
# and any custom RCForecasterBase subclass
trained_model, res_states = train_RCForecaster(model, train_seq, target_seq=target_seq)

# Classifier: works with ESNClassifier and custom RCClassifierBase subclasses
trained_model = train_RCClassifier(model, train_seqs, labels)

# Controller: works with ESNController and custom RCControllerBase subclasses
trained_model, res_states = train_RCController(model, train_seq, control_seq)
```

For continuous-time models, pass the time vector via `**force_kwargs`:

```python
# CESNForecaster via the unified function
trained_model, res_states = train_RCForecaster(
    model, train_seq, target_seq=target_seq, ts=t_train
)
```

### Model-Specific Convenience Functions

The legacy model-specific functions remain available and delegate to the unified functions above:

- `train_ESNForecaster` — equivalent to `train_RCForecaster` with an `ESNForecaster` type check
- `train_CESNForecaster` — equivalent to `train_RCForecaster(..., ts=t_train)` with a `CESNForecaster` type check
- `train_EnsembleESNForecaster` — equivalent to `train_RCForecaster` with an `EnsembleESNForecaster` type check
- `train_ESNClassifier` — equivalent to `train_RCClassifier` with an `ESNClassifier` type check
- `train_ESNController` — equivalent to `train_RCController` with an `ESNController` type check

### Training Parameters

**Core Parameters:**
- `spinup`: Number of initial transient steps to discard

- `beta`: Tikhonov regularization parameter (typically 1e-8 to 1e-6)

- `initial_res_state`: Custom initial reservoir state (default: uses `model.driver.default_state()`)

**Memory Management:**
- `batch_size`: Process parallel reservoirs in batches to reduce memory usage (available for all three unified functions)

**Extra Force Arguments:**
- `**force_kwargs`: Additional keyword arguments forwarded to `model.force()` (e.g., `ts=t_train` for continuous models)

### Advanced Training

```python
# Batched training for memory efficiency with many parallel reservoirs
trained_model, res_states = train_RCForecaster(
    model,
    train_seq,
    target_seq=target_seq,
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
from orc.forecaster import RCForecasterBase, train_RCForecaster
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

# Train using the unified function — no model-specific training code needed
trained_model, res_states = train_RCForecaster(
    CustomForecaster(data_dim=3, res_dim=500),
    train_seq,
    target_seq=target_seq,
    spinup=100,
    beta=8e-8,
)
```

The unified `train_RCForecaster` function delegates shape handling to the readout's `prepare_train`, `prepare_target`, and `set_wout` methods and the driver's `default_state` method, so custom components that implement those methods integrate automatically with training. See the [Readouts](readouts.md) and [Drivers](drivers.md) user guides for details.

## Performance Considerations

- Use `jnp.float64` for numerical precision (default)
- Enable `use_sparse_eigs=True` for large reservoirs (default)
- Use `batch_size` parameter for memory-constrained training with parallel reservoirs
- Consider `chunks > 1` for high-dimensional systems to reduce the required reservoir size

For detailed API documentation, see the [Forecaster API Reference](../api/forecaster.md).
