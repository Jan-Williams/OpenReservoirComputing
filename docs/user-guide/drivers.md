# Driver Layers

## Overview

The driver layer implements the reservoir's temporal dynamics. It takes the embedded input from the embedding layer and the current reservoir state, then computes the next reservoir state according to the specific reservoir dynamics (e.g., Echo State Network equations).

All driver layers inherit from the `DriverBase` abstract class, ensuring a consistent API across different implementations.

## Base Class: DriverBase

The `DriverBase` class defines the core interface that all driver implementations must follow:

### Key Attributes
- `res_dim`: Reservoir dimension
- `dtype`: JAX array dtype (jnp.float32 or jnp.float64)

### Core Methods
- `advance(proj_vars, res_state)`: Advance single reservoir state by one time step
- `batch_advance(proj_vars, res_state)`: Advance batch of reservoir states by one time step
- `__call__(proj_vars, res_state)`: Flexible interface supporting both single and batch updates

## ParallelESNDriver

The `ParallelESNDriver` class implements standard Echo State Network dynamics with tanh nonlinearity and support for both discrete and continuous time modes.

### Basic Usage

```python
import jax.numpy as jnp
from orc.drivers import ParallelESNDriver

# Create a basic ESN driver
driver = ParallelESNDriver(
    res_dim=100,           # Reservoir dimension
    leak=0.6,             # Leak rate (0 < leak <= 1)
    spectral_radius=0.8,   # Spectral radius of reservoir matrix
    density=0.02,         # Connection density (0 < density <= 1) 
    bias=1.6,             # Bias term in tanh nonlinearity
    seed=42               # Random seed for reproducibility
)

# Single time step advance
proj_input = jnp.ones((1, 100))      # Shape: (chunks, res_dim)
res_state = jnp.zeros((1, 100))      # Shape: (chunks, res_dim)
next_state = driver(proj_input, res_state)
```

### Discrete vs Continuous Time Modes

ParallelESNDriver supports both discrete and continuous time dynamics:

#### Discrete Mode (Default)
```python
driver = ParallelESNDriver(
    res_dim=100,
    mode="discrete",  # Default mode
    leak=0.6,
    seed=42
)
# Updates: x[n+1] = leak * tanh(A x[n] + proj_vars[n] + bias) + (1-leak) * x[n]
```

#### Continuous Mode
```python
driver = ParallelESNDriver(
    res_dim=100,
    mode="continuous",
    time_const=50.0,      # Time constant τ
    seed=42
)
# Updates: dx/dt = τ * (-x + tanh(W * x + proj_vars + bias))
```

### Parallel Reservoirs

ParallelESNDriver supports multiple parallel reservoirs for processing spatial or high-dimensional data:

```python
# Multiple parallel reservoirs
driver = ParallelESNDriver(
    res_dim=500,           # Dimension per reservoir
    chunks=10,            # Number of parallel reservoirs
    spectral_radius=0.9,
    density=0.05,
    seed=42
)

# Input and state shapes: (chunks, res_dim)
proj_input = jnp.random.normal(jax.random.key(0), (10, 500))
res_state = jnp.zeros((10, 500))
next_state = driver(proj_input, res_state)  # Shape: (10, 500)
```

### Key Parameters

#### Reservoir Matrix Parameters
- **spectral_radius**: Controls stability and memory (typically < 1)
- **density**: Sparsity of connections
- **chunks**: Number of parallel reservoirs

#### Nonlinearity Parameters  
- **bias**: Shifts tanh activation (affects reservoir dynamics)
- **leak**: Memory retention in discrete mode (0 < leak ≤ 1)
- **time_const**: Time scale in continuous mode (τ > 0)

### Memory and Performance Options

#### Sparse Eigenvalue Computation
```python
driver = ParallelESNDriver(
    res_dim=1000,
    use_sparse_eigs=True,      # Use sparse eigensolvers (default)
    eigenval_batch_size=100,   # Batch eigenvalue computation
    seed=42
)
```

#### Memory Management
- **use_sparse_eigs=True**: Uses sparse eigensolvers for large reservoirs (default)  
- **eigenval_batch_size**: Processes eigenvalues in batches to reduce memory usage
- Automatic fallback to dense eigensolvers for small reservoirs (res_dim < 100)

## Custom Drivers

To create your own driver layer:

```python
from orc.drivers import DriverBase
import equinox as eqx

class CustomDriver(DriverBase):
    def __init__(self, res_dim, **kwargs):
        super().__init__(res_dim)
        # Initialize your parameters
    
    def advance(self, proj_vars, res_state):
        # Implement your reservoir dynamics
        # Return shape depends on parallel reservoir support
        pass
```

### Requirements
- Inherit from `DriverBase`
- Implement the abstract `advance()` method
- Choose your parallel reservoir support level:

#### Single Reservoir Only
For drivers that don't support parallel reservoirs:
```python
def advance(self, proj_vars, res_state):
    # proj_vars shape: (res_dim,)
    # res_state shape: (res_dim,)
    next_state = your_dynamics(proj_vars, res_state)
    return next_state  # Shape: (res_dim,)
```

#### Parallel Reservoir Support  
For drivers that support parallel reservoirs like `ParallelESNDriver`:
```python
def advance(self, proj_vars, res_state):
    # proj_vars shape: (chunks, res_dim)
    # res_state shape: (chunks, res_dim) 
    next_state = your_parallel_dynamics(proj_vars, res_state)
    return next_state  # Shape: (chunks, res_dim)

def __call__(self, proj_vars, res_state):
    # Override to handle both single and batch inputs
    # Handle shapes: (chunks, res_dim) and (seq_len, chunks, res_dim)
    pass
```

### Shape Summary
- **Single reservoir**: `advance()` expects/returns `(res_dim,)`
- **Parallel reservoirs**: `advance()` expects/returns `(chunks, res_dim)`
