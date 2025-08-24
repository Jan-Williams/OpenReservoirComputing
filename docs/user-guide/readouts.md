# Readout Layers

## Overview

The readout layer performs the final mapping from reservoir states to desired outputs. It's the trainable component of reservoir computing architectures, often trained using ridge regression on collected reservoir states and target outputs.

All readout layers inherit from the `ReadoutBase` abstract class, ensuring a consistent API across different implementations.

## Base Class: ReadoutBase

The `ReadoutBase` class defines the core interface that all readout implementations must follow:

### Key Attributes
- `out_dim`: Output dimension
- `res_dim`: Reservoir dimension
- `dtype`: JAX array dtype (jnp.float32 or jnp.float64)

### Core Methods
- `readout(res_state)`: Map a single reservoir state to output
- `batch_readout(res_state)`: Map multiple reservoir states to outputs efficiently  
- `__call__(res_state)`: Flexible interface supporting both single and batch inputs

## LinearReadout Example

The `LinearReadout` class demonstrates a standard linear transformation from reservoir states to outputs with support for parallel reservoirs.

### Basic Usage

```python
import jax.numpy as jnp
from orc.readouts import LinearReadout

# Create a linear readout layer
readout = LinearReadout(
    out_dim=3,      # Output dimension
    res_dim=100,    # Reservoir dimension  
    chunks=1,       # Number of parallel reservoirs
    dtype=jnp.float64
)

# Apply readout to reservoir state
res_state = jnp.ones((1, 100))  # Shape: (chunks, res_dim)
output = readout(res_state)     # Shape: (out_dim,)
```

### Parallel Reservoirs

LinearReadout supports multiple parallel reservoirs, automatically handling the appropriate output concatenation:

```python
# Parallel reservoirs readout
readout = LinearReadout(
    out_dim=12,     # Total output dimension
    res_dim=100,    # Reservoir dimension per chunk
    chunks=4,       # Number of parallel reservoirs
    dtype=jnp.float64
)

# Process parallel reservoir states
res_state = jnp.random.normal(jax.random.key(0), (4, 100))
output = readout(res_state)  # Shape: (12,) - concatenated outputs
```

### Training

The readout weights (`wout`) are initialized to zero and trained externally using ridge regression:

```python
# After training (weights learned via ridge regression)
# readout.wout contains learned parameters with shape (chunks, out_dim/chunks, res_dim)
```

## Custom Readouts

The primary purpose of the readout system is to enable you to create your own readout layers tailored to your specific needs. While RC architectures often are trained via regression, this also provies more flexibility for more complicated readout architectures and training protocols.

```python
from orc.readouts import ReadoutBase
import equinox as eqx
import jax.numpy as jnp

class CustomReadout(ReadoutBase):
    # Define your parameters
    wout: Array
    chunks: int
    
    def __init__(self, out_dim, res_dim, chunks=1, **kwargs):
        super().__init__(out_dim, res_dim)
        self.chunks = chunks
        # Initialize your parameters (e.g., weights, nonlinearities)
        self.wout = jnp.zeros((chunks, int(out_dim/chunks), res_dim))
    
    def readout(self, res_state):
        # Implement your custom readout logic
        # Apply transformations, nonlinearities, etc.
        transformed_state = your_transformation(res_state)
        return jnp.ravel(eqx.filter_vmap(jnp.matmul)(self.wout, transformed_state))
```

### Requirements for Custom Readouts
- Inherit from `ReadoutBase`
- Implement the abstract `readout()` method
- Choose your parallel reservoir support level:

#### Single Reservoir Only
For readouts that don't support parallel reservoirs:
```python
def readout(self, res_state):
    # res_state shape: (res_dim,)
    output = your_transformation(res_state)
    return output  # Shape: (out_dim,)
```

#### Parallel Reservoir Support  
For readouts that support parallel reservoirs:
```python
def readout(self, res_state):
    # res_state shape: (chunks, res_dim)
    output = your_parallel_transformation(res_state)
    return output  # Shape: (out_dim,) - concatenated

def __call__(self, res_state):
    # Override to handle both single and batch inputs
    if len(res_state.shape) == 2:
        return self.readout(res_state)
    elif len(res_state.shape) == 3:
        return self.batch_readout(res_state)
    else:
        raise ValueError("Unsupported input shape")
```

### Shape Summary
- **Single reservoir**: `readout()` expects `(res_dim,)`, returns `(out_dim,)`
- **Parallel reservoirs**: `readout()` expects `(chunks, res_dim)`, returns `(out_dim,)`
- **Batch processing**: `batch_readout()` handles `(seq_len, chunks, res_dim)` → `(seq_len, out_dim)`

## Design Considerations

### Training Integration
- Readout weights are learned separately via ridge regression (see training utilities)
- The readout layer structure determines the feature space for regression
- More complex readouts can capture richer output relationships but may require more training data or different training procedures.


### Integration with RC Models
- Readout layers receive reservoir states from drivers
- The `chunks` dimension must match driver output structure
- Shape consistency: `driver_output.shape` must match `readout.readout()` input expectations