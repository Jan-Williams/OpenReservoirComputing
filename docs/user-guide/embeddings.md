# Embedding Layers

## Overview

The embedding layer serves as the interface between your input data and the reservoir computer. It transforms input vectors of dimension `in_dim` into embedded vectors of dimension `res_dim` that can be processed by the reservoir driver.

All embedding layers inherit from the `EmbedBase` abstract class, ensuring a consistent API across different implementations.

## Base Class: EmbedBase

The `EmbedBase` class defines the core interface that all embedding implementations must follow:

### Key Attributes
- `in_dim`: Input dimension
- `res_dim`: Reservoir dimension  
- `dtype`: JAX array dtype (jnp.float32 or jnp.float64)

### Core Methods
- `embed(in_state)`: Transform a single input vector
- `batch_embed(in_state)`: Transform multiple input vectors efficiently
- `__call__(in_state)`: Flexible interface supporting both single and batch inputs

## LinearEmbedding

The `LinearEmbedding` class provides matrix multiplication-based embedding with support for parallel reservoirs to handle spatial/high-dimensional data.

### Basic Usage

```python
import jax.numpy as jnp
from orc.embeddings import LinearEmbedding

# Create a simple linear embedding
embedding = LinearEmbedding(
    in_dim=10,      # Input dimension
    res_dim=100,    # Reservoir dimension
    scaling=0.1,    # Input matrix values range [-0.1, 0.1]
    seed=42         # Random seed for reproducibility
)

# Embed a single input
input_vector = jnp.ones(10)
embedded = embedding(input_vector)  # Shape: (1, 100) for single reservoir
```

### Parallel Reservoirs

LinearEmbedding supports decomposing high-dimensional inputs into multiple (overlapping) parallel reservoirs:

```python
# Parallel reservoirs for spatial data
embedding = LinearEmbedding(
    in_dim=1000,     # Total spatial dimension
    res_dim=200,     # Reservoir dimension per chunk
    scaling=0.1,
    chunks=50,       # Create 4 parallel reservoirs
    locality=2,     # Each reservoir sees 2 neighbors on each side
                    # in addition to in_dim / chunks values
    periodic=True,  # Use periodic boundary conditions
    seed=42
)

# Input gets decomposed into overlapping chunks
spatial_input = jnp.sin(jnp.linspace(0, 4*jnp.pi, 1000))
embedded = embedding(spatial_input)  # Shape: (4, 200)
```

### Key Parameters

- **scaling**: Controls the magnitude of random input weights (range [-scaling, scaling])
- **chunks**: Number of parallel reservoirs to create
- **locality**: Number of neighboring points each reservoir can see
- **periodic**: Whether to use periodic boundary conditions for spatial decomposition

### Spatial Decomposition

When `chunks > 1`, the embedding automatically:

1. **Localizes**: Splits the input into overlapping chunks based on `locality`
2. **Handles boundaries**: Uses periodic or extended boundary conditions
3. **Embeds**: Applies linear transformation to each chunk independently

This is particularly useful for:
- Spatiotemporal systems (PDEs, cellular automata)

- Problems with translational symmetry

### Boundary Conditions

- **periodic=True**: Connects end and beginning smoothly (default)
- **periodic=False**: Extends edge values into locality regions

## Custom Embeddings

To create your own embedding layer:

```python
from orc.embeddings import EmbedBase
import equinox as eqx

class CustomEmbedding(EmbedBase):
    def __init__(self, in_dim, res_dim, **kwargs):
        super().__init__(in_dim, res_dim)
        # Initialize your parameters
    
    def embed(self, in_state):
        # Implement your embedding logic
        # Return shape depends on parallel reservoir support
        pass
```

### Requirements
- Inherit from `EmbedBase`
- Implement the abstract `embed()` method
- Choose your parallel reservoir support level:

#### Single Reservoir Only
For embeddings that don't support parallel reservoirs:
```python
def embed(self, in_state):
    # Transform input to reservoir dimension
    embedded = your_transformation(in_state)
    return embedded  # Shape: (res_dim,)
```

#### Parallel Reservoir Support  
For embeddings that support parallel reservoirs like `LinearEmbedding`:
```python
def embed(self, in_state):
    # Transform input with parallel processing
    embedded = your_parallel_transformation(in_state) 
    return embedded  # Shape: (chunks, res_dim)
```

### Shape Summary
- **Single reservoir**: `embed()` returns `(res_dim,)`
- **Parallel reservoirs**: `embed()` returns `(chunks, res_dim)`
- The base class `__call__()` method handles single reservoirs automatically
- For parallel reservoir support, override `__call__()` to handle shape processing

## Design Considerations

### Memory and Performance
- Use `chunks > 1` for large spatial systems to reduce memory usage
- The `batch_embed` method is JIT-compiled for efficient batch processing
- All operations use JAX for automatic differentiation and GPU acceleration

### Integration with RC Models
- Embedding output feeds directly into reservoir drivers
- The `chunks` dimension corresponds to parallel reservoir processing
- Shape consistency: `embedding_output.shape[0]` must match driver expectations
