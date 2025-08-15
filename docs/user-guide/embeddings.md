# Embeddings

Embeddings preprocess input data before feeding it into the reservoir.

## Available Embeddings

### LinearEmbedding
Applies linear transformation to map inputs to reservoir dimension:

```python
from orc.embeddings import LinearEmbedding

embedding = LinearEmbedding(
    in_dim=3,      # Input dimension
    res_dim=500,   # Reservoir dimension
    scaling=1.0,   # Input scaling factor
    chunks=1       # Number of parallel reservoirs
)
```

## Key Parameters

- **`in_dim`**: Dimension of input data
- **`res_dim`**: Dimension of reservoir (output of embedding)
- **`scaling`**: Scaling factor for input weights
- **`chunks`**: Number of parallel reservoir chunks
- **`locality`**: For parallel processing, overlap between chunks
- **`periodic`**: Whether to use periodic boundary conditions

## Custom Embeddings

You can create custom embeddings by inheriting from `EmbedBase`:

```python
from orc.embeddings import EmbedBase
import jax.numpy as jnp

class CustomEmbedding(EmbedBase):
    def embed(self, in_state):
        # Your custom embedding logic here
        return jnp.tanh(in_state)  # Example: simple nonlinear embedding
```

For detailed API documentation, see the [Embeddings API Reference](../api/embeddings.md).
