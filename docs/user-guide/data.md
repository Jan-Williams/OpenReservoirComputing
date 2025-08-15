# Data Library

ORC includes a data library with common dynamical systems for testing and examples.

## Available Datasets

### Lorenz63 System
The classic chaotic Lorenz system:

```python
import orc

# Generate Lorenz data
U, t = orc.data.lorenz63(
    tN=100,     # Total time
    dt=0.01,    # Time step
    sigma=10.0, # Lorenz parameters
    rho=28.0,
    beta=8.0/3.0
)
```

### Kuramoto-Sivashinsky Equation
Spatiotemporal chaotic PDE:

```python
import orc

# Generate KS data
U, t = orc.data.KS_1D(
    tN=50,      # Total time
    dt=0.25,    # Time step
    L=22,       # Domain length
    N=64        # Spatial points
)
```

## Using Your Own Data

ORC works with any time series data. Ensure your data is:

- **Shape**: `(time_steps, features)`
- **Type**: NumPy array or JAX array
- **Scale**: Normalized to reasonable range

```python
import numpy as np

# Your data should look like this
data = np.random.randn(1000, 3)  # 1000 time steps, 3 features
```

For detailed API documentation, see the [Data API Reference](../api/data.md).
