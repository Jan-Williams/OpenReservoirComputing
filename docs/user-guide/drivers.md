# Drivers

Drivers define the dynamics of the reservoir and how states evolve over time.

## ESNDriver

The ESNDriver supports both discrete and continuous modes:

### Discrete Mode
```python
from orc.drivers import ESNDriver

driver = ESNDriver(
    res_dim=500,
    spectral_radius=0.9,
    density=0.1,
    mode="discrete"
)
```

### Continuous Mode
```python
from orc.drivers import ESNDriver

driver = ESNDriver(
    res_dim=500,
    spectral_radius=0.9,
    density=0.1,
    mode="continuous",
    time_const=1.0  # Time constant for continuous dynamics
)
```

## Key Parameters

- **`res_dim`**: Number of reservoir neurons
- **`spectral_radius`**: Controls reservoir dynamics stability
- **`density`**: Sparsity of reservoir connections
- **`leak`**: Leakage rate parameter
- **`mode`**: "discrete" or "continuous" dynamics
- **`time_const`**: Time constant for continuous mode

For detailed API documentation, see the [Drivers API Reference](../api/drivers.md).
