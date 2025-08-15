# Readouts

Readouts map reservoir states to desired outputs and handle the training process.

## Available Readouts

### LinearReadout
Simple linear mapping with ridge regression:

```python
from orc.readouts import LinearReadout

readout = LinearReadout(
    res_dim=500,        # Reservoir dimension
    out_dim=3,          # Output dimension
    ridge_param=1e-6    # Ridge regression parameter
)
```

### NonlinearReadout
Nonlinear readout with customizable activation:

```python
from orc.readouts import NonlinearReadout

readout = NonlinearReadout(
    res_dim=500,
    out_dim=3,
    ridge_param=1e-6,
    activation="tanh"   # Activation function
)
```

### QuadraticReadout
Quadratic readout for nonlinear mappings:

```python
from orc.readouts import QuadraticReadout

readout = QuadraticReadout(
    res_dim=500,
    out_dim=3,
    ridge_param=1e-6
)
```

## Training Process

Readouts are trained using collected reservoir states:

1. Drive reservoir with training data
2. Collect reservoir states
3. Solve linear/nonlinear regression problem
4. Store trained weights

## Key Parameters

- **`res_dim`**: Reservoir dimension (input to readout)
- **`out_dim`**: Output dimension
- **`ridge_param`**: Ridge regression regularization parameter
- **`activation`**: Activation function for nonlinear readouts

For detailed API documentation, see the [Readouts API Reference](../api/readouts.md).
