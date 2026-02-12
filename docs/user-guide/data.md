# Data Library

## Overview

ORC includes a library of built-in dynamical systems for benchmarking and experimentation. All systems are integrated using Diffrax ODE/PDE solvers and return JAX arrays ready for use with ORC models.

Each generator returns a tuple `(U, t)` where `U` has shape `(time_steps, state_dim)` and `t` has shape `(time_steps,)`.

## Available Systems

### Low-Dimensional Chaotic Systems

**`orc.data.lorenz63`** — The classic 3D Lorenz attractor. A standard benchmark for chaotic time series forecasting.

```python
U, t = orc.data.lorenz63(tN=100, dt=0.01)
# U.shape: (10000, 3)
```

**`orc.data.rossler`** — The 3D Rössler attractor, exhibiting a simpler single-scroll chaotic behavior.

```python
U, t = orc.data.rossler(tN=100, dt=0.01)
```

**`orc.data.colpitts`** — The Colpitts oscillator, a chaotic electronic circuit model.

```python
U, t = orc.data.colpitts(tN=100, dt=0.01)
```

**`orc.data.sakaraya`** — The Sakaraya system, a 3D autonomous chaotic flow.

```python
U, t = orc.data.sakaraya(tN=100, dt=0.01)
```

**`orc.data.double_pendulum`** — The double pendulum, a classic example of mechanical chaos.

```python
U, t = orc.data.double_pendulum(tN=100, dt=0.01)
```

### Hyperchaotic Systems

**`orc.data.hyper_lorenz63`** — A 4D hyperchaotic extension of the Lorenz system with two positive Lyapunov exponents.

```python
U, t = orc.data.hyper_lorenz63(tN=100, dt=0.01)
```

**`orc.data.hyper_xu`** — The 4D Xu hyperchaotic system.

```python
U, t = orc.data.hyper_xu(tN=100, dt=0.01)
```

### High-Dimensional Systems

**`orc.data.lorenz96`** — The Lorenz-96 model, a configurable high-dimensional system commonly used in weather prediction research.

```python
U, t = orc.data.lorenz96(tN=100, dt=0.01)
```

### Spatiotemporal Systems (PDEs)

**`orc.data.KS_1D`** — The 1D Kuramoto-Sivashinsky equation, a PDE exhibiting spatiotemporal chaos. Useful for testing parallel reservoir architectures.

```python
U, t = orc.data.KS_1D(tN=100, dt=0.01)
```

## Usage with ORC Models

All data generators produce output in the standard format expected by ORC training functions:

```python
import orc

# Generate data
U, t = orc.data.lorenz63(tN=100, dt=0.01)

# Train-test split
split_idx = int(0.8 * U.shape[0])
U_train, U_test = U[:split_idx], U[split_idx:]

# Use directly with ORC models
esn = orc.forecaster.ESNForecaster(data_dim=U.shape[1], res_dim=500, seed=42)
esn, R = orc.forecaster.train_ESNForecaster(esn, U_train)
```

For more detailed examples, see the [Data Library notebook](../examples/data_library.ipynb).
