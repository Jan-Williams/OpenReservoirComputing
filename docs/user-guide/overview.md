# Overview

## What is Reservoir Computing?

Reservoir Computing (RC) is a computational framework for processing temporal and sequential data. Unlike traditional recurrent neural networks that require expensive backpropagation through time, RC keeps the recurrent layer (the *reservoir*) fixed and only trains a simple output layer (the *readout*). This makes training fast, often reducing to a single ridge regression solve, while still capturing complex temporal dynamics.

RC is particularly effective for:

- **Chaotic system forecasting** — predicting trajectories of systems like the Lorenz attractor
- **Spatiotemporal modeling** — forecasting PDEs like the Kuramoto-Sivashinsky equation
- **Time series prediction** — general sequential data analysis
- **System identification and control** — learning input-output relationships of dynamical systems

## Core Architecture

ORC decomposes every reservoir computing model into three modular layers:

```
Input → [Embedding] → [Driver] → [Readout] → Output
```

### Embedding Layer

The embedding layer maps input signals from the data dimension into the reservoir dimension. It defines *how* external data enters the reservoir.

- `ParallelLinearEmbedding` — random linear projection with support for spatial decomposition via overlapping chunks
- `LinearEmbedding` — single-reservoir linear embedding
- `EnsembleLinearEmbedding` — multiple independent embeddings for ensemble models

See [Embeddings](embeddings.md) for details.

### Driver Layer

The driver layer implements the reservoir's temporal dynamics. It takes embedded input and the current reservoir state, then computes the next state.

- `ParallelESNDriver` / `ESNDriver` — Echo State Network dynamics with tanh nonlinearity, supporting discrete and continuous time modes
- `ParallelTaylorDriver` / `TaylorDriver` — Taylor-expanded ESN dynamics for computational efficiency
- `GRUDriver` — Gated Recurrent Unit dynamics

See [Drivers](drivers.md) for details.

### Readout Layer

The readout layer is the trainable component. It maps reservoir states to desired outputs, typically trained via ridge regression.

- `ParallelLinearReadout` / `LinearReadout` — standard linear mapping
- `ParallelNonlinearReadout` / `NonlinearReadout` — user-defined nonlinear transformations before linear mapping
- `ParallelQuadraticReadout` / `QuadraticReadout` — quadratic feature expansion
- `EnsembleLinearReadout` — ensemble averaging across multiple readouts

See [Readouts](readouts.md) for details.

## Available Models

ORC provides ready-to-use models that combine these layers for common tasks:

### Forecasting

- `ESNForecaster` — discrete-time Echo State Network for autonomous prediction
- `CESNForecaster` — continuous-time ESN using Diffrax ODE solvers
- `EnsembleESNForecaster` — multiple independent ESNs with averaged predictions

### Classification

- `ESNClassifier` — time series classification using reservoir features

### Control

- `ESNController` — system control via reservoir-based state estimation

See [Models](models.md) for usage details.

## Training

ORC models are trained using ridge regression (Tikhonov regularization), which solves a linear system rather than iterating with gradient descent. The typical workflow is:

1. **Force** the reservoir with training data to collect reservoir states
2. **Solve** the ridge regression problem to find readout weights
3. **Forecast** autonomously from the final reservoir state

```python
import orc

# Create and train
esn = orc.forecaster.ESNForecaster(data_dim=3, res_dim=500, seed=42)
esn, res_states = orc.forecaster.train_ESNForecaster(esn, U_train)

# Predict
U_pred = esn.forecast(fcast_len=1000, res_state=res_states[-1])
```

## Parallel Reservoirs

For high-dimensional or spatiotemporal data, ORC supports parallel reservoir architectures. Instead of one large reservoir, the system uses multiple smaller reservoirs that each process a local region of the input with configurable overlap (`locality`) and boundary conditions (`periodic`).

This approach reduces computational cost from $O(N^2)$ to $O(N)$ in the spatial dimension while preserving local structure.

## Navigation

New users should start with the [Quick Start](../getting-started/quickstart.md) tutorial, then read the component guides:

1. [Embeddings](embeddings.md) — input-to-reservoir mapping
2. [Drivers](drivers.md) — reservoir dynamics
3. [Readouts](readouts.md) — reservoir-to-output mapping
4. [Models](models.md) — complete model implementations
5. [Data](data.md) — built-in dynamical systems for benchmarking

The [API Reference](../api/forecaster.md) provides complete technical documentation, and the [Examples](../examples/lorenz.ipynb) demonstrate practical applications.
