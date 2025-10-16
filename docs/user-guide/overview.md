## Core Architecture

The **embedding layer** handles input-to-reservoir mapping. `LinearEmbedding` provides matrix multiplication with support for parallel reservoirs and configurable connectivity patterns.

The **driver layer** propagates reservoir state evolution. `ParallelESNDriver` implements Echo State Network dynamics for both discrete-time updates and continuous-time evolution using ODE solvers with adaptive step control.

The **readout layer** represents the trainable component. Options include `LinearReadout`, `QuadraticReadout`, and `NonlinearReadout`.

## Available Models

`ESNForecaster` provides standard discrete-time implementation with configurable leak rates, spectral radii, and density. `CESNForecaster` offers continuous-time dynamics using diffrax solvers with cubic interpolation for input forcing. Both support parallel architectures and memory-efficient batched training.

## Navigation

New users should read [Drivers](drivers.md), [Embeddings](embeddings.md), and [Readouts](readouts.md) for details on model components. [Models](models.md) will then detail how to merge these components. The [API section](../api/) provides complete technical documentation, while examples demonstrate practical applications.
