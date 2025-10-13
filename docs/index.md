# OpenReservoirComputing (ORC)

<div align="center">
  <img src="imgs/ORC_logo_cropped.png" alt="ORC Logo" width="200px" />
</div>

OpenReservoirComputing (ORC) provides GPU-accelerated implementations of reservoir computing architectures for time series forecasting and temporal data analysis. Built on JAX, ORC delivers high-performance computing capabilities with a modular design that supports both research experimentation and practical applications.

## Overview

Reservoir Computing (RC) is a computational framework for processing temporal and sequential data. The approach trains only the output layer (readout) while keeping the reservoir dynamics fixed, making it computationally efficient and particularly effective for chaotic system forecasting, time series prediction, and nonlinear system identification.

## Key Features

**Modular Architecture**: Composable embedding layers, reservoir drivers, and readout functions that can be mixed, matched, or extended with custom implementations.

**Multiple Paradigms**: Support for discrete- and continuous-time Echo State Networks (ESNs) and parallel reservoir architectures.

**Comprehensive Tooling**: Built-in chaotic dynamical systems, visualization utilities, regression solvers, and performance evaluation metrics.

## Architecture

ORC decomposes reservoir computing models into three core components:

- **Embedding Layer**: Maps input signals to reservoir dimension through linear or nonlinear transformations
- **Driver Layer**: Implements reservoir dynamics (ESN, continuous-time ESN, custom)
- **Readout Layer**: Learns mappings from reservoir states to desired outputs

This modular design enables systematic exploration of different architectural choices while maintaining consistent interfaces across implementations.

## Quick Example

```python
import orc

# Generate Lorenz system data
U, t = orc.data.lorenz63(tN=100, dt=0.01)

# Train-test split
test_perc = 0.2
split_idx = int((1 - test_perc) * U.shape[0])
U_train = U[:split_idx, :]
U_test = U[split_idx:, :]

# Create and train Echo State Network
esn = orc.models.ESNForecaster(data_dim=3, res_dim=500, seed=42)
esn, res_states = orc.models.train_ESNForecaster(esn, U_train)

# Generate forecast
U_pred = esn.forecast(fcast_len=U_test.shape[0], res_state=res_states[-1])

# Visualize results
orc.utils.visualization.plot_time_series(
    [U_test, U_pred],
    t[split_idx:] - t[split_idx],
    state_var_names=["x", "y", "z"],
    time_series_labels=["Ground Truth", "Forecast"]
)
```

## Documentation

### Getting Started
- [**Installation**](getting-started/installation.md): System requirements and installation procedures for CPU and GPU configurations
- [**Quick Start**](getting-started/quickstart.md): Step-by-step tutorial for training your first reservoir computer

### User Guide
- [**Overview**](user-guide/overview.md): Comprehensive guide to ORC's capabilities and design principles
- [**Models**](user-guide/models.md): Available reservoir computing implementations and training procedures
- [**Architecture Components**](user-guide/drivers.md): Detailed documentation of embedding, driver, and readout layers

### API Reference
Complete technical documentation of all classes, functions, and interfaces:

- [**Models**](api/models.md): ESN and continuous ESN forecaster implementations

- [**RC Base Classes**](api/rc.md): Core reservoir computing abstractions

- [**Drivers**](api/drivers.md): Driver base class and specific implementations

- [**Embeddings**](api/embeddings.md): Embedding base class and specific implementations

- [**Readouts**](api/readouts.md): Readout base class and specific implementations

- [**Utilities**](api/utils.md): Data generation, visualization, and analysis tools

### Examples
Practical demonstrations using Jupyter notebooks:

- [**Lorenz System Forecasting**](examples/lorenz.ipynb): Classic chaotic system prediction benchmark

- [**RC Background**](examples/rc_background.ipynb): Theoretical foundations and implementation details

## Applications

**Scientific Computing**: Forecasting chaotic dynamical systems, analyzing spatiotemporal phenomena, and modeling complex temporal dependencies.

**Time Series Analysis**: Financial forecasting, climate modeling, sensor data processing, and anomaly detection in sequential data.

**Control Systems**: State estimation, system identification, and real-time prediction for feedback control applications.

**Machine Learning Research**: Investigating novel reservoir architectures, hyperparameter optimization strategies, and performance benchmarking.

## Technical Specifications

**Dependencies**: JAX ecosystem with Equinox for functional programming, Diffrax for ODE integration, and standard scientific Python libraries.

**Performance**: Optimized for both CPU and GPU execution with support for parallel reservoirs and efficient memory management.

**Precision**: x64 floating-point arithmetic enabled by default for numerical stability in chaotic system forecasting.

**Compatibility**: Python 3.10-3.12 on Linux, macOS, and Windows platforms.

## Contributing

ORC welcomes contributions from the reservoir computing community. See the [Contributing Guide](contributing.md) for development procedures, coding standards, and submission guidelines.

## Citation

When using ORC in research publications, please cite:

```bibtex
@software{orc2024,
  title={OpenReservoirComputing: GPU-accelerated reservoir computing in JAX},
  author={Tretiak, Dima and Williams, Jan P.},
  year={2024},
  url={https://github.com/dtretiak/OpenReservoirComputing}
}
```

## License

This project is distributed under the Apache License. See the LICENSE file for complete terms and conditions.