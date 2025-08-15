# User Guide Overview

Welcome to the OpenReservoirComputing (ORC) user guide! This comprehensive guide will help you understand and effectively use all features of the ORC package.

## What You'll Learn

This user guide covers:

- **[Reservoir Computing Basics](rc-basics.md)**: Fundamental concepts and theory
- **[Models](models.md)**: ESN and CESN forecaster implementations
- **[Drivers](drivers.md)**: Reservoir dynamics and state evolution
- **[Embeddings](embeddings.md)**: Input embedding layers
- **[Readouts](readouts.md)**: Output layers
- **[Data Library](data.md)**: Built-in datasets and data generation

## Package Architecture

ORC follows a modular design with clear separation of concerns:

```
src/orc/
├── __init__.py        # Package initialization and imports
├── rc.py              # Base classes for reservoir computing
├── drivers.py         # Reservoir dynamics implementations  
├── embeddings.py      # Input preprocessing layers
├── readouts.py        # Output layers and training
├── models/            # High-level forecaster classes
│   ├── __init__.py
│   └── esn.py         # ESN and CESN forecaster implementations
├── data/              # Data generation and datasets
│   ├── __init__.py
│   └── integrators.py # Dynamical systems and integrators
└── utils/             # Utility functions and helpers
    ├── __init__.py
    ├── numerics.py    # Numerical utilities
    └── visualization.py # Plotting and visualization tools
```

## Design Philosophy

### Modularity
Each component can be mixed and matched or replaced with custom implementations. Want a custom reservoir driver? Just inherit from the base class and implement the required methods.

### Performance
Built on JAX for:
- **GPU acceleration**: Automatic GPU utilization when available
- **JIT compilation**: Fast execution through just-in-time compilation
- **Vectorization**: Efficient batch processing
- **Automatic differentiation**: For advanced optimization techniques

## Common Workflows

### Basic Forecasting
1. Load or generate time series data
2. Create a forecaster (ESN or CESN)
3. Train the forecaster on historical data
4. Generate predictions for future time steps

### Custom Components
1. Inherit from appropriate base classes
2. Implement required abstract methods
3. Test with existing data and models
4. Integrate into forecasting pipeline

## Best Practices

TODO.

## Getting Help

- **Examples**: Check the [Examples](../examples/) section for Jupyter notebooks
- **API Reference**: Detailed documentation in [API Reference](../api/)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/dtretiak/OpenReservoirComputing/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/dtretiak/OpenReservoirComputing/discussions)

Ready to dive deeper? Start with [Reservoir Computing Basics](rc-basics.md) to understand the fundamental concepts.
