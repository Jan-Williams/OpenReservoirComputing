# OpenReservoirComputing (ORC)

<div align="center">
  <img src="imgs/ORC_logo_cropped.png" alt="ORC Logo" width="200px" />
</div>

**ORC** is the one-stop-shop for performant reservoir computing in JAX. This package provides GPU-accelerated implementations of common reservoir computing architectures with a modular design that allows for easy experimentation and customization.

## Key Features

- **Modular Design**: Mix and match layers and reservoir drivers or create your own
- **Multiple Implementations**: Continuous, discrete, serial, and parallel implementations
- **GPU Acceleration**: Built on JAX for high-performance computing on GPUs
- **Hyperparameter Tuning**: Built-in support for hyperparameter optimization with Ray Tune (coming soon!)
- **Multi-GPU Support**: Training and inference across multiple GPUs (coming soon!)

## What is Reservoir Computing?

Reservoir Computing (RC) is a framework for computation based on the principle of generalized synchronization. Only the readout from the reservoir is trained while the reservoir dynamics are fixed, making RC computationally efficient and particularly well-suited for temporal/sequential data processing.

## Quick Example

```python
import orc

# Integrate the Lorenz system 
U, t = orc.data.lorenz63(tN=100, dt=0.01)

# Train-test split
test_perc = 0.2
split_idx = int((1 - test_perc) * U.shape[0])
U_train = U[:split_idx, :]
U_test = U[split_idx:, :]

# Create and train ESN forecaster
forecaster = orc.models.ESNForecaster(res_dim=500)
forecaster, res_states = orc.models.train_ESNForecaster(forecaster, U_train)

# Make predictions
U_pred = forecaster.forecast(fcast_len=len(U_test), res_state=res_states[-1])
```

## Getting Started

- **[Installation](getting-started/installation.md)**: Install ORC and its dependencies
- **[Quick Start](getting-started/quickstart.md)**: Get up and running with your first reservoir computer
- **[User Guide](user-guide/overview.md)**: Comprehensive guide to ORC's features and capabilities

## API Documentation

Explore the complete API reference:

- **[Models](api/models.md)**: ESN and CESN forecaster implementations
- **[RC Base Classes](api/rc.md)**: Core reservoir computing base classes
- **[Drivers](api/drivers.md)**: Reservoir dynamics drivers
- **[Embeddings](api/embeddings.md)**: Input embedding layers
- **[Readouts](api/readouts.md)**: Output readout layers

## Examples

Check out our Jupyter notebook examples:

- **[Lorenz System](examples/lorenz.ipynb)**: Classic chaotic system forecasting
- **[Continuous RC](examples/continuous_rc.ipynb)**: Continuous-time reservoir computing
- **[RC Background](examples/rc_background.ipynb)**: Introduction to reservoir computing concepts

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details on how to get involved.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation

If you use ORC in your research, please cite:

```bibtex
@software{orc2024,
  title={OpenReservoirComputing: GPU-accelerated reservoir computing in JAX},
  author={Tretiak, Dima and Williams, Jan P.},
  year={2024},
  url={https://github.com/dtretiak/OpenReservoirComputing}
}
```
