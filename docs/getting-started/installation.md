# Installation

## System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: CPU or NVIDIA GPU

## Installation Options

### Basic CPU Installation

For CPU-only usage, clone the repository and install:

```bash
git clone https://github.com/dtretiak/OpenReservoirComputing.git
cd OpenReservoirComputing
pip install .
```

### GPU Installation

For GPU acceleration with CUDA support:

```bash
git clone https://github.com/dtretiak/OpenReservoirComputing.git
cd OpenReservoirComputing
pip install .[gpu]
```


### Development Installation

For contributors or advanced users who want to modify the code:

```bash
git clone https://github.com/dtretiak/OpenReservoirComputing.git
cd OpenReservoirComputing
pip install -e .[dev]
```

This includes additional tools for testing, formatting, and documentation.

### Complete Installation

To install all optional dependencies (GPU, development, notebooks, documentation):

```bash
pip install -e .[all]
```

## Verification

Test your installation with this simple script:

```python
import orc
import jax

# Check JAX device (CPU or GPU)
print(f"JAX is using: {jax.devices()}")

# Test basic ORC functionality
U, t = orc.data.lorenz63(tN=10, dt=0.01)
print(f"Generated Lorenz data shape: {U.shape}")

# Test model creation
esn = orc.models.ESNForecaster(data_dim=3, res_dim=100)
print("ESN created successfully!")
print("Installation verified ✓")
```

## Troubleshooting

### Common Issues

**JAX Not Found**
```
ModuleNotFoundError: No module named 'jax'
```
*Solution*: Ensure you're using Python 3.10-3.12 and install with `pip install .[gpu]` or upgrade pip.

**CUDA Errors (GPU)**
```
CUDA_ERROR_NO_DEVICE or similar GPU errors
```
*Solution*: Verify NVIDIA drivers are installed and compatible with CUDA 12.x.

**Import Errors**
```
ImportError: cannot import name 'x' from 'orc'
```
*Solution*: Try reinstalling: `pip uninstall OpenReservoirComputing && pip install .[gpu]`

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/dtretiak/OpenReservoirComputing/issues)
2. Ensure your Python version is supported (3.10-3.12)
3. Try a clean virtual environment installation
4. For GPU issues, verify CUDA installation with `nvidia-smi`
