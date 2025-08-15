# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- JAX (for GPU acceleration)
- NumPy, Matplotlib, and other scientific computing libraries

## Basic Installation

To install ORC, first clone the repository:

```bash
git clone https://github.com/dtretiak/OpenReservoirComputing.git
cd OpenReservoirComputing
```

Then install using pip:

```bash
pip install .
```

## GPU Support

If you would like to use ORC on GPU(s), install the optional GPU dependencies:

```bash
pip install .[gpu]
```

This will install the CUDA-enabled version of JAX for GPU acceleration.

## Development Installation

For development, install in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

## Notebook Dependencies

To run the example notebooks, install the optional notebook dependencies:

```bash
pip install .[notebooks]
```

## Verification

To verify your installation, run:

```python
import orc
print(f"ORC version: {orc.__version__}")

# Test basic functionality
U, t = orc.data.lorenz63(tN=10, dt=0.01)
print(f"Generated Lorenz data shape: {U.shape}")
```

## Troubleshooting

### JAX Installation Issues

If you encounter issues with JAX installation, particularly for GPU support, refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for platform-specific instructions.

### Import Errors

If you encounter import errors, ensure that:

1. You're using a supported Python version (3.10-3.12)
2. All dependencies are properly installed
3. You're in the correct conda/virtual environment

### Performance Issues

For optimal performance:

1. Install the GPU version of JAX if you have a compatible GPU
2. Use appropriate batch sizes for your hardware
3. Consider using JAX's JIT compilation for repeated computations
