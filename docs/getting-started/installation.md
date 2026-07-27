# Installation

## System Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: CPU everywhere; NVIDIA GPU acceleration on Linux only (JAX does not publish
  CUDA builds for macOS or native Windows — Windows users can use WSL2)

## Installation Options

The easiest way to get started with ORC is to install from PyPI:
```bash
pip install OpenReservoirComputing
```

To add ORC to a [uv](https://docs.astral.sh/uv/)-managed project:
```bash
uv add OpenReservoirComputing
```

ORC has two optional extras. For GPU acceleration with CUDA:

```bash
pip install "OpenReservoirComputing[gpu]"
```

For Jupyter, to run the example notebooks:

```bash
pip install "OpenReservoirComputing[notebooks]"
```

Both can be combined: `pip install "OpenReservoirComputing[gpu,notebooks]"`.

!!! warning "The `gpu` extra is Linux-only"
    JAX publishes CUDA wheels for Linux (x86_64 and aarch64) only — there are no CUDA builds
    for macOS or native Windows. The `gpu` extra therefore carries a
    `sys_platform == "linux"` marker: on macOS and native Windows it installs **nothing extra
    and raises no error**, leaving you with the CPU build.

    If you are on Windows with an NVIDIA GPU, install under
    [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install), which reports as Linux and
    does receive the CUDA build.

### Installing from source

If you're interested in the latest, unreleased version, clone the repository and install from
the working tree. Extras use the same names, applied to the local path:

```bash
git clone https://github.com/Jan-Williams/OpenReservoirComputing.git
cd OpenReservoirComputing
pip install .              # CPU only
pip install ".[gpu]"       # with CUDA-enabled JAX (Linux only)
```

### Development Installation

ORC's development workflow uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Jan-Williams/OpenReservoirComputing.git
cd OpenReservoirComputing
uv sync                             # editable install + `dev` group (ruff, ty, pytest, coverage)
uv sync --all-groups                # additionally installs the `docs` group
uv sync --all-groups --all-extras   # everything, incl. GPU on Linux and Jupyter
```

!!! note "Using pip instead of uv"
    The development and documentation dependencies are
    [PEP 735](https://peps.python.org/pep-0735/) dependency *groups* rather than extras, so
    they are not reachable through extras syntax. If you would rather use pip than uv, the
    equivalent needs both the project and the group (pip 25.1 or newer):

    ```bash
    pip install -e . --group dev
    ```

    Note that `pip install --group dev` on its own installs the group's tools but **not** ORC
    itself, so the editable `-e .` is required.

See [Contributing](../contributing.md) for the full development workflow.

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
esn = orc.forecaster.ESNForecaster(data_dim=3, res_dim=100)
print("ESN created successfully!")
print("Installation verified ✓")
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/Jan-Williams/OpenReservoirComputing/issues)
2. Ensure your Python version is supported (3.10-3.13)
3. Try a clean virtual environment installation
4. For GPU issues, verify CUDA installation with `nvidia-smi`
