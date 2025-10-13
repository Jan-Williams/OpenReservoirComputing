# Contributing to OpenReservoirComputing

We welcome contributions to the OpenReservoirComputing project! This guide will help you get started.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/OpenReservoirComputing.git
   cd OpenReservoirComputing
   ```

2. **Create a development environment**:
   ```bash
   conda create -n orc-dev python=3.12
   conda activate orc-dev
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .[dev,notebooks,gpu]
   ```

## Code Style

We use the following tools for code quality:

- **Ruff**: Linting and formatting
- **pytest**: Testing

Run these before submitting:
```bash
ruff format src/ tests/
ruff check src/ tests/
pytest tests/
```

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting

## Documentation

- Update docstrings for new functions/classes
- Follow numpy docstring style

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Update documentation as needed
4. Ensure all checks pass
5. Submit a pull request with a clear description

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

Thank you for contributing!
