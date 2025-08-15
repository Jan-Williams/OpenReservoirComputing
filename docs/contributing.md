# Contributing to OpenReservoirComputing

We welcome contributions to the OpenReservoirComputing project! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/OpenReservoirComputing.git
   cd OpenReservoirComputing
   ```

2. **Create a development environment**:
   ```bash
   conda create -n orc-dev python=3.11
   conda activate orc-dev
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .[dev,notebooks,gpu]
   ```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **pytest**: Testing

Run these before submitting:
```bash
black src/ tests/
ruff check src/ tests/
pytest tests/
```

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting
- Add docstring examples that can be tested

## Documentation

- Update docstrings for new functions/classes
- Add examples to the documentation
- Update the changelog for significant changes

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
