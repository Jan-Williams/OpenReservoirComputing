# Contributing to OpenReservoirComputing

We welcome contributions to the OpenReservoirComputing project! This guide will help you get started.

## Development Setup

ORC uses [uv](https://docs.astral.sh/uv/) for dependency and environment management. uv
downloads and manages the Python interpreter for you, so conda or pyenv are not required.

1. **Install uv** (see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Jan-Williams/OpenReservoirComputing.git
   cd OpenReservoirComputing
   ```

3. **Create the development environment**:
   ```bash
   uv sync
   ```
   This creates `.venv` using the Python version pinned in `.python-version`, installs ORC
   in editable mode, and installs the `dev` dependency group at the exact versions recorded
   in `uv.lock`.

   Optional additions:
   ```bash
   uv sync --all-groups        # + the `docs` group (mkdocs, notebook, ipykernel)
   uv sync --extra gpu         # + CUDA-enabled JAX (Linux only)
   uv sync --extra notebooks   # + Jupyter, for running the examples
   ```

!!! note "Running commands"
    There is no need to activate the virtual environment. Prefix commands with `uv run` and
    uv will use — and if necessary update — the project environment automatically.

!!! warning "The dev extra has become a dependency group"
    Development and documentation dependencies are now
    [PEP 735](https://peps.python.org/pep-0735/) dependency *groups* rather than extras, so
    `pip install -e ".[dev]"` no longer works. Use `uv sync` as above, or with pip 25.1 or
    newer:

    ```bash
    pip install -e . --group dev
    ```

    Both the `-e .` and the `--group dev` are needed — `--group` installs the group's tools
    but not ORC itself.

## Code Style

We use the following tools for code quality:

- **Ruff**: Linting and formatting
- **pytest**: Testing
- **ty**: Type checking

Run these before submitting:
```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run pytest tests/
uv run ty check src/
```

## Dependencies and the lockfile

`uv.lock` is committed to the repository and is the source of truth for CI. To add or change
a dependency:

```bash
uv add <package>               # runtime dependency
uv add --group dev <package>   # development tool
uv add --group docs <package>  # documentation tool
```

`uv add` updates both `pyproject.toml` and `uv.lock`. **Commit `uv.lock` alongside your
change** — CI runs `uv sync --locked`, which fails if the lockfile is out of date. To refresh
pinned versions without changing any constraints, run `uv lock --upgrade`.

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting

## Documentation

- Update docstrings for new functions/classes
- Follow numpy docstring style
- Ensure type annotations are correct with `ty`
- Preview the site with `uv run mkdocs serve` (this **executes** every example notebook, so
  the first build takes a few minutes)

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
