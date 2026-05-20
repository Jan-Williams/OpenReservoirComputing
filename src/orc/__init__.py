"""Components of models."""

import warnings

import jax

from orc import (
    classifier,
    control,
    data,
    drivers,
    embeddings,
    forecaster,
    readouts,
    utils,
)

if not getattr(jax.config, "jax_enable_x64", False):
    warnings.warn(
        "For good performance, orc often requires float64 precision. Enable it " \
        "before importing orc with: "
        "jax.config.update('jax_enable_x64', True)",
        UserWarning,
        stacklevel=2,
    )

__all__ = [
    "forecaster",
    "classifier",
    "control",
    "drivers",
    "embeddings",
    "readouts",
    "data",
    "utils",
]
