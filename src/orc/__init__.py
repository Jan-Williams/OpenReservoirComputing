"""Components of models. Setting x64 on import."""

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

jax.config.update("jax_enable_x64", True)

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
