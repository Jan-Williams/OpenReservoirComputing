"""Reservoir Computer Controllers.

This module provides reservoir computing functionality for controlled dynamical
systems where exogenous control inputs influence the system at each time step.
"""

from orc.control.base import RCControllerBase
from orc.control.models import ESNController
from orc.control.train import train_ESNController

__all__ = [
    "RCControllerBase",
    "ESNController",
    "train_ESNController",
]
