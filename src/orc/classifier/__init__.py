"""Classification with Reservoir Computers."""

from orc.classifier.base import RCClassifierBase
from orc.classifier.models import ESNClassifier
from orc.classifier.train import train_ESNClassifier, train_RCClassifier

__all__ = [
    "RCClassifierBase",
    "ESNClassifier",
    "train_RCClassifier",
    "train_ESNClassifier",
]
