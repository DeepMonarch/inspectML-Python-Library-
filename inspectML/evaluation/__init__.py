"""
InspectML Evaluation Module
----------------------------
Contains metrics for regression and classification (clustering) models.
"""

from . import regression
from . import classification

__all__ = ["regression", "classification"]
