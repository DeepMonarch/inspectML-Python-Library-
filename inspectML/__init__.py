
"""
InspectML: Lightweight Evaluation Metrics Library for ML Models
---------------------------------------------------------------
Provides regression and classification/clustering evaluation metrics.

Submodules:
- evaluation.regression
- evaluation.classification
- utils
"""

from . import evaluation
from . import utils

from .evaluation.regression import evaluate as evaluate_regression
from .evaluation.classification import F1, NMI, ARI, HOM, COMP, VMEAS

__all__ = [
    "evaluation",
    "utils",
    "evaluate_regression",
    "F1", "NMI", "ARI", "HOM", "COMP", "VMEAS"
]
