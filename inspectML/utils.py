"""
Utility functions for InspectML
-------------------------------
General helper functions used across modules.
"""

import numpy as np

def safe_divide(a, b, default=np.nan):
    """Safely divide two numbers or arrays, returning default if division by zero occurs."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = default  # -inf, inf, NaN -> default
    return result

def check_inputs(y_true, y_pred):
    """Validate y_true and y_pred arrays before metric calculation."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length.")
    return np.array(y_true), np.array(y_pred)
