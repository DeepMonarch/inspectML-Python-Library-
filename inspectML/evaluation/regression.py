import numpy as np

def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(mse(y_true, y_pred))

def evaluate(y_true, y_pred):
    """Evaluate regression metrics"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        'MSE': mse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred)
    }

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true, y_pred):
    """R-squared (Coefficient of Determination)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def adj_r2(y_true, y_pred, n_features):
    """Adjusted R-squared"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    r_squared = r2(y_true, y_pred)
    return 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)

def msle(y_true, y_pred):
    """Mean Squared Logarithmic Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(msle(y_true, y_pred))

def medAE(y_true, y_pred):
    """Median Absolute Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs(y_true - y_pred))

def evs(y_true, y_pred):
    """Explained Variance Score"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y

