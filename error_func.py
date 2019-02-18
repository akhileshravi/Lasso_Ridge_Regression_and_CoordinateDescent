import numpy as np

def rmse_calc(y_test, pred):
    mse = sum((y_test - pred)**2)/len(y_test)
    return np.sqrt(mse)

def mae_calc(y_test, pred):
    mae = sum(abs(y_test - pred))/len(y_test)
    return mae
