import numpy as np

def relu(x):
    return np.maximum(0, x) # Derivative at 0 should be defined as 0

