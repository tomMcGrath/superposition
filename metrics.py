import numpy as np


def get_superpos(W, idx):
    """Measures the degree to which W[idx] is superposed with other rows."""
    sp = 0
    
    for i in range(W.shape[0]):
        if i == idx:
            continue
            
        else:
            sp += np.dot(W[idx], W[i])**2
            
    return sp
