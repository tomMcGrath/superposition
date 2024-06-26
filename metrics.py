import numpy as np
import torch


def get_superpos(W, idx):
    """Measures the degree to which W[idx] is superposed with other rows."""
    sp = 0
    
    for i in range(W.shape[0]):
        if i == idx:
            continue
            
        else:
            sp += np.dot(W[idx], W[i])**2
            
    return sp


def mean_l0(x):
    """Returns mean number of entries of x that are larger than eps."""
    per_sample_norms = torch.linalg.vector_norm(x, ord=0, dim=1)
    return torch.mean(per_sample_norms)


def live_neurons(f):
    """Returns the number of neurons that ever fire on a batch."""
    return torch.count_nonzero(torch.sum(f, dim=0))
