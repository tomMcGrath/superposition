import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def smooth(x, span):
    """Smooths a time series using pandas EWM."""
    x = pd.Series(x)
    return x.ewm(span=span).mean()


def feature_mat(model):
    """Visualises a model's feature matrix."""
    W = model.W.to('cpu').detach().numpy()
    feature_mat = W @ W.T
    max_abs = np.max(np.abs(feature_mat))
    p = plt.matshow(feature_mat, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    plt.colorbar(p)
    plt.show()