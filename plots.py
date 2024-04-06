import numpy as np
import matplotlib.pyplot as plt


def feature_mat(model):
    W = model.W.to('cpu').detach().numpy()
    feature_mat = W @ W.T
    max_abs = np.max(np.abs(feature_mat))
    plt.matshow(feature_mat, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    plt.show()