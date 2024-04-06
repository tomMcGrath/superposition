import torch
import numpy as np


class SparseDataset:
    """Synthetic dataset from "Toy Models of Superposition"
    https://transformer-circuits.pub/2022/toy_model/index.html
    """

    def __init__(self, raw_dist, sparsity_dist):
        """Takes two pytorch distributions, one for p(sparse) per dim and the other for the unsparsified distribution."""
        self._raw_dist = raw_dist
        self._sparsity_dist = sparsity_dist
        self._check_dims_match()
        self.d = self._raw_dist.low.shape[0]  # dimensionality of data
        self.device = self._raw_dist.low.device

    def _check_dims_match(self):
        """Check the two distributions have the same shape samples."""
        raw_sample = self._raw_dist.sample(sample_shape=[2,])
        sparsity_sample = self._sparsity_dist.sample(sample_shape=[2,])
        assert raw_sample.shape == sparsity_sample.shape     
    
    def sample(self, n_samples):
        """Sample from the distribution by first sampling from the raw distribution and then sparsifying."""
        sample = self._raw_dist.sample(sample_shape=[n_samples,])
        return sample * self._sparsity_dist.sample(sample_shape=[n_samples,])
        
    def make_importance_weights(self, decay_val):
        """Create set of feature importance weights: w_i = decay_val^i"""
        return torch.pow(decay_val, torch.arange(self.d)).to(self.device)
    
    
def make_bernoulli_dist(p, d, device="cpu"):
    probs = torch.tensor([p] * d).to(device)
    return torch.distributions.bernoulli.Bernoulli(probs)


def make_uniform_dist(d, device="cpu"):
    low = torch.tensor([0.,] * d).to(device)
    high = torch.tensor([1.,] * d).to(device)
    return torch.distributions.uniform.Uniform(low, high)
