import numpy as np
import torch


class Linear(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.W = torch.nn.parameter.Parameter(
        torch.randn((input_dim, hidden_dim)) / np.sqrt(input_dim)
    )
    self.b = torch.nn.parameter.Parameter(
        torch.zeros(input_dim)
    )

  def forward(self, x, with_activations=False):
    h = x @ self.W
    z = h @ self.W.T + self.b
    if with_activations:
      return z, {'h': h, 'z': z}
    else:
      return z
    

class ReluOut(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.W = torch.nn.parameter.Parameter(
        torch.randn((input_dim, hidden_dim)) / np.sqrt(input_dim)
    )
    self.b = torch.nn.parameter.Parameter(
        torch.zeros(input_dim)
    )

  def forward(self, x, with_activations=False):
    h = x @ self.W
    z = h @ self.W.T + self.b
    if with_activations:
      return torch.nn.functional.relu(z), {'h': h, 'z': z}
    else:
      return torch.nn.functional.relu(z)


class SparseAutoEncoder(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.encoder = torch.nn.Linear(
      input_dim, hidden_dim
    )
    self.decoder = torch.nn.Linear(
      hidden_dim, input_dim
    )

  def forward(self, x, with_activations=False):
    f_preactivations = self.encoder(x)
    f = torch.nn.functional.relu(f_preactivations)
    l1_norm = torch.linalg.norm(f, ord=1, dim=1)
    if with_activations:
        return self.decoder(f), l1_norm, f
    else:
        return self.decoder(f), l1_norm
