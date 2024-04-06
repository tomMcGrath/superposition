import datasets
import models
import numpy as np
import torch
import tqdm


def make_loss_fn(importance_weights):
    def loss_fn(x, x_hat):
        errs = (x - x_hat)**2
        errs = errs * importance_weights
        per_sample_errs = torch.sum(errs, dim=1)
        return torch.mean(per_sample_errs)
    
    return loss_fn


# Use this inside a LambdaLR
def build_scheduler(warmup_steps, lr_max, total_steps):
  def scheduler(t):
    if t <= warmup_steps:  # linear warmup
      return t * lr_max / warmup_steps

    else:  # cosine decay -> 0
      t_cos = t - warmup_steps
      cos_steps = total_steps - warmup_steps
      return 0.5 * lr_max * (1 +  np.cos(np.pi * t_cos / cos_steps))

  return scheduler


def train(config, progressbar=False):
    """Full training loop"""
    # Unpack configs
    data_cfg = config['data_config']
    model_cfg = config['model_config']
    train_cfg = config['training_config']
    device = config['train_device']
    
    # Dataset setup
    n = data_cfg['data_dim']
    S = data_cfg['p_sparse']
    raw_dist = datasets.make_uniform_dist(n, device)
    sparsity_dist = datasets.make_bernoulli_dist(S, n, device=device)
    dataset = datasets.SparseDataset(raw_dist, sparsity_dist)
    
    # Model setup
    m = model_cfg['hidden_dim']
    model = models.ReluOut(n, m).to(device)
    
    # Training setup
    importance_weights = dataset.make_importance_weights(
       data_cfg['importance_weight'])
    loss_fn = make_loss_fn(importance_weights)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1)  # lr from scheduler
    lr_fn = build_scheduler(
       train_cfg['warmup_steps'],
       train_cfg['lr'],
       train_cfg['n_steps']
       )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    
    # Train loop
    losses = []
    if progressbar:
        step_iter = tqdm.tqdm(range(train_cfg['n_steps']))

    else:
        step_iter = range(train_cfg['n_steps'])

    for _ in step_iter:
        x = dataset.sample(train_cfg['batch_size'])
        x_hat = model(x)
        loss = loss_fn(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        
    # Prepare return values
    train_outputs = {
        'model': model,
        'dataset': dataset,
        'losses': losses
    }
    
    return train_outputs