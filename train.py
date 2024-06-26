import datasets
import models
import metrics
import numpy as np
import torch
import tqdm
import wandb


"""
Toy model training code
"""
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

    # W&B setup
    run = wandb.init(
       project="toy-saes-base",
       config=config,
    )
    
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
    # TODO(tom): add weight decay
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

    for step in step_iter:
        # Might want to stick to full-batch GD, e.g. for
        # https://transformer-circuits.pub/2023/toy-double-descent/index.html
        if step == 0:
           x = dataset.sample(train_cfg['batch_size'])

        if step > 0 and train_cfg['resample']:
            x = dataset.sample(train_cfg['batch_size'])
        x_hat = model(x)
        loss = loss_fn(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        # Logging
        wandb.log({'loss': loss.item()})
        
    # Prepare return values
    train_outputs = {
        'model': model,
        'dataset': dataset,
        'losses': losses
    }
    
    return train_outputs


"""
SAE training code
"""
def mse(x, x_hat):
    errs = (x - x_hat)**2
    return torch.mean(torch.sum(errs, dim=1))


def train_sae(config, model, dataset, progressbar=True):
    # Unpack configs
    _ = config['data_config']
    model_cfg = config['model_config']
    train_cfg = config['training_config']
    sae_cfg = config['sae_config']
    device = config['train_device']

    # Setup W&B
    run = wandb.init(
       project="toy-saes-sae",
       config=config,
    )

    # Hyperparams
    n_steps = train_cfg['n_steps_sae']
    sae_dim = sae_cfg['sae_dim']
    l1_weight = sae_cfg['l1_weight']

    # Setup SAE
    sae = models.SparseAutoEncoder(
       model_cfg["hidden_dim"], 
       sae_dim,
       norm_p=sae_cfg['norm_p']
       ).to(device)
    optimizer=torch.optim.AdamW(sae.parameters(), lr=1.)
    lr_fn = build_scheduler(
       train_cfg['warmup_steps_sae'],
       train_cfg['lr_sae'],
       train_cfg['n_steps_sae']
       )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    # Training loop
    losses = []
    for t in tqdm.tqdm(range(n_steps)):
        x = dataset.sample(train_cfg['batch_size_sae'])  # sample from dataset
        _, activations = model(x, with_activations=True)  # put into superpos
        x_recon, l1_norm, f = sae(activations['h'], with_activations=True)  # SAE
        recon_loss = mse(activations['h'], x_recon)
        l1_norm = torch.mean(l1_norm)
        loss = recon_loss + (1. * t / n_steps) * l1_norm * l1_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        losses.append(loss.item())
        wandb.log({
           'sae_loss': loss.item(),
           'recon_loss': recon_loss.item(),
           'l1_norm': l1_norm.item(),
           'live_neurons': metrics.live_neurons(f).item(),
           'l0': metrics.mean_l0(f).item(),
        })

    train_outs = {
       'sae': sae,
       'losses': losses,
    }

    return train_outs
