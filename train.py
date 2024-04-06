import datasets
import models
import torch
import tqdm


def make_loss_fn(importance_weights):
    def loss_fn(x, x_hat):
        errs = (x - x_hat)**2
        errs = errs * importance_weights
        per_sample_errs = torch.sum(errs, dim=1)
        return torch.mean(per_sample_errs)
    
    return loss_fn


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
    print(model.W.shape)
    
    # Training setup
    importance_weights = dataset.make_importance_weights(data_cfg['importance_weight'])
    loss_fn = make_loss_fn(importance_weights)
    optimizer=torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'])
    
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

        losses.append(loss.item())
        
    # Prepare return values
    train_outputs = {
        'model': model,
        'dataset': dataset,
        'losses': losses
    }
    
    return train_outputs