import config
import copy
import datasets
import functools
import models
import metrics
import torch
import wandb
import train
import utils


if __name__ == '__main__':
    # Get config
    cfg = config.cfg
    device = cfg['train_device']
    sweep_cfg = utils.convert_or_recurse(
        copy.deepcopy(cfg),
        utils.convert_to_sweep_format
        )

    # Initialise sweep
    sweep_cfg['method'] = 'grid'
    sweep_cfg['sae_config'].update({
        'l1_weight': {'values': [1e-3, 1e-2, 1e-1, 1e0]},
        'norm_p': {'values': [0.2, 0.5, 0.7, 1.]},
        })

    # Load model
    model_path = 'models/test.pt'
    model = torch.load(model_path)

    # Initialise dataset
    data_cfg = cfg['data_config']
    n = data_cfg['data_dim']
    S = data_cfg['p_sparse']
    raw_dist = datasets.make_uniform_dist(n, device)
    sparsity_dist = datasets.make_bernoulli_dist(S, n, device=device)
    dataset = datasets.SparseDataset(raw_dist, sparsity_dist)

    # Sweep
    train_with_fixed_model_and_ds = functools.partial(
        train.train_sae,
        model=model,
        dataset=dataset,
        progressbar=False,
    )
    sweep_id = wandb.sweep(sweep_cfg, project="sae-sweeps-test")
    wandb.agent(sweep_id, train_with_fixed_model_and_ds)
