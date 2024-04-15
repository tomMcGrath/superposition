import config
import datasets
import torch
import train


if __name__ == '__main__':
    # Get config
    cfg = config.cfg
    device = cfg['train_device']

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

    # SAE training
    sae_train_outs = train.train_sae(cfg, model, dataset)
