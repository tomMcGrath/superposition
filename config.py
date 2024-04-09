data_cfg = {
    "data_dim": 20,
    "p_sparse": 0.9,
    "importance_weight": 0.7,
}

model_cfg = {
    "hidden_dim": 5,
}

train_cfg = {
    "warmup_steps": int(1e3),
    "lr": 3e-4,
    "n_steps": int(1e4),
    "batch_size": 8192,
    "resample": True,
}

cfg = {
    "data_config": data_cfg,
    "model_config": model_cfg,
    "training_config": train_cfg,
    "train_device": "cuda:0",
}