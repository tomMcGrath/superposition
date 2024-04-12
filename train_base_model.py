import config
import torch
import train


if __name__ == '__main__':
    """Train and then serialise a base model."""
    model_outputs = train.train(config.cfg, progressbar=True)
    model = model_outputs['model']
    torch.save(model, 'models/test.pt')
