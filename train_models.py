import config
import train


if __name__ == '__main__':
    model_outputs = train.train(config.cfg, progressbar=True)
