import argparse
from pathlib import Path
import wandb
import torchdata.datapipes as dp


def load_data():
    raise NotImplementedError


def train():
    raise NotImplementedError


def test():
    raise NotImplementedError


def main():
    train_data, dev_data, test_data = load_data()
    if args.train:
        train(train_data, dev_data)

    if args.test:
        test(test_data)


if __name__ == '__main__':
    wandb.init(project="kaggle-Titanic")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_data', type=Path, default='data/train.csv')
    parser.add_argument('--test_data', type=Path, default='data/test.csv')
    parser.add_argument('--models', type=Path, default='models/')
    args = parser.parse_args()

    main()
