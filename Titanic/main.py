from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import DataLoader
import torch

import torchdata.datapipes as dp

import wandb


def load_data() -> Tuple[Union[DataLoader, None], Union[DataLoader, None], Union[DataLoader, None]]:
    """Load the data in `wandb.config.data_path` and return the converted results to the `DataLoader` format.

    Returns:
        Tuple[Union[DataLoader, None], Union[DataLoader, None], Union[DataLoader, None]]: three `Dataloader`s for train/dev/test
    """

    train_dl = None
    dev_dl = None
    test_dl = None

    def build_datapipe(data_path: Path):
        datapipe = dp.iter.FileOpener([str(data_path)], mode='rt')
        datapipe = dp.iter.CSVParser(datapipe, skip_lines=1, delimiter=',')

        def row_processor(row):
            label = row[0]
            data = row[1:]
            # todo: convert data to torch types in order to put them in models right away
            return {"label": label, "data": data}
        datapipe = dp.iter.Mapper(datapipe, row_processor)
        return datapipe

    if wandb.config.is_train:
        datapipe = build_datapipe(Path(wandb.config.data_path) / "train.csv")
        train_datapipe, dev_datapipe = dp.iter.RandomSplitter(datapipe,
                                                              weights=wandb.config.train_dev_ratio,
                                                              seed=0,
                                                              total_length=wandb.config.train_file_len)

        train_dl = DataLoader(
            train_datapipe, batch_size=wandb.config.batch_size, shuffle=True)
        dev_dl = DataLoader(
            dev_datapipe, batch_size=wandb.config.batch_size, shuffle=False)

    if wandb.config.is_test:
        test_datapipe = build_datapipe(
            Path(wandb.config.data_path) / "test.csv")
        test_dl = DataLoader(
            test_datapipe, batch_size=wandb.config.batch_size, shuffle=False)

    return train_dl, dev_dl, test_dl


def train(train_data: DataLoader, dev_data: DataLoader):
    # train model with `train_data` and `dev_data` and save it in `model_path`
    
    raise NotImplementedError


def test(test_data: DataLoader):
    # test the model in `model_path` using `test_data`
    raise NotImplementedError


def main():
    train_dataloader, dev_dataloader, test_dataloader = load_data()

    if wandb.config.is_train:
        train(train_dataloader, dev_dataloader)

    if wandb.config.is_test:
        test(test_dataloader)


if __name__ == '__main__':
    wandb.init(project="kaggle-Titanic")
    main()
