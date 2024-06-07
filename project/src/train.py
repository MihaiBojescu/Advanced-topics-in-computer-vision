#!/usr/bin/env python3

import utils.prerequisites
from torch.utils.data import DataLoader
from data.dataloader import ImageDataloader
from data.dataset import TensorDataset
from data.label_transforms import (
    label_scaling,
    label_to_tensor,
)
from nn.model.model import Model
from utils.args import parse_arguments


def main():
    args = parse_arguments()
    label_transforms = [label_to_tensor]

    train_dataset = TensorDataset(
        data_path="./dataset/128x128-augmented-1",
        data_file_path="./train.csv",
        label_transforms=label_transforms,
    )
    validation_dataset = TensorDataset(
        data_path="./dataset/128x128-augmented-1",
        data_file_path="./validation.csv",
        label_transforms=label_transforms,
    )
    test_dataset = TensorDataset(
        data_path="./dataset/128x128-augmented-1",
        data_file_path="./test.csv",
        label_transforms=label_transforms,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset, batch_size=args.batch_size, num_workers=4
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, num_workers=4
    )

    model = Model(input_shape=(None, 128, 128, 1))
    model.fit(
        x=train_dataloader, validation_data=validation_dataloader, epochs=args.epochs
    )
    model.evaluate(x=test_dataloader)
    outputs = model.predict(x=test_dataloader)

    print(outputs)


if __name__ == "__main__":
    main()
