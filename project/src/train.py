#!/usr/bin/env python3

from data.dataloader import ImageDataloader
from data.dataset import TensorDataset
from data.label_transforms import (
    label_to_ints,
)
from nn.model.model import Model
from utils.args import parse_arguments


def main():
    args = parse_arguments()
    label_transforms = [label_to_ints]

    train_dataset = TensorDataset(
        data_path="./dataset/256x256",
        data_file_path="./train.csv",
        label_transforms=label_transforms,
    )
    validation_dataset = TensorDataset(
        data_path="./dataset/256x256",
        data_file_path="./validation.csv",
        label_transforms=label_transforms,
    )
    test_dataset = TensorDataset(
        data_path="./dataset/256x256",
        data_file_path="./test.csv",
        label_transforms=label_transforms,
    )

    train_dataloader = ImageDataloader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_dataloader = ImageDataloader(
        dataset=validation_dataset, batch_size=args.batch_size
    )
    test_dataloader = ImageDataloader(dataset=test_dataset, batch_size=args.batch_size)

    model = Model()
    model.fit(
        x=train_dataloader, validation_data=validation_dataloader, epochs=args.epochs
    )
    model.evaluate(x=test_dataloader)
    outputs = model.predict(x=test_dataloader)

    print(outputs)


if __name__ == "__main__":
    main()
