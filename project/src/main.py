#!/usr/bin/env bash

from data.dataloader import ImageDataloader
from data.dataset import ImageDataset
from data.transforms import (
    grayscale_transform,
    image_resize,
    normalise_tensor,
    to_tensor,
)
from data.label_transforms import (
    label_to_ints,
    normalize_coordinates,
)
from nn.model.model import Model
from utils.args import parse_arguments


def main():
    args = parse_arguments()
    transforms = [grayscale_transform, to_tensor, image_resize, normalise_tensor]
    label_transforms = [label_to_ints]

    train_dataset = ImageDataset(
        data_path="./dataset",
        data_file_path="./train.csv",
        transforms=transforms,
        label_transforms=label_transforms,
    )
    validation_dataset = ImageDataset(
        data_path="./dataset",
        data_file_path="./validation.csv",
        transforms=transforms,
        label_transforms=label_transforms,
    )
    test_dataset = ImageDataset(
        data_path="./dataset",
        data_file_path="./test.csv",
        transforms=transforms,
        label_transforms=label_transforms,
    )

    train_dataloader = ImageDataloader(dataset=train_dataset, batch_size=256)
    validation_dataloader = ImageDataloader(dataset=validation_dataset, batch_size=256)
    test_dataloader = ImageDataloader(dataset=test_dataset, batch_size=256)

    model = Model()
    model.fit(train_dataloader)

    # normalized_labels = [normalize_coordinates(label, (img.width, img.height)) for img, label in zip(dataset.__data, dataset.__labels)]
    # dataset.__labels = normalized_labels

    # max_x = -1
    # max_y = -1

    # for pic, _ in train_dataset:
    #     max_x = max(pic.size[0], max_x)
    #     max_y = max(pic.size[1], max_y)
    # for pic, _ in validation_dataset:
    #     max_x = max(pic.size[0], max_x)
    #     max_y = max(pic.size[1], max_y)
    # for pic, _ in test_dataset:
    #     max_x = max(pic.size[0], max_x)
    #     max_y = max(pic.size[1], max_y)

    # print(max_x)
    # print(max_y)

    # normalized_labels = [
    #     normalize_coordinates(label, (img.width, img.height))
    #     for img, label in zip(dataset.__data, dataset.__labels)
    # ]
    # dataset.__labels = normalized_labels

    # TODO: implement the rest:
    #    1. add image transformations to enhance accuracy (ideas: remove background, extract eyes only)
    #    2. add image normalisations
    #    3. add neural network
    #    4. add client-side code for Windows/Linux/MacOS
    #    5. ...


if __name__ == "__main__":
    main()
