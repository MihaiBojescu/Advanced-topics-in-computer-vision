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
    validation_dataloader = ImageDataloader(dataset=validation_dataset, batch_size=args.batch_size)
    test_dataloader = ImageDataloader(dataset=test_dataset, batch_size=args.batch_size)

    model = Model()
    model.fit(x=train_dataloader, validation_data=validation_dataloader, epochs=args.epochs)
    model.evaluate(x=test_dataloader)
    outputs = model.predict(x=test_dataloader)

    print(outputs)

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
