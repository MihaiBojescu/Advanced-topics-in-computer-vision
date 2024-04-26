#!/usr/bin/env bash

import os

os.environ["KERAS_BACKEND"] = "torch"

import typing as t
import keras

from data.dataset import ImageDataset
from data.transforms import grayscale_transform, normalize_coordinates

from utils.args import parse_arguments



def main():
    args = parse_arguments()
    transforms: t.Callable[[keras.KerasTensor], keras.KerasTensor] = [grayscale_transform]

    dataset = ImageDataset(data_path="./dataset", transforms=transforms)

    normalized_labels = [normalize_coordinates(label, (img.width, img.height)) for img, label in zip(dataset.__data, dataset.__labels)]
    dataset.__labels = normalized_labels


    # TODO: implement the rest:
    #    1. add image transformations to enhance accuracy (ideas: remove background, extract eyes only)
    #    2. add image normalisations
    #    3. add neural network
    #    4. add client-side code for Windows/Linux/MacOS
    #    5. ...

if __name__ == "__main__":
    main()
