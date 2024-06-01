#!/usr/bin/env python3

import csv
import multiprocessing
import os
import typing as t
import keras
import numpy as np
from data.transforms import (
    ImageResize,
    grayscale_transform,
    normalise_tensor,
    to_tensor,
)

args = {
    "csv_source_files": ["test.csv", "train.csv", "validation.csv"],
    "transforms": [
        grayscale_transform,
        to_tensor,
        ImageResize(size=(256, 256)),
        normalise_tensor,
    ],
    "output_dir": "256x256",
}


def main():
    make_output_dir(output_dir=args["output_dir"])

    photo_paths = get_photo_paths(csv_source_files=args["csv_source_files"])
    transform_photos(
        photo_paths=photo_paths,
        transforms=args["transforms"],
        output_dir=args["output_dir"],
    )
    copy_csv_source_files(
        csv_source_files=args["csv_source_files"], output_dir=args["output_dir"]
    )


def get_photo_paths(csv_source_files: t.List[str]) -> t.List[str]:
    print("Reading CSV files for photos")

    photo_files = []

    for csv_source_file in csv_source_files:
        with open(f"./dataset/original/{csv_source_file}") as csv_file:
            csv_reader = csv.reader(csv_file)

            next(csv_reader)

            for row in csv_reader:
                photo_files.append(row[0])

            print(f"\tRead ./dataset/original/{csv_source_file}")

    return photo_files


def make_output_dir(output_dir: str):
    print("Making output directory")

    try:
        os.makedirs(f"./dataset/{output_dir}")
        os.makedirs(f"./dataset/{output_dir}/images")
    except:
        pass


def transform_photos(
    photo_paths: t.List[str],
    transforms: t.List[t.Callable[[any], keras.KerasTensor]],
    output_dir: str,
):
    print("Transforming photo files...")

    total = len(photo_paths)
    pool_args = [
        (index, total, photo_path, transforms, output_dir)
        for index, photo_path in enumerate(photo_paths)
    ]

    with multiprocessing.Pool(10) as pool:
        pool.starmap(transform_photo, pool_args)


def transform_photo(
    index: int,
    total: int,
    photo_path: str,
    transforms: t.List[t.Callable[[any], keras.KerasTensor]],
    output_dir: str,
):
    photo = keras.utils.load_img(f"./dataset/original/images/{photo_path}")

    for transform in transforms:
        photo = transform(photo)

    photo = keras.ops.convert_to_numpy(photo)
    np.save(f"./dataset/{output_dir}/images/{photo_path}", photo)

    print(f"\tTransformed ./dataset/{output_dir}/{photo_path} ({index + 1}/{total})")


def copy_csv_source_files(csv_source_files: t.List[str], output_dir: str):
    print("Copying CSV files")

    total = len(csv_source_files)

    for index, csv_source_file in enumerate(csv_source_files):
        with open(f"./dataset/original/{csv_source_file}") as csv_input_file:
            with open(
                f"./dataset/{output_dir}/{csv_source_file}", "w"
            ) as csv_output_file:
                csv_reader = csv.reader(csv_input_file)
                csv_writer = csv.writer(csv_output_file)

                csv_writer.writerow(next(csv_reader))

                for row in csv_reader:
                    row[0] = f"{row[0]}.npy"
                    csv_writer.writerow(row)

                print(
                    f"\tCopied ./dataset/{output_dir}/{csv_source_file} ({index + 1}/{total})"
                )


if __name__ == "__main__":
    main()
