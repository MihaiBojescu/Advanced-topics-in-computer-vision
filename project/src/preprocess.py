#!/usr/bin/env python3

import copy
import csv
import ctypes
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
    "training_csv_file": "train.csv",
    "validation_csv_file": "validation.csv",
    "test_csv_file": "test.csv",
    "copies": 8,
    "preprocessing": [grayscale_transform, to_tensor, ImageResize(size=(256, 256))],
    "processing": [
        keras.layers.RandomFlip(),
        keras.layers.RandomTranslation(height_factor=0.5, width_factor=0.5),
        keras.layers.RandomRotation(factor=0.5),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        keras.layers.RandomContrast(factor=0.3),
        keras.layers.RandomBrightness(factor=0.3),
    ],
    "postprocessing": [
        normalise_tensor,
    ],
    "input_dir": "original",
    "output_dir": "256x256",
}


def main():
    make_output_dir(output_dir=args["output_dir"])

    training_csv_header, training_csv_rows = read_input_csv(
        input_dir=args["input_dir"], csv_file=args["training_csv_file"]
    )
    validation_csv_header, validation_csv_rows = read_input_csv(
        input_dir=args["input_dir"], csv_file=args["validation_csv_file"]
    )
    test_csv_header, test_csv_rows = read_input_csv(
        input_dir=args["input_dir"], csv_file=args["test_csv_file"]
    )

    processed_training_photo_paths_batches = transform_photos(
        rows=training_csv_rows,
        copies=args["copies"],
        preprocessing=args["preprocessing"],
        processing=args["processing"],
        postprocessing=args["postprocessing"],
        output_dir=args["output_dir"],
    )
    processed_validation_photo_paths_batches = transform_photos(
        rows=validation_csv_rows,
        copies=1,
        preprocessing=args["preprocessing"],
        processing=[],
        postprocessing=args["postprocessing"],
        output_dir=args["output_dir"],
    )
    processed_test_photo_paths_batches = transform_photos(
        rows=test_csv_rows,
        copies=1,
        preprocessing=args["preprocessing"],
        processing=[],
        postprocessing=args["postprocessing"],
        output_dir=args["output_dir"],
    )

    write_output_csv(
        header=training_csv_header,
        rows=training_csv_rows,
        processed_photo_paths_batches=processed_training_photo_paths_batches,
        output_dir=args["output_dir"],
        csv_file=args["training_csv_file"],
    )
    write_output_csv(
        header=validation_csv_header,
        rows=validation_csv_rows,
        processed_photo_paths_batches=processed_validation_photo_paths_batches,
        output_dir=args["output_dir"],
        csv_file=args["validation_csv_file"],
    )
    write_output_csv(
        header=test_csv_header,
        rows=test_csv_rows,
        processed_photo_paths_batches=processed_test_photo_paths_batches,
        output_dir=args["output_dir"],
        csv_file=args["test_csv_file"],
    )


def read_input_csv(input_dir: str, csv_file: str) -> t.Tuple[t.Tuple, t.List[t.Tuple]]:
    print(f"Reading input CSV file {csv_file}")

    header = ()
    rows = []

    with open(f"./dataset/{input_dir}/{csv_file}") as file:
        csv_reader = csv.reader(file)

        header = next(csv_reader)

        for row in csv_reader:
            rows.append(row)

        print(f"\tRead ./dataset/{input_dir}/{csv_file}")

    return header, rows


def make_output_dir(output_dir: str):
    print("Making output directory")

    try:
        os.makedirs(f"./dataset/{output_dir}")
        os.makedirs(f"./dataset/{output_dir}/images")
    except:
        pass


def transform_photos(
    rows: t.List[t.Tuple],
    copies: int,
    preprocessing: t.List[t.Callable[[any], any]],
    processing: t.List[t.Callable[[any], any]],
    postprocessing: t.List[t.Callable[[any], any]],
    output_dir: str,
):
    print("Transforming photo files...")

    total = len(rows)
    index = multiprocessing.Value(ctypes.c_uint, 0)
    pool_args = [
        (
            total,
            row,
            copies,
            preprocessing,
            processing,
            postprocessing,
            output_dir,
        )
        for row in rows
    ]
    outputs = []

    with multiprocessing.Pool(10, initializer=pool_init, initargs=(index,)) as pool:
        outputs = pool.starmap(transform_photo, pool_args)

    return outputs


def pool_init(shared_index):
    global global_shared_index
    global_shared_index = shared_index


def transform_photo(
    total: int,
    row: t.Tuple,
    copies: int,
    preprocessing: t.List[t.Callable[[any], any]],
    processing: t.List[t.Callable[[any], any]],
    postprocessing: t.List[t.Callable[[any], any]],
    output_dir: str,
):
    photo = keras.utils.load_img(f"./dataset/original/images/{row[0]}")
    photo_name = ".".join(row[0].split(".")[:-1])
    photo_extension = row[0].split(".")[-1]
    outputs: list[str] = []

    for transform in preprocessing:
        photo = transform(photo)

    for copy_index in range(copies):
        photo_copy = copy.copy(photo)

        for transform in processing:
            photo_copy = transform(photo_copy)

        for transform in postprocessing:
            photo_copy = transform(photo_copy)

        photo_copy = keras.ops.convert_to_numpy(photo_copy)
        np.save(
            f"./dataset/{output_dir}/images/{photo_name}-{copy_index}.{photo_extension}",
            photo,
        )
        outputs.append(f"{photo_name}-{copy_index}.{photo_extension}.npy")

    with global_shared_index.get_lock():
        global_shared_index.value += 1
        print(
            f"\tProcessed ./dataset/{output_dir}/{row[0]}, made {copies} copies ({global_shared_index.value}/{total})"
        )

    return outputs


def write_output_csv(
    header: t.Tuple,
    rows: t.List[t.Tuple],
    processed_photo_paths_batches: t.List[t.List[str]],
    output_dir: str,
    csv_file: str,
):
    print(f"Writing output CSV file {csv_file}")

    with open(f"./dataset/{output_dir}/{csv_file}", "w") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(header)

        for row, processed_photo_paths_batch in zip(
            rows, processed_photo_paths_batches
        ):
            for processed_photo_path in processed_photo_paths_batch:
                row_copy = copy.copy(row)
                row_copy[0] = processed_photo_path

                csv_writer.writerow(row_copy)

    print(f"\tWritten ./dataset/{output_dir}/{csv_file}")


if __name__ == "__main__":
    main()
