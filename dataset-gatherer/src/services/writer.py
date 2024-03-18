import os
import csv
import time
import numpy as np
import imageio.v3 as iio


class Writer:
    _path: str
    _image_filename_slug: str

    def __init__(self, path: str, image_filename_slug: str) -> None:
        self._path = path
        self._image_filename_slug = image_filename_slug

    def run(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        resolution_x: int,
        resolution_y: int,
        width: int,
        height: int,
        distance: float,
    ):
        now = time.time()
        filename = f"{self._image_filename_slug}{now}.png"

        if not os.path.isdir(self._path):
            os.makedirs(self._path, exist_ok=True)
            os.makedirs(f"{self._path}/images", exist_ok=True)

        iio.imwrite(f"{self._path}/images/{filename}", frame)

        with open(f"{self._path}/dataset.csv", mode="a+", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    filename,
                    x,
                    y,
                    resolution_x,
                    resolution_y,
                    width,
                    height,
                    distance,
                )
            )
