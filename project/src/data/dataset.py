import csv
import typing as t
import keras
import numpy as np
from dataclasses import dataclass


@dataclass
class Metadata:
    filename: str
    x: int
    y: int
    width: int
    height: int
    distance: float


@dataclass
class Label:
    x: int
    y: int


T = t.TypeVar("T")
U = t.TypeVar("U")
L = t.TypeVar("L")


class TensorDataset(keras.utils.PyDataset):
    _data_path: str
    _data_file_path: str
    _label_transforms: t.Optional[t.Callable[[T], U]]

    _current_loaded_index: int
    _data_file_rows: t.List[t.Tuple]
    _data: T
    _label: L
    _meta: Metadata

    def __init__(
        self,
        *args,
        data_path: str,
        data_file_path: t.Optional[str],
        label_transforms: t.Optional[t.Callable[[L], L]] = None,
    ):
        super().__init__(*args, use_multiprocessing=True, workers=8)

        self._data_path = data_path
        self._data_file_path = (
            data_file_path if data_file_path is not None else "train.csv"
        )
        self._label_transforms = label_transforms

        self._data = None
        self._label = None
        self._meta = None
        self._current_loaded_index = None
        self._data_file_rows = self._read_data_file()

    def _read_data_file(self):
        rows = []

        with open(f"{self._data_path}/{self._data_file_path}") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                rows.append(row)

        return rows

    def _load_data(
        self, index: int
    ) -> t.Tuple[t.List[T], t.List[L], t.List[Metadata]]:
        row = self._data_file_rows[index]

        filename = row[0]
        x = float(row[-2])
        y = float(row[-1])
        width = int(row[3])
        height = int(row[4])
        distance = float(row[5])

        data = keras.ops.convert_to_tensor(np.load(f"{self._data_path}/images/{filename}"))
        label = Label(x, y)
        meta = Metadata(filename, x, y, width, height, distance)

        return data, label, meta

    def __len__(self) -> int:
        return len(self._data_file_rows)

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self._current_loaded_index:
            self._data, self._label, self._meta = self._load_data(index=index)
            self._current_loaded_index = index

        data = self._data
        label = self._label

        if self._label_transforms:
            for transform in self._label_transforms:
                label = transform(label)

        return data, label

class ImageDataset(TensorDataset):
    _transforms: t.Optional[t.Callable[[T], U]]
    _label_transforms: t.Optional[t.Callable[[T], U]]

    def _load_data(
        self, index: int
    ) -> t.Tuple[t.List[T], t.List[L], t.List[Metadata]]:
        row = self._data_file_rows[index]

        filename = row[0]
        x = int(float(row[1]))
        y = int(float(row[2]))
        width = int(row[3])
        height = int(row[4])
        distance = float(row[5])

        data = keras.utils.load_img(f"{self._data_path}/images/{filename}")
        label = Label(x, y)
        meta = Metadata(filename, x, y, width, height, distance)

        return data, label, meta

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self._current_loaded_index:
            self._data, self._label, self._meta = self._load_data(index=index)
            self._current_loaded_index = index

        data = self._data
        label = self._label

        if self._transforms:
            for transform in self._transforms:
                data = transform(data)

        if self._label_transforms:
            for transform in self._label_transforms:
                label = transform(label)

        return data, label
