from dataclasses import dataclass
import os
import csv
import typing as t
import keras
from keras.preprocessing.image import load_img
from PIL import Image


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
    width: int
    height: int


class ImageDataset:
    __transforms: t.Optional[t.Callable[[keras.KerasTensor], keras.KerasTensor]]
    __data: t.List[keras.KerasTensor]
    __labels: t.List[Label]
    __metas: t.List[Metadata]

    def __init__(
        self,
        *args,
        data_path: str,
        transforms: t.Optional[
            t.Callable[[keras.KerasTensor], keras.KerasTensor]
        ] = None,
    ):
        super().__init__(*args)

        self.__transforms = transforms
        self.__data, self.__labels, self.__metas = self.__load_data(data_path)

    def __load_data(
        self, data_path: str
    ) -> t.Tuple[t.List[keras.KerasTensor], t.List[Label], t.List[Metadata]]:
        data: t.List[t.Tuple[str, keras.KerasTensor]] = []
        labels: t.List[Label] = []
        metas: t.List[Metadata] = []

        with open(f"{data_path}/dataset.csv") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                filename = row[0]
                x = int(row[1])
                y = int(row[2])
                width = int(row[3])
                height = int(row[4])
                distance = float(row[5])

                labels.append(Label(x, y, width, height))
                metas.append(Metadata(filename, x, y, width, height, distance))

        for entry in metas:
            tensor = load_img(f"{data_path}/images/{entry.filename}")
            data.append(tensor)

        return data, labels, metas

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> t.Tuple[keras.KerasTensor, Label]:
        tensor = self.__data[index]
        label = self.__labels[index]

        if self.__transforms:
            tensor = self.__transforms(tensor)

        return tensor, label
