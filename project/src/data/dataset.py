import csv
import typing as t
from dataclasses import dataclass
from keras.preprocessing.image import load_img


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


class ImageDataset(t.Generic[T, U]):
    __transforms: t.Optional[t.Callable[[T], U]]
    __label_transforms: t.Optional[t.Callable[[T], U]]
    __data: t.List[T]
    __labels: t.List[L]
    __metas: t.List[Metadata]

    def __init__(
        self,
        *args,
        data_path: str,
        data_file: t.Optional[str],
        transforms: t.Optional[t.Callable[[T], U]] = None,
        label_transforms: t.Optional[t.Callable[[L], L]] = None,
    ):
        super().__init__(*args)

        self.__transforms = transforms
        self.__label_transforms = label_transforms
        self.__data, self.__labels, self.__metas = self.__load_data(
            data_path=data_path, data_file=data_file
        )

    def __load_data(
        self, data_path: str, data_file: t.Optional[str]
    ) -> t.Tuple[t.List[T], t.List[L], t.List[Metadata]]:
        data: t.List[T] = []
        labels: t.List[L] = []
        metas: t.List[Metadata] = []
        data_file = data_file if data_file is not None else "train.csv"

        with open(f"{data_path}/{data_file}") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                filename = row[0]
                x = int(float(row[1]))
                y = int(float(row[2]))
                width = int(row[3])
                height = int(row[4])
                distance = float(row[5])

                labels.append(Label(x, y))
                metas.append(Metadata(filename, x, y, width, height, distance))

        for entry in metas:
            image = load_img(f"{data_path}/images/{entry.filename}")
            data.append(image)

        return data, labels, metas

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> t.Tuple[U, L]:
        data = self.__data[index]
        label = self.__labels[index]

        if self.__transforms:
            for transform in self.__transforms:
                data = transform(data)

        if self.__label_transforms:
            for transform in self.__transforms:
                label = transform(label)

        return data, label
