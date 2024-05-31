import csv
import typing as t
from keras.preprocessing.image import load_img
from data.common import Metadata, T, U, L, Label


class ImageDataset(t.Generic[T, U]):
    __data_path: str
    __data_file_path: str
    __transforms: t.Optional[t.Callable[[T], U]]
    __label_transforms: t.Optional[t.Callable[[T], U]]

    __current_loaded_index: int
    __data_file_rows: t.List[t.Tuple]
    __data: T
    __label: L
    __meta: Metadata

    def __init__(
        self,
        *args,
        data_path: str,
        data_file_path: t.Optional[str],
        transforms: t.Optional[t.Callable[[T], U]] = None,
        label_transforms: t.Optional[t.Callable[[L], L]] = None,
    ):
        super().__init__(*args)

        self.__data_path = data_path
        self.__data_file_path = (
            data_file_path if data_file_path is not None else "train.csv"
        )
        self.__transforms = transforms
        self.__label_transforms = label_transforms

        self.__data = None
        self.__label = None
        self.__meta = None
        self.__current_loaded_index = None
        self.__data_file_rows = self.__read_data_file()

    def __read_data_file(self):
        rows = []

        with open(f"{self.__data_path}/{self.__data_file_path}") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                rows.append(row)

        return rows

    def __load_data(
        self, index: int
    ) -> t.Tuple[t.List[T], t.List[L], t.List[Metadata]]:
        row = self.__data_file_rows[index]

        filename = row[0]
        x = int(float(row[1]))
        y = int(float(row[2]))
        width = int(row[3])
        height = int(row[4])
        distance = float(row[5])

        data = load_img(f"{self.__data_path}/images/{filename}")
        label = Label(x, y)
        meta = Metadata(filename, x, y, width, height, distance)

        return data, label, meta

    def __len__(self) -> int:
        return len(self.__data_file_rows)

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self.__current_loaded_index:
            self.__data, self.__label, self.__meta = self.__load_data(index=index)
            self.__current_loaded_index = index

        data = self.__data
        label = self.__label

        if self.__transforms:
            for transform in self.__transforms:
                data = transform(data)

        if self.__label_transforms:
            for transform in self.__label_transforms:
                label = transform(label)

        return data, label


