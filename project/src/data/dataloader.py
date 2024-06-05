import typing as t
import keras
import numpy as np
from data.common import L, T, U
from data.dataset import TensorDataset
from PIL import Image
from data.transforms import (
    grayscale_transform,
    to_tensor,
    ImageResize,
    normalise_tensor,
)

class ImageDataloader(keras.utils.PyDataset):
    _batch_size: int
    _dataset: TensorDataset
    _dataset_indices: np.ndarray[t.Literal["N"], int]

    _currently_loaded_batch_index: t.Optional[int]
    _data: np.ndarray[t.Literal["N"], T]
    _labels: np.ndarray[t.Literal["N"], L]

    def __init__(
        self, *args, dataset: TensorDataset, batch_size: int, shuffle: bool = False
    ):
        super().__init__(*args, use_multiprocessing=True, workers=8)

        self._dataset = dataset
        self._dataset_indices = np.arange(len(dataset))
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._currently_loaded_batch_index = None

        if self._shuffle:
            np.random.shuffle(self._dataset_indices)

    def __load_data(
        self, index: int
    ) -> t.Tuple[
        np.ndarray[t.Literal["N"], T], np.ndarray[t.Literal["N"], L], list[int]
    ]:
        batch_start_row_index = index * self._batch_size
        batch_end_row_index = batch_start_row_index + self._batch_size
        batch_end_row_index = min(batch_end_row_index, len(self._dataset))

        difference = batch_end_row_index - batch_start_row_index
        difference = max(difference, 0)

        batch_indices = self._dataset_indices[
            batch_start_row_index : batch_start_row_index + difference
        ]

        first_entry = self._dataset[batch_indices[0]]

        data = np.empty((difference, *first_entry[0].shape), dtype=np.float32)
        labels = np.empty((difference, *first_entry[1].shape), dtype=np.int32)

        data[0][:, :, :] = first_entry[0]
        labels[0][:] = first_entry[1]

        for i in range(1, difference):
            value = self._dataset[batch_indices[i]]
            data[i][:, :, :] = value[0]
            labels[i][:] = value[1]

        return data, labels

    def __len__(self) -> int:
        return len(self._dataset) // self._batch_size + (
            1 if len(self._dataset) % self._batch_size > 0 else 0
        )

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self._currently_loaded_batch_index:
            self._data, self._labels = self.__load_data(index=index)
            self._currently_loaded_batch_index = index

        return self._data, self._labels
    
    def on_epoch_end(self):
        np.random.shuffle(self._dataset_indices)
        

class SingleImageDataLoader(keras.utils.PyDataset):
    def __init__(self, image_file):
        self.image_file = image_file
        self.transforms = [
            grayscale_transform,
            to_tensor,
            ImageResize(size=(128, 128)),
            normalise_tensor,
        ]

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        image = Image.open(self.image_file)
        for transform in self.transforms:
            image = transform(image)
        image_array = np.expand_dims(image, axis=0)
        return image_array