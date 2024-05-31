import typing as t
import numpy as np
from data.common import L, T, U
from data.dataset import ImageDataset


class ImageDataloader(t.Generic[T, L]):
    _batch_size: int
    _dataset: ImageDataset
    _dataset_indices: np.array[int]

    _current_loaded_batch: int
    _data: np.ndarray[T, np.dtype]
    _labels: np.ndarray[L, np.dtype]

    def __init__(
        self,
        *args,
        dataset: ImageDataset,
        batch_size: int,
        shuffle: bool = False
    ):
        super().__init__(*args)

        self._dataset = dataset
        self._dataset_indices = np.array(range(len(dataset)))
        self._dataset_indices = np.random.shuffle(self._dataset_indices) if shuffle else self._dataset_indices
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._current_loaded_batch = None

    def __load_data(
        self, index: int
    ) -> t.Tuple[np.ndarray[T, np.dtype], np.ndarray[L, np.dtype]]:
        batch_start_row_index = index * self._batch_size
        batch_end_row_index = batch_start_row_index + self._batch_size
        batch_end_row_index = min(batch_end_row_index, len(self._dataset))

        difference = batch_end_row_index - batch_start_row_index
        difference = max(difference, 0)

        data = np.empty(difference, dtype=object)
        labels = np.empty(difference, dtype=object)

        for i in range(batch_start_row_index, batch_end_row_index):
            data[i % self._batch_size], labels[i % self._batch_size] = self._dataset[self._dataset_indices[i]]

        return data, labels

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self._current_loaded_batch:
            self._data, self._labels = self.__load_data(index=index)
            self._current_loaded_batch = index

        return self._data, self._labels
