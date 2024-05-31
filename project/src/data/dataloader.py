import typing as t
import keras
import numpy as np
from data.common import L, T, U
from data.dataset import ImageDataset


class ImageDataloader(keras.utils.Sequence):
    _batch_size: int
    _dataset: ImageDataset
    _dataset_indices: np.ndarray[t.Literal["N"], int]

    _current_loaded_batch: int
    _data: np.ndarray[t.Literal["N"], T]
    _labels: np.ndarray[t.Literal["N"], L]

    def __init__(
        self, *args, dataset: ImageDataset, batch_size: int, shuffle: bool = False
    ):
        super().__init__(*args, use_multiprocessing=True, workers=8)

        self._dataset = dataset
        self._dataset_indices = np.arange(len(dataset))
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._current_loaded_batch_indices = []

        if self._shuffle:
            np.random.shuffle(self._dataset_indices)

    def __load_data(
        self, index: int
    ) -> t.Tuple[np.ndarray[t.Literal["N"], T], np.ndarray[t.Literal["N"], L], list[int]]:
        batch_start_row_index = index // self._batch_size * self._batch_size
        batch_end_row_index = batch_start_row_index + self._batch_size
        batch_end_row_index = min(batch_end_row_index, len(self._dataset))

        difference = batch_end_row_index - batch_start_row_index
        difference = max(difference, 0)

        batch_indices = self._dataset_indices[batch_start_row_index:batch_start_row_index + difference]

        first_entry = self._dataset[batch_indices[0]]

        data = np.empty((difference, *first_entry[0].shape), dtype=np.float32)
        labels = np.empty((difference, *first_entry[1].shape), dtype=np.int32)

        for i in range(1, self._batch_size):
            value = self._dataset[batch_indices[i]]
            data[i][:, :, :] = value[0]
            labels[i][:] = value[1]

        return data, labels, batch_indices

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index not in self._current_loaded_batch_indices:
            self._data, self._labels, self._current_loaded_batch_indices = (
                self.__load_data(index=index)
            )

        return self._data, self._labels
