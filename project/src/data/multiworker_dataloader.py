from multiprocessing import Pool
import typing as t
import numpy as np
from copy import copy
from data.common import L, T, U, Label
from data.dataloader import ImageDataloader
from data.dataset import ImageDataset


class MultiworkerImageLoaderMapper(t.Generic[T, L]):
    __dataset: ImageDataset
    __dataset_indices: np.array[int]

    def __init__(self, dataset: ImageDataset, dataset_indices: np.array[int]):
        self.__dataset = copy(dataset)
        self.__dataset_indices = dataset_indices

    def __call__(self, index_batch: t.List[int]) -> t.List[t.Tuple[T, L]]:
        return [self.__dataset[self.__dataset_indices[i]] for i in index_batch]


class MultiworkerImageLoader(ImageDataloader[T, L]):
    __work_per_worker: int
    __pool: Pool
    __mapper: MultiworkerImageLoaderMapper[T, L]

    def __init__(
        self,
        *args,
        dataset: ImageDataset,
        batch_size: int,
        n_workers: int,
        shuffle: bool = False
    ):
        super().__init__(*args, dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        self.__work_per_worker = batch_size // n_workers
        self.__pool = Pool(n_workers)
        self.__mapper = MultiworkerImageLoaderMapper(
            dataset=dataset, dataset_indices=self._dataset_indices
        )

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

        index_batches = [
            list(range(i, i + self.__work_per_worker))
            for i in range(
                batch_start_row_index,
                batch_end_row_index,
                self.__work_per_worker,
            )
        ]
        batches = self.__pool.map(self.__mapper, index_batches)

        for i, batch in enumerate(batches):
            for j, entry in enumerate(batch):
                data_index = i * self.__work_per_worker + j
                data[data_index], labels[data_index] = entry

        return data, labels

    def __getitem__(self, index) -> t.Tuple[U, L]:
        if index != self._current_loaded_batch:
            self._data, self._labels = self.__load_data(index=index)
            self._current_loaded_batch = index

        return self._data, self._labels
