import os
import typing as t
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage
from PIL import Image


class ImageDataset(Dataset):
    __transforms: t.Optional[t.Callable[[Tensor], Tensor]]
    __data: t.List[t.Tuple[Tensor, str]]

    def __init__(
        self,
        *args,
        data_path: str,
        transforms: t.Optional[t.Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__(*args)

        self.__transforms = transforms
        self.__data = self.__load_data(data_path)

    def __load_data(self, data_path: str) -> t.List[Tensor]:
        data: t.List[Tensor] = []
        transform = ToImage()
        files = os.listdir(data_path)

        for file in files:
            image = Image.open(f"{data_path}/{file}")
            tensor = transform(image)
            data.append((tensor, os.path.splitext(file)[0]))

        return data

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> Tensor:
        tensor, _ = self.__data[index]

        if self.__transforms:
            tensor = self.__transforms(tensor)

        return tensor
