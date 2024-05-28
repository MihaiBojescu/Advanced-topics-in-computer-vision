import typing as t
import keras
from PIL import Image
import numpy as np
from keras import KerasTensor
from keras.ops import convert_to_tensor

def grayscale_transform(img: keras.KerasTensor) -> keras.KerasTensor:
    return img.convert('L')

def normalize_coordinates(label: t.Tuple[int, int, int, int], img_size: t.Tuple[int, int]) -> t.Tuple[float, float, float, float]:
    x, y, width, height = label
    max_x, max_y = img_size
    
    normalized_x = x / max_x
    normalized_y = y / max_y
    normalized_width = width / max_x
    normalized_height = height / max_y
    
    return normalized_x, normalized_y, normalized_width, normalized_height


class Padding:
    _max_x: int
    _max_y: int
    _pad_value: float

    def __init__(self, max_x: int, max_y: int, pad_value: float = 0):
        self._max_x = max_x
        self._max_y = max_y
        self._pad_value = pad_value

    def __call__(self, tensor: KerasTensor) -> KerasTensor:
        padding_x = int((self._max_x - tensor.shape[0]) / 2)
        padding_y = int((self._max_y - tensor.shape[1]) / 2)

        padded_tensor = zeros(
            (self._max_x, self._max_y, tensor.shape[2]),
            dtype=tensor.dtype,
            requires_grad=False,
        )
        padded_tensor[:, :, :] = self._pad_value
        padded_tensor[
            padding_x : (tensor.shape[0] + padding_x),
            padding_y : (tensor.shape[1] + padding_y),
            :,
        ] = tensor

        return convert_to_tensor(x=padded_tensor, dtype=tensor.dtype)
