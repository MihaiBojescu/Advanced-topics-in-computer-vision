import typing as t
from PIL.Image import Image
from keras import KerasTensor, ops

def grayscale_transform(img: Image) -> Image:
    return img.convert('L')

def to_tensor(img: Image) -> KerasTensor:
    return ops.convert_to_tensor(img)

def normalise_tensor(tensor: KerasTensor) -> KerasTensor:
    max_val = ops.max(tensor)
    min_val = ops.min(tensor)

    return (tensor - min_val) / (max_val - min_val)

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

        padded_tensor = ops.zeros(
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

        return ops.convert_to_tensor(x=padded_tensor, dtype=tensor.dtype)
