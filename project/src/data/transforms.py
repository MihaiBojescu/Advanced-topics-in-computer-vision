from torch import zeros
from keras import KerasTensor
from keras.ops import convert_to_tensor


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
