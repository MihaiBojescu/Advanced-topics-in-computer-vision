import typing as t
from PIL.Image import Image
from keras import KerasTensor, ops
import keras


def grayscale_transform(img: Image) -> Image:
    return img.convert("L")


def to_tensor(img: Image) -> KerasTensor:
    tensor = ops.convert_to_tensor(img)
    return ops.expand_dims(tensor, axis=2)


class ImageResize:
    __size: tuple[int, int]

    def __init__(self, size: tuple[int, int]):
        self.__size = size

    def __call__(self, image):
        return keras.ops.image.resize(
            image, self.__size, interpolation="bilinear", antialias=True
        )


def normalise_tensor(tensor: KerasTensor) -> KerasTensor:
    return keras.ops.convert_to_tensor(
        keras.ops.convert_to_numpy(tensor).astype("float32") / 255.0
    )


def crop_images(image):
    shape = image.shape

    return keras.ops.image.crop_images(
        image,
        top_cropping=int(shape[1] / 7),
        bottom_cropping=int(shape[1] / 4),
        left_cropping=int(shape[0] / 3),
        right_cropping=int(shape[0] / 3),
    )


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
