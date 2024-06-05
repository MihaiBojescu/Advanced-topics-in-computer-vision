import typing as t
import keras
from data.common import Label


def label_to_tensor(label: Label):
    return keras.ops.convert_to_tensor([label.x, label.y])

def label_scaling(label):
    return label * 2 - 1


def normalize_coordinates(
    label: t.Tuple[int, int, int, int], img_size: t.Tuple[int, int]
) -> t.Tuple[float, float, float, float]:
    x, y, width, height = label
    max_x, max_y = img_size

    normalized_x = x / max_x
    normalized_y = y / max_y
    normalized_width = width / max_x
    normalized_height = height / max_y

    return normalized_x, normalized_y, normalized_width, normalized_height
