import typing as t
import keras
from PIL import Image
import numpy as np

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