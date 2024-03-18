import numpy as np
import imageio.v3 as iio

class BaseCamera:
    def __next__(self):
        return np.array([])
    
    def __iter__(self):
        return self

class ImageIOCamera(BaseCamera):
    def __init__(self):
        self._reader = iio.imiter("<video0>")

    def __next__(self):
        return next(self._reader)

    def __iter__(self):
        return self
