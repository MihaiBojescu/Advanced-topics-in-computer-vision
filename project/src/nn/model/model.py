import keras
import typing as t


class Model(keras.Model):
    _conv1: keras.layers.Conv2D
    _conv1_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _conv2: keras.layers.Conv2D
    _conv2_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _conv3: keras.layers.Conv2D
    _conv3_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Adjust layers, parameter sizes

        self._conv1 = keras.layers.Conv2D(filters=9, kernel_size=(3, 3))
        self._conv1_activation = keras.layers.ReLU()
        self._conv2 = keras.layers.Conv2D(filters=27, kernel_size=(3, 3))
        self._conv2_activation = keras.layers.ReLU()
        self._conv3 = keras.layers.Conv2D(filters=1, kernel_size=(3, 3))
        self._conv3_activation = keras.layers.ReLU()

    def call(self, x: keras.KerasTensor):
        x = self._conv1_activation(self._conv1(x))
        x = self._conv2_activation(self._conv2(x))
        x = self._conv3_activation(self._conv3(x))

        return x
