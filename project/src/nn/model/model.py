import keras
import typing as t


class Model(keras.Model):
    _conv1: keras.layers.Conv2D
    _conv1_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _conv2: keras.layers.Conv2D
    _conv2_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _conv3: keras.layers.Conv2D
    _conv3_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _conv4: keras.layers.Conv2D
    _conv4_activation: t.Callable[[keras.KerasTensor], keras.KerasTensor]
    _output_pooling: keras.layers.GlobalAveragePooling2D
    _output: keras.layers.Dense
    _model: keras.Model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._conv1 = keras.layers.Conv2D(filters=3, kernel_size=(5, 5))
        self._conv1_activation = keras.layers.ReLU()
        self._conv2 = keras.layers.Conv2D(filters=6, kernel_size=(5, 5))
        self._conv2_activation = keras.layers.ReLU()
        self._conv3 = keras.layers.Conv2D(filters=9, kernel_size=(5, 5))
        self._conv3_activation = keras.layers.ReLU()
        self._conv4 = keras.layers.Conv2D(filters=12, kernel_size=(5, 5))
        self._conv4_activation = keras.layers.ReLU()
        self._output_pooling = keras.layers.GlobalAveragePooling2D()
        self._output = keras.layers.Dense(units=2)

        self._model = keras.Sequential(
            [
                self._conv1,
                self._conv2_activation,
                self._conv2,
                self._conv2_activation,
                self._conv3,
                self._conv3_activation,
                self._conv4,
                self._conv4_activation,
                self._output_pooling,
                self._output
            ]
        )
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )

    def fit(self, dataset):
        return self._model.fit(dataset, epochs=5)

    def predict(self, x: keras.KerasTensor):
        return self._model.predict(x)
