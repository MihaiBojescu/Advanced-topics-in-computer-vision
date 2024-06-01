import keras
import time
import typing as t


class Model(keras.models.Model):
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
    _model: keras.models.Model

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
            loss=keras.losses.MeanSquaredLogarithmicError(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )

    def fit(self, dataset):
        output = self._model.fit(dataset, epochs=20)
        last_loss = output.history['loss'][-1]

        self._model.save(f'./outputs/model_{time.time_ns()}_loss_{last_loss:.4f}.keras')

    def predict(self, x: keras.KerasTensor):
        return self._model.predict(x)
