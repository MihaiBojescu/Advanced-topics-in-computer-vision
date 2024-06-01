import keras
import time


class Model(keras.models.Model):
    _model: keras.models.Model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = keras.Sequential(
            [
                keras.layers.Conv2D(filters=3, kernel_size=(5, 5)),
                keras.layers.ReLU(),
                keras.layers.Conv2D(filters=6, kernel_size=(5, 5)),
                keras.layers.ReLU(),
                keras.layers.Conv2D(filters=9, kernel_size=(5, 5)),
                keras.layers.ReLU(),
                keras.layers.Conv2D(filters=12, kernel_size=(5, 5)),
                keras.layers.ReLU(),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(units=2),
            ]
        )
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredLogarithmicError(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )

    def fit(self, dataset):
        output = self._model.fit(
            dataset,
            epochs=20,
            callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
        )
        last_loss = output.history["loss"][-1]

        self._model.save(f"./outputs/model_{time.time_ns()}_loss_{last_loss:.4f}.keras")

    def predict(self, x: keras.KerasTensor):
        return self._model.predict(x)
