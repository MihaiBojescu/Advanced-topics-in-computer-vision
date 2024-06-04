import keras
import time
from nn.model.loss import BoundedMeanSquaredError

class Model(keras.models.Model):
    _model: keras.Sequential

    def __init__(
        self,
        input_shape=(None, 256, 256, 1),
        optimizer=None,
        loss=None,
        metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if optimizer is None:
            optimizer = keras.optimizers.Adam(clipnorm=1)
        if loss is None:
            loss = BoundedMeanSquaredError(lower_bound=0, upper_bound=1, penalty=1000)
        if metrics is None:
            metrics = [keras.metrics.MeanAbsoluteError()]

        self._model = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=3,
                    kernel_size=(5, 5),
                    kernel_initializer="he_normal",
                    activation="relu",
                    input_shape=input_shape[1:],
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(5, 5),
                    kernel_initializer="he_normal",
                    activation="relu",
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(
                    filters=9,
                    kernel_size=(5, 5),
                    kernel_initializer="he_normal",
                    activation="relu",
                ),
                keras.layers.Conv2D(
                    filters=12,
                    kernel_size=(5, 5),
                    kernel_initializer="he_normal",
                    activation="relu",
                ),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalMaxPool2D(),
                keras.layers.Dense(units=2, kernel_initializer="he_normal"),
            ]
        )
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def fit(self, x, validation_data, epochs: int):
        output = self._model.fit(
            x=x,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)],
        )
        last_loss = output.history["loss"][-1]
        last_val_loss = output.history["val_loss"][-1]
        model_name = (
            f"model_epochs{epochs}_loss{last_loss:.4f}_val-loss{last_val_loss:.4f}"
        )

        self._model.save_weights(
            f"./outputs/{model_name}_{time.time_ns()}.weights.h5",
            overwrite=True,
        )

    def evaluate(self, x):
        results = self._model.evaluate(x=x)
        mean_absolute_error = results[1]

        print(f"Test loss           = {results[0]:.4f}")
        print(f"Test absolute error = {mean_absolute_error:.4f}")

    def load_weights(self, filepath: str):
        self._model.load_weights(filepath)

    def predict(self, x):
        return self._model.predict(x)
