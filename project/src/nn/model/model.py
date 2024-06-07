import keras
import time


class Model(keras.models.Model):
    _model: keras.Sequential

    def __init__(
        self,
        input_shape=(None, 128, 128, 1),
        optimizer=None,
        loss=None,
        metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if optimizer is None:
            optimizer = keras.optimizers.Adam()
        if loss is None:
            loss = keras.losses.MeanSquaredError()
        if metrics is None:
            metrics = [keras.metrics.MeanAbsoluteError()]

        self._model = self._make_model(input_shape=input_shape)
        self._model.build(input_shape)
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def _make_model(self, input_shape=(None, 128, 128, 1)):
        inputs = keras.layers.Input(shape=input_shape[1:])

        left_eye_model = self._make_eye_model(input_shape=input_shape)(inputs)
        right_eye_model = self._make_eye_model(input_shape=input_shape)(inputs)
        face_model = self._make_face_model(input_shape=input_shape)(inputs)

        # eyes_concatenated = keras.layers.concatenate(
        #     [left_eye_model, right_eye_model], axis=1
        # )
        # eyes_dense = keras.layers.Dense(128, activation="relu")(eyes_concatenated)

        # concatenated = keras.layers.concatenate([eyes_dense, face_model])
        concatenated = keras.layers.concatenate([left_eye_model, right_eye_model, face_model])

        dense = keras.layers.Dense(
            128, activation="relu", kernel_initializer="he_normal"
        )(concatenated)
        output = keras.layers.Dense(2, kernel_initializer="he_normal")(dense)

        return keras.Model(
            inputs=[inputs],
            outputs=output,
        )

    def _make_eye_model(self, input_shape=(None, 128, 128, 1)):
        return keras.Sequential(
            [
                keras.layers.Input(input_shape[1:]),
                keras.layers.Conv2D(
                    96,
                    kernel_size=(15, 15),
                    strides=4,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(
                    256,
                    kernel_size=(7, 7),
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(
                    384,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                keras.layers.Conv2D(
                    64,
                    kernel_size=(1, 1),
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                keras.layers.Flatten(),
            ]
        )

    def _make_face_model(self, input_shape=(None, 128, 128, 1)):
        return keras.Sequential(
            [
                keras.layers.Input(input_shape[1:]),
                self._make_eye_model(input_shape=input_shape),
                keras.layers.Dense(
                    256, activation="relu", kernel_initializer="he_normal"
                ),
                keras.layers.Dense(
                    128, activation="relu", kernel_initializer="he_normal"
                ),
            ]
        )

    def fit(self, x, validation_data, epochs: int):
        try:
            output = self._model.fit(
                x=x,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
                ],
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
        except KeyboardInterrupt:
            model_name = f"model_epochs{epochs}"
            self._model.save_weights(
                f"./outputs/{model_name}_{time.time_ns()}.weights.h5",
                overwrite=True,
            )

    def evaluate(self, x):
        results = self._model.evaluate(x=x)
        mean_squared_error = results[1]

        print(f"Test loss               = {results[0]:.4f}")
        print(f"Test mean squared error = {mean_squared_error:.4f}")

    def load_weights(self, filepath: str):
        self._model.load_weights(filepath)

    def predict(self, x):
        return self._model.predict(x)
