import keras


class BoundedMeanSquaredError:
    """
    Ensures as much as possible that the loss is between a lower and upper bound (lower_bound <= loss <= upper_bound)
    """
    _lower_bound: float
    _upper_bound: float
    _penalty: float

    def __init__(
        self, lower_bound: float = 0, upper_bound: float = 1, penalty: float = 1000
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._penalty = penalty

    def __call__(self, y_true, y_pred):
        mse = keras.ops.mean(keras.ops.square(y_pred - y_true), axis=-1)
        loss = mse * self._penalty * keras.ops.mean(
            keras.ops.square(
                y_pred - keras.ops.clip(y_pred, self._lower_bound, self._upper_bound)
            ),
            axis=-1,
        )

        return loss
