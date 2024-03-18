class BaseDistance:
    def get(self) -> float:
        return -1


class FixedDistance(BaseDistance):
    _distance: float

    def __init__(self, distance: float):
        self._distance = distance

    def get(self) -> float:
        return self._distance
