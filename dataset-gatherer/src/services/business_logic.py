from services.camera import BaseCamera
from services.display import BaseDisplay
from services.distance import BaseDistance
from services.writer import Writer


class BusinessLogic:
    _camera: BaseCamera
    _display: BaseDisplay
    _distance: BaseDistance
    _writer: Writer

    def __init__(
        self,
        camera: BaseCamera,
        display: BaseDisplay,
        distance: BaseDistance,
        writer: Writer,
    ) -> None:
        self._camera = camera
        self._display = display
        self._distance = distance
        self._writer = writer

    def run(self, x: int, y: int):
        display = self._display.get()
        distance = self._distance.get()
        frame = next(self._camera)

        self._writer.run(
            frame=frame,
            x=x,
            y=y,
            resolution_x=display.resolution_x,
            resolution_y=display.resolution_y,
            width=display.width,
            height=display.height,
            distance=distance,
        )
