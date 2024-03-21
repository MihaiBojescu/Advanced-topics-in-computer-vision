from dataclasses import dataclass
from argparse import ArgumentParser


@dataclass
class Arguments:
    image_filename_slug: str
    distance: int
    port: int


class ArgsParser:
    def run(self) -> Arguments:
        parser = ArgumentParser()
        parser.add_argument(
            "--image_filename_slug",
            type=str,
            default="rh_",
            required=False,
            help="Slug for the image filename. Used to ensure unicity",
        )
        parser.add_argument(
            "--distance",
            type=int,
            default=100,
            required=False,
            help="Distance from the camera when using fixed distancing",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=3000,
            required=False,
            help="Port to use for the web server",
        )
        result = parser.parse_args()

        return Arguments(
            image_filename_slug=result.image_filename_slug,
            distance=result.distance,
            port=result.port,
        )
