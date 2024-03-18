#!/usr/bin/env python3
from services.args import ArgsParser
from services.business_logic import BusinessLogic
from services.camera import ImageIOCamera
from services.display import Framework13Display
from services.distance import FixedDistance
from services.web import Web
from services.writer import Writer


def main():
    args_parser = ArgsParser()
    args = args_parser.run()

    camera = ImageIOCamera()
    display = Framework13Display()
    distance = FixedDistance(distance=args.distance)
    writer = Writer(
        path="./outputs",
        image_filename_slug=args.image_filename_slug,
    )
    business_logic = BusinessLogic(
        camera=camera, display=display, distance=distance, writer=writer
    )
    web = Web(business_logic=business_logic, port=args.port)

    web.run()


if __name__ == "__main__":
    main()
