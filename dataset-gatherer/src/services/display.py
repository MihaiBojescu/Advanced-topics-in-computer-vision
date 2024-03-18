from dataclasses import dataclass

@dataclass
class DisplayAttributes:
    resolution_x: int
    resolution_y: int
    width: int
    height: int

class BaseDisplay:
    def get(self) -> DisplayAttributes:
        return DisplayAttributes(
            width=-1,
            height=-1,
            resolution_x=-1,
            resolution_y=-1,
        )

class Framework13Display:
    def get(self) -> DisplayAttributes:
        return DisplayAttributes(
            width=297,
            height=229,
            resolution_x=2256,
            resolution_y=1504,
        )
