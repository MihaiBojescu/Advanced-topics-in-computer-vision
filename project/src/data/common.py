from dataclasses import dataclass
import typing as t


@dataclass
class Metadata:
    filename: str
    x: int
    y: int
    width: int
    height: int
    distance: float


@dataclass
class Label:
    x: int
    y: int


T = t.TypeVar("T")
U = t.TypeVar("U")
L = t.TypeVar("L")
