from dataclasses import dataclass
from argparse import ArgumentParser


@dataclass
class Args:
    epochs: int
    batch_size: int


def parse_arguments() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs",
        help="Epochs to run",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="The batch size to use. Use 0 for on batched (offline) learning. Use 1 for online training.",
        default=25,
        type=int,
    )
    result = parser.parse_args()

    if result.epochs < 0:
        raise RuntimeError(f"Epochs number error: {result.epochs} < 0")

    if result.batch_size < 0:
        raise RuntimeError(f"Batch size error: {result.batch_size} < 0")

    return Args(epochs=result.epochs, batch_size=result.batch_size)
