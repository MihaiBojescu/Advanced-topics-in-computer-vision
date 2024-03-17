import typing as t
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from data.dataset import ImageDataset
from data.cacheable_tensor_dataset import CacheableTensorDataset
from utils.args import parse_arguments
from utils.device import get_default_device


def main():
    args = parse_arguments()
    device = get_default_device()
    transforms: t.Callable[[Tensor], Tensor] = []

    dataset = ImageDataset(data_path="./dataset", transforms=transforms)
    cached_dataset = CacheableTensorDataset(dataset=dataset, cache=True)
    batched_image_dataloader = DataLoader(
        dataset=cached_dataset,
        batch_size=len(cached_dataset) if args.batch_size == 0 else args.batch_size,
        shuffle=True,
    )

    # TODO: implement the rest:
    #    1. add image transformations to enhance accuracy (ideas: remove background, extract eyes only)
    #    2. add image normalisations
    #    3. add neural network
    #    4. add client-side code for Windows/Linux/MacOS
    #    5. ...

if __name__ == "__main__":
    main()
