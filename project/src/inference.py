from nn.model.model import Model
from data.dataset import TensorDataset
from data.label_transforms import label_to_ints
import keras

from data.dataloader import ImageDataloader


def main():
    label_transforms = [label_to_ints]

    test_dataset = TensorDataset(
        data_path="./dataset/256x256",
        data_file_path="./test.csv",
        label_transforms=label_transforms,
    )
    test_dataloader = ImageDataloader(dataset=test_dataset, batch_size=1024)

    model = Model()
    model.load_weights("./outputs/unnormalised/model_epochs25_loss162038.4062_val-loss243631.0156_1717439475669035715.weights.h5")
    
    result = model.predict(x=test_dataloader)

    print(result)

if __name__ == "__main__":
    main()
