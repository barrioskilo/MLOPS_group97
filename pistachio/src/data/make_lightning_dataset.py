import os

import click
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PistachioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=32, transform=None, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.dims = (3, 256, 256)
        self.num_classes = 2

    def setup(self, stage=None):
        # Set seed for reproducibility
        pl.seed_everything(self.seed)

        # Load dataset
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)

        # Calculate sizes of train, validation, and test sets
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
