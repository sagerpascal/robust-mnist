from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def image_transform():
    return transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5), std=(0.5)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


class MNISTCDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(x, mode='L')
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def load_mnistc(dirname=None, train=False, transform=image_transform()):
    if dirname is None:
        dirname = 'line'
    root = Path('data') / 'mnist_c' / dirname
    dataset_type = 'train' if train else 'test'

    images = np.load(str(root / (dataset_type + '_images.npy')))
    images = images.transpose(0, 3, 1, 2)[:, 0, ...]

    labels = np.load(str(root / (dataset_type + '_labels.npy'))).reshape((images.shape[0],))
    return MNISTCDataset(images, labels, transform=transform)
