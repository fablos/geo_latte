from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv

import numpy as np
from typing import Literal

from .utils import select_classes, select_num_samples, image_to_numpy


class MNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        path_root="data/",
        set_purp: Literal["train", "val", "test"] = "train",
        n_samples: int = None,
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        download=True,
        normalizing_stats=None,
        transform=None,
    ):
        self.set = set_purp
        self.path = Path(path_root)
        if self.set == "train" or self.set == "val":
            self.dataset = tv.datasets.MNIST(root=self.path, train=True, download=download)
        else:
            self.dataset = tv.datasets.MNIST(root=self.path, train=False, download=download)
        self.transform = transform

        class_to_index = {c: i for i, c in enumerate(cls)}
        if len(cls) < 10:
            self.dataset = select_classes(self.dataset, class_to_index)
        if n_samples is not None:
            self.dataset = select_num_samples(self.dataset, n_samples, class_to_index)

        self.data, self.targets = (self.dataset.data.float() / 255.0).numpy(), F.one_hot(
            self.dataset.targets, 10
        ).numpy()

    def __getitem__(self, index):
        img, target = np.expand_dims(self.data[index], axis=0), self.targets[index]
        if self.transform is not None:
            img = self.transform(torch.from_numpy(img)).numpy()
        return img, target

    def __len__(self):
        return len(self.data)