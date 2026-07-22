import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


template = [
    "a dermoscopic image of {}.",
    "a dermatoscopic photo of {}.",
]

CLASSNAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevus",
    "vascular lesions",
]


class DermMNISTDataset(TorchDataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


class DermMNIST:
    dataset_dir = "dermamnist"

    def __init__(self, root, npz_filename=None):
        root = os.path.abspath(os.path.expanduser(root))
        npz_path = self._resolve_npz_path(root, npz_filename)
        self.dataset_dir = os.path.dirname(npz_path)
        self.template = template
        self.classnames = CLASSNAMES

        data = np.load(npz_path)
        images = data["test_images"]
        labels = data["test_labels"]
        labels = self._labels_to_single(labels)

        self.test = DermMNISTDataset(images, labels, transform=None)

    @staticmethod
    def _labels_to_single(labels):
        labels = labels.astype(np.int64)
        return labels.reshape(-1)

    @staticmethod
    def _resolve_npz_path(root, npz_filename=None):
        if npz_filename:
            candidates = [
                os.path.join(root, "dermamnist", npz_filename),
                os.path.join(root, npz_filename),
            ]
        else:
            candidates = [
                os.path.join(root, "dermamnist", "dermamnist.npz"),
                os.path.join(root, "dermamnist.npz"),
            ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        target = npz_filename or "dermamnist.npz"
        raise FileNotFoundError(
            "Could not find {} under {} (checked: {})".format(
                target, root, ", ".join(candidates)
            )
        )
