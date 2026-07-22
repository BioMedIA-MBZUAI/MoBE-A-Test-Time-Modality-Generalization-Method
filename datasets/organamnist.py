import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


template = [
    "a CT slice of the {}.",
    "an abdominal CT image of the {}.",
]

CLASSNAMES = [
    "bladder",
    "femur (left)",
    "femur (right)",
    "heart",
    "kidney (left)",
    "kidney (right)",
    "liver",
    "lung (left)",
    "lung (right)",
    "pancreas",
    "spleen",
]


class OrganAMNISTDataset(TorchDataset):
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


class OrganAMNIST:
    dataset_dir = "organamnist"

    def __init__(self, root, npz_filename=None):
        root = os.path.abspath(os.path.expanduser(root))
        npz_path = self._resolve_npz_path(root, npz_filename)
        self.dataset_dir = os.path.dirname(npz_path)
        self.template = template

        data = np.load(npz_path)
        images = data["test_images"]
        labels = data["test_labels"]
        labels = self._labels_to_single(labels)

        self.classnames = self._resolve_classnames(labels)
        self.test = OrganAMNISTDataset(images, labels, transform=None)

    @staticmethod
    def _labels_to_single(labels):
        labels = labels.astype(np.int64)
        return labels.reshape(-1)

    @staticmethod
    def _resolve_classnames(labels):
        num_classes = int(labels.max()) + 1
        if num_classes == len(CLASSNAMES):
            return CLASSNAMES
        return [f"class {i}" for i in range(num_classes)]

    @staticmethod
    def _resolve_npz_path(root, npz_filename=None):
        if npz_filename:
            candidates = [
                os.path.join(root, "organamnist", npz_filename),
                os.path.join(root, npz_filename),
            ]
        else:
            candidates = [
                os.path.join(root, "organamnist", "organamnist.npz"),
                os.path.join(root, "organamnist.npz"),
            ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        target = npz_filename or "organamnist.npz"
        raise FileNotFoundError(
            "Could not find {} under {} (checked: {})".format(
                target, root, ", ".join(candidates)
            )
        )
