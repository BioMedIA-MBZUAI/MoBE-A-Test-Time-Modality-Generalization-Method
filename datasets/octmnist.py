import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


template = [
    "an OCT image of {}.",
    "a retinal OCT scan showing {}.",
]

CLASSNAMES = [
    "choroidal neovascularization",
    "diabetic macular edema",
    "drusen",
    "normal",
]

FOLDER_CLASS_MAP = {
    "choroidal_neovascularization": 0,
    "diabetic_macular_edema": 1,
    "drusen": 2,
    "normal_OCT_scan": 3,
}


class OCTMNISTDataset(TorchDataset):
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


class OCTMNISTFolderDataset(TorchDataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        impath, label = self.items[idx]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class OCTMNIST:
    dataset_dir = "octmnist"

    def __init__(self, root, npz_filename=None):
        root = os.path.abspath(os.path.expanduser(root))
        self.template = template

        npz_path = self._resolve_npz_path(root, npz_filename)
        if npz_path:
            self.dataset_dir = os.path.dirname(npz_path)
            data = np.load(npz_path)
            images = data["test_images"]
            labels = data["test_labels"]
            labels = self._labels_to_single(labels)

            self.classnames = self._resolve_classnames(labels)
            self.test = OCTMNISTDataset(images, labels, transform=None)
            return
        if npz_filename:
            raise FileNotFoundError(
                "Could not find {} under {}.".format(npz_filename, root)
            )

        folder_path = self._resolve_folder_path(root)
        if not folder_path:
            raise FileNotFoundError(
                "Could not find octmnist.npz or an OCTMNIST folder under {}.".format(root)
            )
        self.dataset_dir = folder_path
        items, classnames = self._load_folder_items(folder_path)
        self.classnames = classnames
        self.test = OCTMNISTFolderDataset(items, transform=None)

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
                os.path.join(root, "octmnist", npz_filename),
                os.path.join(root, npz_filename),
            ]
        else:
            candidates = [
                os.path.join(root, "octmnist", "octmnist.npz"),
                os.path.join(root, "octmnist.npz"),
                os.path.join(root, "octamnist", "octamnist.npz"),
                os.path.join(root, "octamnist.npz"),
            ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _resolve_folder_path(root):
        candidates = [
            root,
            os.path.join(root, "OCTMNIST"),
            os.path.join(root, "octmnist"),
            os.path.join(root, "OCTAMNIST"),
            os.path.join(root, "octamnist"),
        ]
        expected = set(FOLDER_CLASS_MAP.keys())
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            subdirs = {
                d
                for d in os.listdir(candidate)
                if os.path.isdir(os.path.join(candidate, d))
            }
            if expected.issubset(subdirs):
                return candidate
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            subdirs = [
                d
                for d in os.listdir(candidate)
                if os.path.isdir(os.path.join(candidate, d))
            ]
            if not subdirs:
                continue
            for subdir in subdirs:
                files = os.listdir(os.path.join(candidate, subdir))
                if any(f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) for f in files):
                    return candidate
        return None

    @staticmethod
    def _load_folder_items(folder_path):
        subdirs = [
            d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
        ]
        expected = set(FOLDER_CLASS_MAP.keys())
        if expected.issubset(subdirs):
            ordered = list(FOLDER_CLASS_MAP.keys())
            class_map = FOLDER_CLASS_MAP
            classnames = CLASSNAMES
        else:
            ordered = sorted(subdirs)
            class_map = {name: idx for idx, name in enumerate(ordered)}
            classnames = [name.replace("_", " ") for name in ordered]

        items = []
        for class_name in ordered:
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                impath = os.path.join(class_dir, filename)
                items.append((impath, class_map[class_name]))

        if not items:
            raise FileNotFoundError(
                "No images found under {} (expected subfolders with images).".format(
                    folder_path
                )
            )
        return items, classnames
