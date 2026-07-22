import os

from PIL import Image
from torch.utils.data import Dataset as TorchDataset


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_TEMPLATE = [
    "a medical image of {}.",
    "a medical scan showing {}.",
]


def _is_image(filename):
    return filename.lower().endswith(IMG_EXTS)


def _format_classname(name):
    return name.replace("_", " ").strip()


class HardbenchFolderDataset(TorchDataset):
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


class HardbenchFolderBase:
    dataset_dir = None
    template = DEFAULT_TEMPLATE

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        data_dir = self._resolve_data_dir(root)

        class_dirs = self._list_class_dirs(data_dir)
        if not class_dirs:
            raise FileNotFoundError(
                "No class folders found under {}.".format(data_dir)
            )

        items = []
        classnames = [_format_classname(name) for name in class_dirs]
        for label, class_name in enumerate(class_dirs):
            class_dir = os.path.join(data_dir, class_name)
            for filename in sorted(os.listdir(class_dir)):
                if filename.startswith(".") or not _is_image(filename):
                    continue
                impath = os.path.join(class_dir, filename)
                items.append((impath, label))

        if not items:
            raise FileNotFoundError(
                "No images found under {}.".format(data_dir)
            )

        self.dataset_dir = data_dir
        self.template = self.template
        self.classnames = classnames
        self.test = HardbenchFolderDataset(items, transform=None)

    def _resolve_data_dir(self, root):
        if not self.dataset_dir:
            raise ValueError("dataset_dir must be set on the dataset class.")
        candidate = os.path.join(root, self.dataset_dir)
        if os.path.isdir(candidate):
            return candidate
        if os.path.isdir(root) and os.path.basename(root) == self.dataset_dir:
            return root
        raise FileNotFoundError(
            "Could not find {} under {}. Please unzip the dataset.".format(
                self.dataset_dir, root
            )
        )

    @staticmethod
    def _list_class_dirs(data_dir):
        class_dirs = []
        for name in sorted(os.listdir(data_dir)):
            if name.startswith("."):
                continue
            full_path = os.path.join(data_dir, name)
            if os.path.isdir(full_path):
                class_dirs.append(name)
        return class_dirs


class Hardbench_BTMRI(HardbenchFolderBase):
    dataset_dir = "BTMRI"


class Hardbench_BUSI(HardbenchFolderBase):
    dataset_dir = "BUSI"


class Hardbench_CHMNIST(HardbenchFolderBase):
    dataset_dir = "CHMNIST"


class Hardbench_CTKidney(HardbenchFolderBase):
    dataset_dir = "CTKidney"


class Hardbench_COVID19(HardbenchFolderBase):
    dataset_dir = "COVID_19"


class Hardbench_DermaMNIST(HardbenchFolderBase):
    dataset_dir = "DermaMNIST"


class Hardbench_KneeXray(HardbenchFolderBase):
    dataset_dir = "KneeXray"


class Hardbench_Kvasir(HardbenchFolderBase):
    dataset_dir = "Kvasir"


class Hardbench_LungColon(HardbenchFolderBase):
    dataset_dir = "LungColon"


class Hardbench_Retina(HardbenchFolderBase):
    dataset_dir = "RETINA"
