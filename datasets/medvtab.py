import os

from .utils import Datum, DatasetBase, listdir_nohidden


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

template = [
    "a brain MRI of {}.",
    "a brain MRI scan showing {}.",
    "a medical scan of {}.",
    "a T1-weighted brain MRI of {}.",
    "a T2-weighted brain MRI of {}.",
    "an axial brain MRI of {}.",
]

CLASSNAME_MAP = {
    "glioma": "glioma tumor",
    "meningioma": "meningioma tumor",
    "notumor": "no tumor",
    "no_tumor": "no tumor",
    "pituitary": "pituitary tumor",
}


class MedVTAB(DatasetBase):
    dataset_dir = "MedVTAB"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        testing_dir = self._resolve_testing_dir(root)
        self.dataset_dir = os.path.dirname(testing_dir)
        self.template = template

        test = self._read_folder(testing_dir)
        super().__init__(test=test)

    @staticmethod
    def _resolve_testing_dir(root):
        candidates = [
            # os.path.join(root, "BTMRI"),
            # os.path.join(root, "BTMRI", "testing"),
            os.path.join(root, "MedVTAB", "Testing"),
            os.path.join(root, "MedVTAB", "testing"),
            os.path.join(root, "medvtab", "Testing"),
            os.path.join(root, "medvtab", "testing"),
            os.path.join(root, "Testing"),
            os.path.join(root, "testing"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find MedVTAB Testing folder under {}. "
            "Expected MedVTAB/Testing or Testing.".format(root)
        )

    @staticmethod
    def _is_image(filename):
        return filename.lower().endswith(IMG_EXTS)

    @classmethod
    def _read_folder(cls, testing_dir):
        class_dirs = [
            d
            for d in listdir_nohidden(testing_dir, sort=True)
            if os.path.isdir(os.path.join(testing_dir, d))
        ]
        if not class_dirs:
            raise FileNotFoundError(
                "No class folders found under {}.".format(testing_dir)
            )

        items = []
        for label, class_name in enumerate(class_dirs):
            class_dir = os.path.join(testing_dir, class_name)
            key = class_name.strip().lower()
            classname = CLASSNAME_MAP.get(key, class_name.replace("_", " ").strip())
            for filename in listdir_nohidden(class_dir, sort=True):
                if not cls._is_image(filename):
                    continue
                impath = os.path.join(class_dir, filename)
                items.append(Datum(impath=impath, label=label, classname=classname))

        if not items:
            raise FileNotFoundError(
                "No images found under {}.".format(testing_dir)
            )
        return items
