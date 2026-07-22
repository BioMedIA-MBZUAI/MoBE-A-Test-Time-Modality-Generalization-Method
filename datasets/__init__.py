from .bloodmnist import BloodMNIST
from .breastmnist import BreastMNIST
from .chestmnist import ChestMNIST
from .dermamnist import DermMNIST
from .hardbench import (
    Hardbench_BTMRI,
    Hardbench_BUSI,
    Hardbench_CHMNIST,
    Hardbench_COVID19,
    Hardbench_CTKidney,
    Hardbench_DermaMNIST,
    Hardbench_KneeXray,
    Hardbench_Kvasir,
    Hardbench_LungColon,
    Hardbench_Retina,
)
from .medvtab import MedVTAB
from .octmnist import OCTMNIST
from .organamnist import OrganAMNIST
from .organcmnist import OrganCMNIST
from .organsmnist import OrganSMNIST, OrganSMNIST224, OrganSMNIST64
from .pathmnist import PathMNIST
from .retinamnist import RetinaMNIST
from .tissuemnist import TissueMNIST


dataset_list = {
    "bloodmnist": BloodMNIST,
    "breastmnist": BreastMNIST,
    "chestmnist": ChestMNIST,
    "dermamnist": DermMNIST,
    "pathmnist": PathMNIST,
    "organamnist": OrganAMNIST,
    "organcmnist": OrganCMNIST,
    "organsmnist": OrganSMNIST,
    "organsmnist_224": OrganSMNIST224,
    "organsmnist_64": OrganSMNIST64,
    "octmnist": OCTMNIST,
    "octamnist": OCTMNIST,
    "retinamnist": RetinaMNIST,
    "tissuemnist": TissueMNIST,
    "medvtab": MedVTAB,
    "hardbench_btmri": Hardbench_BTMRI,
    "hardbench_busi": Hardbench_BUSI,
    "hardbench_chmnist": Hardbench_CHMNIST,
    "hardbench_ctkidney": Hardbench_CTKidney,
    "hardbench_covid19": Hardbench_COVID19,
    "hardbench_dermamnist": Hardbench_DermaMNIST,
    "hardbench_kneexray": Hardbench_KneeXray,
    "hardbench_kvasir": Hardbench_Kvasir,
    "hardbench_lungcolon": Hardbench_LungColon,
    "hardbench_retina": Hardbench_Retina,
    "hardbench_octmnist": OCTMNIST,
}

MEDMNIST_VARIANT_BASES = {
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "pathmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "retinamnist",
    "tissuemnist",
    "octmnist",
}


def _resolve_variant_name(dataset):
    if "_" not in dataset:
        return None
    base, suffix = dataset.rsplit("_", 1)
    if not suffix.isdigit() or base not in MEDMNIST_VARIANT_BASES:
        return None
    return base, suffix


def build_dataset(dataset, root_path):
    if dataset in dataset_list:
        return dataset_list[dataset](root_path)

    variant = _resolve_variant_name(dataset)
    if variant:
        base, suffix = variant
        npz_filename = f"{base}_{suffix}.npz"
        return dataset_list[base](root_path, npz_filename=npz_filename)

    raise KeyError("Unknown dataset: {}".format(dataset))
