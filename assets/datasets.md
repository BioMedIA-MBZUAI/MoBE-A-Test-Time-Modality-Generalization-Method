# Dataset Preparation

This project expects all datasets under `datasets_all/`, with MedMNIST `.npz` files separated from folder-style HardBench datasets.

```text
datasets_all/
├── medmnist/
│   ├── bloodmnist_224.npz
│   ├── breastmnist_224.npz
│   ├── chestmnist_224.npz
│   ├── dermamnist_224.npz
│   ├── organamnist_224.npz
│   ├── organcmnist_224.npz
│   ├── organsmnist_224.npz
│   ├── pathmnist_224.npz
│   ├── retinamnist_224.npz
│   ├── tissuemnist_224.npz
│   └── octmnist_224.npz
├── hardbench/
│   ├── BTMRI/
│   ├── BUSI/
│   ├── CHMNIST/
│   ├── COVID_19/
│   ├── CTKidney/
│   ├── DermaMNIST/
│   ├── KneeXray/
│   ├── Kvasir/
│   ├── LungColon/
│   └── RETINA/
```

The config `hardbench_octmnist.yaml` uses the same OCTMNIST loader and should use `datasets_all/medmnist/octmnist_224.npz`; it does not require a separate HardBench folder.

## MedMNIST

The MedMNIST files used by this repo are the 224-resolution `.npz` files from:

https://zenodo.org/records/10519652

You can use the provided script:

```bash
scripts/download_medmnist.sh
```

The script downloads into:

```text
datasets_all/medmnist/
```

## HardBench

Download and unzip only the HardBench datasets used by the current code. Place each folder directly under `datasets_all/hardbench/`.

| Config name | Folder name | Modality | Organ(s) | Classes | Train/val/test | Source |
|---|---|---|---|---|---|---|
| `hardbench_btmri` | `BTMRI` | MRI | Brain | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor | 2854/1141/1717 | [Drive](https://drive.google.com/file/d/1_lJLZRUmczqZqoN-dNqkAzGzmi4ONoU5/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI.zip) |
| `hardbench_busi` | `BUSI` | Ultrasound | Breast | Benign Tumors, Malignant Tumors, Normal Scans | 389/155/236 | [Drive](https://drive.google.com/file/d/1hB5M7wcAUTV9EtiYrijACoQ36R6VmQaa/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUSI.zip) |
| `hardbench_chmnist` | `CHMNIST` | Histopathology | Colorectal | Adipose Tissue, Complex Stroma, Debris, Empty Background, Immune Cells, Normal Mucosal Glands, Simple Stroma, Tumor Epithelium | 2496/1000/1504 | [Drive](https://drive.google.com/file/d/1tyQiYQmqAGNaY4SCK_8U5vEbbaa1AD-g/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CHMNIST.zip) |
| `hardbench_covid19` | `COVID_19` | X-Ray | Chest | COVID-19, Lung Opacity, Normal Lungs, Viral Pneumonia | 10582/4232/6351 | [Drive](https://drive.google.com/file/d/1zMLN5q5e_tmH-deSZQiY4Xq0M1EqCrML/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/COVID_19.zip) |
| `hardbench_ctkidney` | `CTKidney` | CT | Kidney | Kidney Cyst, Kidney Stone, Kidney Tumor, Normal Kidney | 6221/2487/3738 | [Drive](https://drive.google.com/file/d/1PBZ299k--mZL8JU7nhC1Wy8yEmlqmVDh/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CTKidney.zip) |
| `hardbench_dermamnist` | `DermaMNIST` | Dermatoscopy | Skin | Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanocytic Nevus, Melanoma, Vascular Lesion | 7007/1003/2005 | [Drive](https://drive.google.com/file/d/1Jxd1-DWljunRDZ8fY80dl5zUMefriQXt/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/DermaMNIST.zip) |
| `hardbench_kneexray` | `KneeXray` | X-Ray | Knee | No, Doubtful, Minimal, Moderate, Severe Osteoarthritis | 5778/826/1656 | [Drive](https://drive.google.com/file/d/1DBVraYJmxy2UcQ_nGLYvTB2reITOm453/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/KneeXray.zip) |
| `hardbench_kvasir` | `Kvasir` | Endoscopy | Colon | Dyed Lifted Polyps, Normal Cecum, Esophagitis, Dyed Resection Margins, Normal Pylorus, Normal Z Line, Polyps, Ulcerative Colitis | 2000/800/1200 | [Drive](https://drive.google.com/file/d/1T_cqnNIjmGazNeg6gziarvCNWGsFEkRi/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/Kvasir.zip) |
| `hardbench_lungcolon` | `LungColon` | Histopathology | Lung, Colon | Colon Adenocarcinoma, Colon Benign Tissue, Lung Adenocarcinoma, Lung Benign Tissue, Lung Squamous Cell Carcinoma | 12500/5000/7500 | [Drive](https://drive.google.com/file/d/1YIu5fqMXgyemisiL1L1HCvES2nVpCtun/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/LungColon.zip) |
| `hardbench_retina` | `RETINA` | Fundus Photography | Retina | Cataract, Diabetic Retinopathy, Glaucoma, Normal Retina | 2108/841/1268 | [Drive](https://drive.google.com/file/d/18U-Gc22h5QryomNNzY4r4Qfrq52yf5EO/view?usp=sharing) / [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/RETINA.zip) |

Each unzipped dataset should expose class folders. For example:

```text
datasets_all/hardbench/BTMRI/
├── glioma_tumor/
├── meningioma_tumor/
├── normal_brain/
└── pituitary_tumor/
```

Note: `medvtab` and the heterogeneous `hardbench_btmri` setting use the same brain MRI data, so MedVTAB is not listed as a separate download here.

## Notes

The dataset folders are intentionally git-ignored. Keep downloaded archives, extracted images, and `.npz` files local under `datasets_all/`.
