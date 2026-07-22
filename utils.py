import os
import csv
import yaml
import torch
import math
import numpy as np
from datasets import build_dataset
from datasets.utils import AugMixAugmenter, build_data_loader
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

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

HARDBENCH_DATASETS = {
    "hardbench_btmri",
    "hardbench_busi",
    "hardbench_chmnist",
    "hardbench_ctkidney",
    "hardbench_covid19",
    "hardbench_dermamnist",
    "hardbench_kneexray",
    "hardbench_kvasir",
    "hardbench_lungcolon",
    "hardbench_retina",
    "hardbench_octmnist"
}

DATASET_DISPLAY_NAMES = {
    "medvtab": "MedVTAB",
    "bloodmnist": "BloodMNIST",
    "breastmnist": "BreastMNIST",
    "chestmnist": "ChestMNIST",
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
    "retinamnist": "RetinaMNIST",
    "tissuemnist": "TissueMNIST",
    "octmnist": "OCTMNIST",
    "hardbench_btmri": "BTMRI",
    "hardbench_busi": "BUSI",
    "hardbench_chmnist": "CHMNIST",
    "hardbench_ctkidney": "CTKIDNEY",
    "hardbench_covid19": "COVID QU-Ex",
    "hardbench_dermamnist": "DermaMNIST",
    "hardbench_kneexray": "KneeXray",
    "hardbench_kvasir": "Kvasir",
    "hardbench_lungcolon": "LC25000",
    "hardbench_retina": "RETINA",
    "hardbench_octmnist": "OCTMNIST",
}


def _is_medmnist_variant(dataset_name):
    if "_" not in dataset_name:
        return False
    base, suffix = dataset_name.rsplit("_", 1)
    return suffix.isdigit() and base in MEDMNIST_VARIANT_BASES


def _is_medmnist_dataset(dataset_name):
    return (
        dataset_name in MEDMNIST_VARIANT_BASES
        or dataset_name in {"octamnist", "hardbench_octmnist"}
        or _is_medmnist_variant(dataset_name)
    )


def _root_candidates(root_path):
    root = os.path.abspath(os.path.expanduser(str(root_path)))
    base = os.path.basename(root.rstrip(os.sep))
    parent = os.path.dirname(root.rstrip(os.sep))

    if base in {"medmnist", "hardbench"}:
        dataset_all_root = parent
    else:
        dataset_all_root = root

    return root, dataset_all_root


def resolve_dataset_root(dataset_name, root_path):
    """
    Resolve mixed MedMNIST/HardBench runs from either:
    - .../datasets_all
    - .../datasets_all/medmnist
    - .../datasets_all/hardbench
    """
    root, dataset_all_root = _root_candidates(root_path)

    if _is_medmnist_dataset(dataset_name):
        med_root = os.path.join(dataset_all_root, "medmnist")
        return med_root if os.path.isdir(med_root) else root

    if dataset_name in HARDBENCH_DATASETS:
        hard_root = os.path.join(dataset_all_root, "hardbench")
        return hard_root if os.path.isdir(hard_root) else root

    if dataset_name == "medvtab":
        medvtab_root = os.path.join(dataset_all_root, "medvtab")
        return medvtab_root if os.path.isdir(medvtab_root) else root

    return root


def dataset_display_name(dataset_name):
    """Paper-friendly dataset name for figure titles."""
    name = str(dataset_name)
    if name in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[name]
    if _is_medmnist_variant(name):
        base, _ = name.rsplit("_", 1)
        return DATASET_DISPLAY_NAMES.get(base, base)
    return name

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    target = target.to(output.device)
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, tokenizer):
    with torch.no_grad():
        clip_weights = []
        device = next(clip_model.parameters()).device

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = tokenizer(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        device = next(clip_model.parameters()).device
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
        else:
            images = images.to(device)

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def estimate_image_encoder_gflops(model, image_shape, device=None, forward_fn=None):
    """
    Estimate image-encoder GFLOPs for Conv2d/Linear layers.
    Uses one dummy image and counts one multiply-add as two FLOPs.
    """
    hooks = []
    macs = {"total": 0}

    def conv_hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs["total"] += int(output.numel()) * int(kernel_ops)

    def linear_hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        macs["total"] += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    try:
        param = next(model.parameters())
        model_device = param.device
        dtype = param.dtype if param.is_floating_point() else torch.float32
    except StopIteration:
        model_device = torch.device(device or "cpu")
        dtype = torch.float32

    device = torch.device(device) if device is not None else model_device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, *image_shape), device=device, dtype=dtype)
        if forward_fn is not None:
            forward_fn(dummy)
        elif hasattr(model, "encode_image"):
            model.encode_image(dummy)
        elif hasattr(model, "get_image_features"):
            model.get_image_features(pixel_values=dummy)
        else:
            model(dummy)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return 2.0 * float(macs["total"]) / 1e9


def compute_single_encoder_stats(model, sample_images, device=None, forward_fn=None, image_forwards_per_sample=1):
    image_encoder_gflops = estimate_image_encoder_gflops(
        model,
        tuple(sample_images.shape[1:]),
        device=device,
        forward_fn=forward_fn,
    )
    gflops_per_sample = image_encoder_gflops * float(image_forwards_per_sample)
    return {
        "avg_selected_experts": 1.0,
        "image_encoder_gflops": image_encoder_gflops,
        "gflops_per_sample": gflops_per_sample,
    }


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    root_path = resolve_dataset_root(dataset_name, root_path)

    if dataset_name == 'I':
        raise ValueError("ImageNet is not bundled in this release repo.")
    
    elif dataset_name in ['A','V','R','S']:
        raise ValueError("ImageNet OOD datasets are not bundled in this release repo.")

    elif dataset_name in ['medvtab']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['chestmnist', 'dermamnist', 'breastmnist', 'pathmnist', 'bloodmnist', 'organamnist', 'organcmnist',       'organsmnist', 'organsmnist_224', 'organsmnist_64', 'retinamnist', 'tissuemnist', 'octmnist', 'octamnist', \
                          \
                          'hardbench_btmri', 'hardbench_busi', 'hardbench_chmnist', 'hardbench_ctkidney', 'hardbench_covid19', 'hardbench_dermamnist', 'hardbench_kneexray', 'hardbench_kvasir', 'hardbench_lungcolon', 'hardbench_retina', 'hardbench_octmnist'] or _is_medmnist_variant(dataset_name):
        dataset = build_dataset(dataset_name, root_path)
        dataset.test.transform = preprocess
        test_loader = torch.utils.data.DataLoader(
            dataset.test,
            batch_size=1,
            num_workers=8,
            shuffle=True,
            drop_last=False,
            pin_memory=(torch.cuda.is_available())
        )

    else:
        raise ValueError("Dataset is not from the chosen list: {}".format(dataset_name))
    
    return test_loader, dataset.classnames, dataset.template


# ----------------------------
# Logging helpers
# ----------------------------
def _safe_getattr(obj, name, default=""):
    return getattr(obj, name, default)


def init_log(results_path):
    """
    Initialize the CSV log file.
    Creates the file and writes header ONLY if it does not exist.
    """
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "dataset",
                "accuracy",
                "avg_time",
                "topk",
                "tau",
                "hard_mix",
                "mix_strategy",
                "update_topk",
                # "backbone"
            ])


def log_results(results_path, dataset_name, acc, avg_time, args, timestamp):
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        mix_strategy = getattr(args, 'mix_strategy', None)
        if mix_strategy is None:
            mix_strategy = "NA"
        update_topk = getattr(args, 'update_topk', None)
        if update_topk is None:
            update_topk = "NA"
        writer.writerow([
            timestamp,
            dataset_name,
            acc,
            avg_time,
            args.topk,
            args.tau,
            args.hard_mix,
            mix_strategy,
            update_topk,
            # args.backbone
        ])


def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))


def _to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 2 and x.shape[1] > 1:
        return x.argmax(axis=1)
    return x.reshape(-1)


def _to_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [N, C], got {x.shape}.")
    return x


def reliability_bin_stats(probs, labels, n_bins=15):
    """Return per-bin confidence, accuracy, count, and ECE in percent."""
    probs_np = _to_numpy_2d(probs)
    labels_np = _to_numpy_1d(labels).astype(np.int64)
    if probs_np.shape[0] != labels_np.shape[0]:
        raise ValueError("probs and labels must contain the same number of samples.")

    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    correct = (predictions == labels_np).astype(np.float64)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_conf = np.full(int(n_bins), np.nan, dtype=np.float64)
    bin_acc = np.full(int(n_bins), np.nan, dtype=np.float64)
    bin_count = np.zeros(int(n_bins), dtype=np.int64)
    ece = 0.0

    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        bin_count[b] = int(in_bin.sum())
        if bin_count[b] == 0:
            continue
        bin_conf[b] = float(confidences[in_bin].mean())
        bin_acc[b] = float(correct[in_bin].mean())
        ece += (bin_count[b] / max(1, len(labels_np))) * abs(bin_acc[b] - bin_conf[b])

    return {
        "bin_edges": edges,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
        "ece": 100.0 * float(ece),
        "accuracy": 100.0 * float(correct.mean()) if len(correct) else 0.0,
        "n_samples": int(len(labels_np)),
    }


def save_reliability_figure(
    probs,
    labels,
    save_path,
    dataset_name=None,
    method_name=None,
    n_bins=15,
    title=None,
    dpi=450,
    save_companion=True,
):
    """
    Save a square, paper-friendly reliability diagram.

    The main path extension controls the primary format. If the extension is
    .pdf, a high-DPI .png companion is saved too; if it is .png, a .pdf
    companion is saved too. Returns the bin statistics dictionary.
    """
    import sys
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stats = reliability_bin_stats(probs, labels, n_bins=n_bins)
    edges = stats["bin_edges"]
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    acc = stats["bin_accuracy"]
    counts = stats["bin_count"]
    valid = counts > 0
    diagonal = edges[1:]
    gap_bottom = np.minimum(acc, diagonal)
    gap_height = np.abs(diagonal - acc)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.25, 3.25), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot([0, 1], [0, 1], color="#9b9b9b", linewidth=1.0, linestyle="--", zorder=1)
    ax.bar(
        centers[valid],
        gap_height[valid],
        bottom=gap_bottom[valid],
        width=widths[valid] * 0.98,
        align="center",
        facecolor=(1.0, 1.0, 1.0, 0.35),
        edgecolor="#f2a3a3",
        linewidth=0.45,
        hatch="//",
        label="Overconfidence",
        zorder=2,
    )
    ax.bar(
        centers[valid],
        acc[valid],
        width=widths[valid] * 0.98,
        align="center",
        color="#9be7f5",
        edgecolor="#008fa3",
        linewidth=0.45,
        alpha=1.0,
        label="Outputs",
        zorder=3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, color="#cfcfcf", linewidth=0.65, linestyle=(0, (1.2, 2.6)), alpha=0.95, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.65)

    if title is None:
        suffix_parts = [p for p in (method_name, dataset_name) if p]
        suffix = "-".join(suffix_parts)
        title = f"Reliability-Diagram {suffix}" if suffix else "Reliability-Diagram"
    ax.set_title(title, pad=3)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Outputs"), labels.index("Overconfidence")] if "Outputs" in labels and "Overconfidence" in labels else None
    if order is not None:
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    handles.append(Line2D([], [], linestyle="none", marker=None, color="none"))
    labels.append(f"ECE = {stats['ece']:.2f}%")
    ax.legend(handles, labels, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#d7d7d7", framealpha=0.95, borderpad=0.35)

    root, ext = os.path.splitext(save_path)
    ext = ext.lower() or ".pdf"
    primary_path = root + ext
    fig.savefig(primary_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    companion_path = None
    if save_companion:
        companion_ext = ".png" if ext == ".pdf" else ".pdf"
        companion_path = root + companion_ext
        fig.savefig(companion_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats["figure_path"] = primary_path
    stats["companion_figure_path"] = companion_path
    return stats


def save_dataset_reliability_figure(
    probs,
    labels,
    dataset_name,
    log_dir,
    method_name=None,
    n_bins=15,
    figure_dir=None,
):
    """Save <log_dir>/reliability_figures/<dataset_name>_rf.png."""
    if figure_dir is None:
        figure_dir = os.path.join(log_dir, "reliability_figures")
    figure_path = os.path.join(figure_dir, f"{sanitize_filename(dataset_name)}_rf.png")
    return save_reliability_figure(
        probs,
        labels,
        save_path=figure_path,
        dataset_name=dataset_name,
        method_name=method_name,
        n_bins=n_bins,
        title=dataset_display_name(dataset_name),
        save_companion=False,
    )
