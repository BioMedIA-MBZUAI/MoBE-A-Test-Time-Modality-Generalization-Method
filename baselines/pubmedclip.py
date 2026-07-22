import os
import sys
import csv
import time
import random
import argparse
from datetime import datetime

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor

try:
    import wandb
except ImportError:
    wandb = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC


def get_arguments():
    default_datasets = "chestmnist_224/organcmnist_224/pathmnist_224/dermamnist"
    default_backbone = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    default_dataroot = os.path.join(PROJECT_ROOT, "datasets_all")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-log", dest="wandb", action="store_true")
    parser.add_argument("--datasets", type=str, default=default_datasets)
    parser.add_argument("--data-root", type=str, default=default_dataroot)
    parser.add_argument("--backbone", type=str, default=default_backbone)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--log-dir", type=str, default="./results/pubmedclip")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def build_hf_preprocess(processor: CLIPProcessor):
    image_processor = processor.image_processor
    mean = image_processor.image_mean
    std = image_processor.image_std

    resize_size = 224
    crop_h, crop_w = 224, 224

    size_cfg = getattr(image_processor, "size", None)
    if isinstance(size_cfg, dict):
        if "shortest_edge" in size_cfg:
            resize_size = int(size_cfg["shortest_edge"])
        elif "height" in size_cfg and "width" in size_cfg:
            resize_size = (int(size_cfg["height"]), int(size_cfg["width"]))
    elif isinstance(size_cfg, int):
        resize_size = int(size_cfg)

    crop_cfg = getattr(image_processor, "crop_size", None)
    if isinstance(crop_cfg, dict):
        crop_h = int(crop_cfg.get("height", crop_h))
        crop_w = int(crop_cfg.get("width", crop_w))
    elif isinstance(crop_cfg, int):
        crop_h = crop_w = int(crop_cfg)

    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=BICUBIC),
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def maybe_apply_projection(features, projection, name):
    if features.shape[-1] == projection.in_features:
        return projection(features)
    if features.shape[-1] == projection.out_features:
        return features
    raise RuntimeError(
        f"Unexpected {name} feature dimension {features.shape[-1]}; "
        f"expected {projection.in_features} before projection or {projection.out_features} after projection."
    )


@torch.no_grad()
def get_text_features_hf(clip_model, text_inputs):
    text_features = clip_model.get_text_features(**text_inputs)
    if torch.is_tensor(text_features):
        return text_features

    if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
        return text_features.text_embeds

    if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
        pooled_output = text_features.pooler_output
    elif isinstance(text_features, (tuple, list)) and len(text_features) > 1:
        pooled_output = text_features[1]
    else:
        raise TypeError(f"Unexpected text feature output type: {type(text_features)}")

    if hasattr(clip_model, "text_projection"):
        pooled_output = maybe_apply_projection(pooled_output, clip_model.text_projection, "text")
    return pooled_output


@torch.no_grad()
def get_image_features_hf(clip_model, pixel_values):
    image_features = clip_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(image_features):
        return image_features

    if hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
        return image_features.image_embeds

    if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
        pooled_output = image_features.pooler_output
    elif isinstance(image_features, (tuple, list)) and len(image_features) > 1:
        pooled_output = image_features[1]
    else:
        raise TypeError(f"Unexpected image feature output type: {type(image_features)}")

    if hasattr(clip_model, "visual_projection"):
        pooled_output = maybe_apply_projection(pooled_output, clip_model.visual_projection, "image")
    return pooled_output


@torch.no_grad()
def clip_classifier_hf(classnames, template, clip_model, processor, device):
    clip_weights = []
    text_max_length = int(clip_model.config.text_config.max_position_embeddings)

    for classname in classnames:
        classname = classname.replace("_", " ")
        texts = [prompt.format(classname) for prompt in template]
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        class_embeddings = get_text_features_hf(clip_model, text_inputs)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embedding = F.normalize(class_embeddings.mean(dim=0), dim=0)
        clip_weights.append(class_embedding)

    return torch.stack(clip_weights, dim=1).to(device)


@torch.no_grad()
def get_clip_logits_hf(images, clip_model, clip_weights):
    device = next(clip_model.parameters()).device
    if isinstance(images, list):
        images = torch.cat(images, dim=0)
    images = images.to(device)

    image_features = get_image_features_hf(clip_model, images)
    image_features = F.normalize(image_features, dim=-1)
    clip_logits = 100.0 * image_features @ clip_weights
    return image_features, clip_logits


def _prepare_single_label_targets(target):
    if target.dim() == 0:
        return target.view(1).long()
    if target.dim() == 1:
        return target.long()
    if target.dim() == 2 and target.size(1) == 1:
        return target.squeeze(1).long()
    if target.dim() == 2 and target.size(1) > 1:
        return target.argmax(dim=1).long()
    return target.view(target.size(0), -1).argmax(dim=1).long()


@torch.no_grad()
def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros((), device=probs.device)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        if b == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += prop * torch.abs(acc_bin - conf_bin)
    return 100.0 * ece.item()


@torch.no_grad()
def brier_score(probs, labels):
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.sum((probs - one_hot) ** 2, dim=1).mean().item()


@torch.no_grad()
def negative_log_likelihood(probs, labels):
    return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels, reduction="mean").item()


def log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time_per_sample, run_time, args,
                       compute_stats=None):
    file_exists = os.path.exists(results_path)
    fieldnames = [
        "run_time", "method", "dataset", "accuracy", "ece_15bins", "brier", "nll",
        "avg_time_per_sample", "avg_selected_experts", "image_encoder_gflops",
        "gflops_per_sample", "backbone"
    ]
    compute_stats = compute_stats or {}
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not file_exists) or os.path.getsize(results_path) == 0:
            writer.writeheader()
        writer.writerow({
            "run_time": run_time,
            "method": "PubMedCLIP",
            "dataset": dataset_name,
            "accuracy": float(acc),
            "ece_15bins": float(ece),
            "brier": float(brier),
            "nll": float(nll),
            "avg_time_per_sample": float(avg_time_per_sample),
            "avg_selected_experts": compute_stats.get("avg_selected_experts", ""),
            "image_encoder_gflops": compute_stats.get("image_encoder_gflops", ""),
            "gflops_per_sample": compute_stats.get("gflops_per_sample", ""),
            "backbone": args.backbone,
        })


@torch.no_grad()
def run_pubmedclip_inference(loader, clip_model, clip_weights):
    accuracies = []
    all_probs, all_targets = [], []
    device = clip_weights.device

    for i, (images, target) in enumerate(tqdm(loader, desc="Processed test images: ")):
        _, clip_logits = get_clip_logits_hf(images, clip_model, clip_weights)
        labels = _prepare_single_label_targets(target).to(device)

        acc = cls_acc(clip_logits, labels)
        accuracies.append(acc)

        probs_eval = clip_logits.softmax(dim=-1).detach().cpu()
        labels_eval = labels.detach().cpu()
        all_probs.append(probs_eval)
        all_targets.append(labels_eval)

        if wandb is not None and wandb.run is not None:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % 150 == 0:
            print("---- PubMedCLIP test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    avg_acc = sum(accuracies) / len(accuracies)
    probs_all = torch.cat(all_probs, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    ece = expected_calibration_error(probs_all, targets_all, n_bins=15)
    brier = brier_score(probs_all, targets_all)
    nll = negative_log_likelihood(probs_all, targets_all)

    print("---- PubMedCLIP test accuracy: {:.2f}. ----\n".format(avg_acc))
    print(f"---- PubMedCLIP metrics: Acc={avg_acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} ----")

    if wandb is not None and wandb.run is not None:
        wandb.log({
            "Final accuracy": avg_acc,
            "ECE_15bins": ece,
            "Brier_score": brier,
            "NLL": nll,
        }, commit=True)

    return {
        "acc": avg_acc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "probs": probs_all,
        "labels": targets_all,
    }


def main():
    args = get_arguments()

    if args.wandb and wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --wandb-log.")

    device = args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    clip_model = CLIPModel.from_pretrained(args.backbone).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.backbone, use_fast=True)
    preprocess = build_hf_preprocess(processor)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.log_dir, "results.csv")

    datasets = args.datasets.split("/")
    summary = []
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier_hf(classnames, template, clip_model, processor, device)
        sample_images, _ = next(iter(test_loader))
        compute_stats = compute_single_encoder_stats(
            clip_model,
            sample_images,
            device=device,
            forward_fn=lambda pixel_values: clip_model.get_image_features(pixel_values=pixel_values),
        )
        print(
            f"---- Compute estimate: image encoder={compute_stats['image_encoder_gflops']:.2f} GFLOPs | "
            f"GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={compute_stats['avg_selected_experts']:.2f} ----"
        )

        if args.wandb:
            run = wandb.init(project="ETTA-CLIP", config=vars(args), group=group_name, name=dataset_name)

        start_time = time.perf_counter()
        metrics = run_pubmedclip_inference(test_loader, clip_model, clip_weights)
        end_time = time.perf_counter()

        acc = metrics["acc"]
        ece = metrics["ece"]
        brier = metrics["brier"]
        nll = metrics["nll"]
        avg_time = (end_time - start_time) / max(1, len(test_loader.dataset))

        summary.append((dataset_name, acc, ece, brier, nll, compute_stats))
        log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time, run_time, args, compute_stats)
        fig_stats = save_dataset_reliability_figure(
            metrics["probs"], metrics["labels"], dataset_name, args.log_dir, method_name="PubMedCLIP"
        )
        print(f"---- Saved reliability figure: {fig_stats['figure_path']} ----")

        if args.wandb:
            wandb.log({
                f"{dataset_name}/acc": acc,
                f"{dataset_name}/ece": ece,
                f"{dataset_name}/brier": brier,
                f"{dataset_name}/nll": nll,
            })
            run.finish()

    if summary:
        print("\nFinal metrics:")
        for name, acc, ece, brier, nll, compute_stats in summary:
            print(
                f"- {name}: Acc={acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} "
                f"| GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={compute_stats['avg_selected_experts']:.2f}"
            )


if __name__ == "__main__":
    main()
