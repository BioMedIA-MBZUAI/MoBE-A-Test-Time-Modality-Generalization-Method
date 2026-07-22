import argparse
import os
import random
import time
from datetime import datetime

import open_clip
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from utils import *


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_arguments():
    default_config = os.path.join(PROJECT_ROOT, "configs")
    default_datasets = "dermamnist"
    default_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    default_dataroot = os.path.join(PROJECT_ROOT, "datasets_all")
    default_experts_dir = os.path.join(PROJECT_ROOT, "experts")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--wandb-log", dest="wandb", action="store_true")
    parser.add_argument(
        "--datasets",
        type=str,
        default=default_datasets,
        help="Datasets separated by '/'. Example: chestmnist_224/breastmnist_224",
    )
    parser.add_argument("--data-root", type=str, default=default_dataroot)
    parser.add_argument("--backbone", type=str, default=default_backbone)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument(
        "--experts-dir",
        type=str,
        default=default_experts_dir,
        help="Folder containing expert_Angiogram_0.pt, expert_CT_0.pt, ...",
    )
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "results", "mobe_orig"))
    parser.add_argument(
        "--modalities",
        type=str,
        default="Angiogram,CT,MRI,Ultrasound,Xray",
        help="Comma-separated modalities matching expert checkpoint filenames.",
    )
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--hard-mix", default=True)
    parser.add_argument("--mix-strategy", type=str, default="exp", choices=["exp", "softmax"])
    parser.add_argument("--gap-thr", type=float, default=0.07)
    return parser.parse_args()


def _cfg_value(cfg, *names):
    for name in names:
        if name in cfg:
            return cfg[name]
    raise KeyError(f"Missing required MOBE config key. Expected one of: {', '.join(names)}")


def _cfg_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def load_mobe_config(cfg, dataset_name):
    if not isinstance(cfg, dict):
        raise ValueError(f"Config for {dataset_name} must be a dictionary.")

    mobe_cfg = cfg.get("mobe")
    if mobe_cfg is None:
        mobe_cfg = cfg.get("MOBE")
    if mobe_cfg is None:
        raise KeyError(f"Config for {dataset_name} must contain a 'mobe' section.")

    return {
        "use_mobe": _cfg_bool(_cfg_value(mobe_cfg, "use_mobe", "use_MOBE")),
        "route_with_mobe": _cfg_bool(_cfg_value(mobe_cfg, "route_with_mobe", "route_with_MOBE")),
        "thr1": float(_cfg_value(mobe_cfg, "thr1", "MOBE_thr1")),
        "c1": int(_cfg_value(mobe_cfg, "c1", "MOBE_c1")),
        "thr2": float(_cfg_value(mobe_cfg, "thr2", "MOBE_thr2")),
        "c2": int(_cfg_value(mobe_cfg, "c2", "MOBE_c2")),
        "temp": float(_cfg_value(mobe_cfg, "temp", "MOBE_temp")),
        "lambda": float(_cfg_value(mobe_cfg, "lambda", "MOBE_lambda")),
    }


def mobe_init(clip_weights, num_experts, init_c1, init_c2, device):
    mu0 = clip_weights.t().to(device)
    mu = mu0.unsqueeze(0).repeat(num_experts, 1, 1).contiguous()
    pi = torch.eye(mu0.size(0), device=device).unsqueeze(0).repeat(num_experts, 1, 1)
    c1 = torch.full((num_experts, mu0.size(0)), float(init_c1), device=device)
    c2 = torch.full((num_experts, mu0.size(0)), float(init_c2), device=device)
    return mu, pi, c1, c2


def mobe_forward_and_update(feature, expert_idx, mu, pi, c1, c2, thr1, thr2, temp):
    cluster_logits = temp * (feature @ mu[expert_idx].t())
    cluster_probs = cluster_logits.softmax(dim=-1)
    mobe_probs = cluster_probs @ pi[expert_idx]
    confidence, prediction = mobe_probs.max(dim=-1)
    confidence_value = float(confidence.mean().item())
    predicted_class = int(prediction[0].item())

    if confidence_value > thr1:
        mu[expert_idx, predicted_class] = (
            c1[expert_idx, predicted_class] * mu[expert_idx, predicted_class] + feature[0]
        ) / (c1[expert_idx, predicted_class] + 1.0)
        mu[expert_idx, predicted_class] = F.normalize(mu[expert_idx, predicted_class], dim=0)
        c1[expert_idx, predicted_class] += 1.0

    if confidence_value > thr2:
        pi[expert_idx, predicted_class] = (
            c2[expert_idx, predicted_class] * pi[expert_idx, predicted_class] + mobe_probs[0]
        ) / (c2[expert_idx, predicted_class] + 1.0)
        c2[expert_idx, predicted_class] += 1.0

    return mobe_probs, confidence_value


def load_expert_mlp(model: torch.nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len([key for key in state_dict.keys() if key in model.state_dict()])
    return loaded, missing, unexpected


def build_expert_models(model_name: str, pretrained: str, device: str, experts_dir: str, modalities: list):
    expert_models = []
    for modality in modalities:
        model = open_clip.create_model(model_name, pretrained=pretrained).to(device).eval()
        ckpt_path = os.path.join(experts_dir, f"expert_{modality}_0.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing expert checkpoint: {ckpt_path}")
        loaded, missing, unexpected = load_expert_mlp(model, ckpt_path)
        print(f"[Expert:{modality}] loaded_keys={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
        expert_models.append(model)
    return expert_models


@torch.no_grad()
def run_test_moe_mobe(loader, expert_models, clip_weights, args, mobe_cfg):
    device = clip_weights.device
    accuracies = []
    num_experts = len(expert_models)
    use_mobe = bool(mobe_cfg["use_mobe"])

    if use_mobe:
        mu, pi, c1, c2 = mobe_init(
            clip_weights=clip_weights,
            num_experts=num_experts,
            init_c1=mobe_cfg["c1"],
            init_c2=mobe_cfg["c2"],
            device=device,
        )

    for i, (images, target) in enumerate(tqdm(loader, desc="Processed test images")):
        target = target.to(device, non_blocking=True)
        expert_features = []
        expert_logits = []
        entropies = torch.empty(num_experts, device=device)

        if use_mobe:
            mobe_confidences = torch.empty(num_experts, device=device)

        for expert_idx, expert_model in enumerate(expert_models):
            features, logits, _, _, _ = get_clip_logits(images, expert_model, clip_weights)
            features = F.normalize(features, dim=-1)
            expert_features.append(features)
            expert_logits.append(logits)

            probs = logits.softmax(dim=-1)
            entropies[expert_idx] = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()

            if use_mobe:
                _, confidence = mobe_forward_and_update(
                    feature=features,
                    expert_idx=expert_idx,
                    mu=mu,
                    pi=pi,
                    c1=c1,
                    c2=c2,
                    thr1=mobe_cfg["thr1"],
                    thr2=mobe_cfg["thr2"],
                    temp=mobe_cfg["temp"],
                )
                mobe_confidences[expert_idx] = confidence

        max_k = min(args.topk, num_experts)
        if use_mobe and mobe_cfg["route_with_mobe"]:
            sorted_scores, sorted_indices = torch.sort(mobe_confidences, descending=True)
            best_score = sorted_scores[0].item()
            topk_experts = [int(sorted_indices[0].item())]
            for idx in range(1, max_k):
                if (best_score - sorted_scores[idx].item()) < args.gap_thr:
                    topk_experts.append(int(sorted_indices[idx].item()))
                else:
                    break
        else:
            sorted_scores, sorted_indices = torch.sort(entropies, descending=False)
            best_score = sorted_scores[0].item()
            topk_experts = [int(sorted_indices[0].item())]
            for idx in range(1, max_k):
                if (sorted_scores[idx].item() - best_score) < args.gap_thr:
                    topk_experts.append(int(sorted_indices[idx].item()))
                else:
                    break

        if len(topk_experts) == 1 and num_experts > 1:
            topk_experts.append(int(sorted_indices[1].item()))

        weights = torch.zeros(num_experts, device=device)
        if args.hard_mix:
            for expert_idx in topk_experts:
                weights[expert_idx] = 1.0 / len(topk_experts)
        elif args.mix_strategy == "softmax":
            top_scores = (-entropies[topk_experts]) / args.tau
            weights[topk_experts] = torch.softmax(top_scores, dim=0)
        else:
            top_scores = 1.0 / (torch.exp(entropies[topk_experts]) + 1e-12)
            weights[topk_experts] = top_scores / top_scores.sum()

        final_logits = torch.zeros_like(expert_logits[0])
        for expert_idx in topk_experts:
            final_logits += weights[expert_idx] * expert_logits[expert_idx]

        if use_mobe and mobe_cfg["lambda"] > 0:
            mobe_mix_probs = 0.0
            for expert_idx in topk_experts:
                cluster_logits = mobe_cfg["temp"] * (expert_features[expert_idx] @ mu[expert_idx].t())
                cluster_probs = cluster_logits.softmax(dim=-1)
                expert_probs = cluster_probs @ pi[expert_idx]
                mobe_mix_probs = mobe_mix_probs + weights[expert_idx] * expert_probs
            mobe_logits = (mobe_mix_probs + 1e-12).log()
            final_logits = (1.0 - mobe_cfg["lambda"]) * final_logits + mobe_cfg["lambda"] * mobe_logits

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)

        if wandb.run is not None:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % 150 == 0:
            print(f"---- DynK MOBE-Router avg acc: {sum(accuracies) / len(accuracies):.2f} ----")

    avg_acc = sum(accuracies) / len(accuracies)
    print(f"---- DynK MOBE-Router avg acc: {avg_acc:.2f} ----")
    return avg_acc


def main():
    args = get_arguments()
    model_name = "ViT-B-16" if args.backbone == "ViT-B/16" else args.backbone
    pretrained = args.pretrained
    if pretrained is None:
        pretrained = None if model_name.startswith("hf-hub:") else "openai"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    base_model = base_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    modalities = [modality.strip() for modality in args.modalities.split(",") if modality.strip()]
    expert_models = build_expert_models(model_name, pretrained, device, args.experts_dir, modalities)

    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.log_dir, "results.csv")
    init_log(results_path)

    summary = []
    for dataset_name in args.datasets.split("/"):
        print(f"\nProcessing {dataset_name} dataset.")

        cfg = get_config_file(args.config, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        mobe_cfg = load_mobe_config(cfg, dataset_name)
        print("Running MOBE config:")
        print(mobe_cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, base_model, tokenizer).to(device)

        if args.wandb:
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=dataset_name)

        start_time = time.perf_counter()
        acc = run_test_moe_mobe(test_loader, expert_models, clip_weights, args=args, mobe_cfg=mobe_cfg)
        end_time = time.perf_counter()

        avg_time_per_sample = (end_time - start_time) / max(1, len(test_loader.dataset))
        summary.append((dataset_name, acc))
        log_results(results_path, dataset_name, acc, avg_time_per_sample, args, run_time)

        if args.wandb:
            wandb.log({f"{dataset_name}/acc": acc})
            run.finish()

    print("\nFinal accuracies:")
    for dataset_name, acc in summary:
        print(f"- {dataset_name}: {acc:.2f}")


if __name__ == "__main__":
    main()
