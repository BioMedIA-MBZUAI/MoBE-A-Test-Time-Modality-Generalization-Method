# mome_entropy_tta.py
# MoME-style Test-Time Adaptation via Entropy Minimization on Mixture Weights
#
# Assumptions:
# - You have 5 expert checkpoints: experts_dir/expert_{Modality}_0.pt
# - utils.py provides:
#   - get_config_file(config_dir, dataset_name)
#   - build_test_data_loader(dataset_name, data_root, preprocess) -> (loader, classnames, template)
#   - clip_classifier(classnames, template, base_model, tokenizer) -> clip_weights [D,C]
#   - get_clip_logits(images, model, clip_weights) -> (features [B,D], logits [B,C], loss, prob_map, extra)
#   - cls_acc(logits, target) -> float accuracy for batch
#   - init_log(results_path), log_results(...)
#
# Key idea:
# - Freeze BiomedCLIP + expert MLPs
# - Per test sample, optimize mixture weights w over experts by minimizing entropy of p(y|x)
# - Online mode: warm-start gating from previous step (optional)

import os
import sys
import csv
import time
import random
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *


# ----------------------------
# Args
# ----------------------------
def get_arguments():
    default_config = os.path.join(PROJECT_ROOT, "configs")
    # 'hardbench_btmri', 'hardbench_busi', 'hardbench_chmnist', 'hardbench_ctkidney', 'hardbench_covid19', 'hardbench_dermamnist', 'hardbench_kneexray', 'hardbench_kvasir', 'hardbench_lungcolon', 'hardbench_retina', 'hardbench_octmnist'
    default_datasets = "hardbench_btmri/hardbench_covid19/hardbench_ctkidney/hardbench_kvasir/hardbench_chmnist/hardbench_lungcolon/hardbench_retina/hardbench_kneexray/hardbench_busi"
    default_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    default_dataroot = os.path.join(PROJECT_ROOT, "datasets_all")
    default_experts_dir = os.path.join(PROJECT_ROOT, "experts")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--wandb-log", dest="wandb", action="store_true")
    parser.add_argument("--datasets", type=str, default=default_datasets,
                        help="Datasets separated by '/'. Example: chestmnist_224/breastmnist_224")
    parser.add_argument("--data-root", type=str, default=default_dataroot)
    parser.add_argument("--backbone", type=str, default=default_backbone)
    parser.add_argument("--pretrained", type=str, default=None)

    # Experts
    parser.add_argument("--experts-dir", type=str, default=default_experts_dir)
    parser.add_argument("--modalities", type=str, default="Angiogram,CT,MRI,Ultrasound,Xray",
                        help="Comma-separated modalities (must match expert filenames).")

    # Logging
    parser.add_argument("--log-dir", type=str, default="./results/mome")

    # MoME entropy-min routing optimization
    parser.add_argument("--tta-steps", type=int, default=5,
                        help="Number of gradient steps per test sample for gating optimization.")
    parser.add_argument("--tta-lr", type=float, default=5e-2,
                        help="Learning rate for gating optimization.")
    parser.add_argument("--tta-wd", type=float, default=0.0,
                        help="Weight decay for gating optimizer (usually 0).")
    parser.add_argument("--gating-temp", type=float, default=1.0,
                        help="Softmax temperature for gating weights.")
    parser.add_argument("--entropy-temp", type=float, default=1.0,
                        help="Optional temperature on class logits before entropy.")
    parser.add_argument("--warm-start", action="store_true",
                        help="Warm-start gating parameters from previous sample (online).")
    parser.add_argument("--gate-reg", type=float, default=0.0,
                        help="L2 regularization on gating logits (stabilizes updates).")
    parser.add_argument("--gate-kl", type=float, default=0.0,
                        help="KL(w || w_prev) penalty weight for temporal stability (requires warm-start).")
    parser.add_argument("--topk", type=int, default=0,
                        help="If >0, restrict mixture to top-k experts by initial entropy (speed). 0 = use all experts.")

    # Eval
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


# ----------------------------
# Expert MLP loading
# ----------------------------
def load_expert_mlp(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = len([k for k in sd.keys() if k in model.state_dict()])
    return loaded, missing, unexpected


def build_expert_models(model_name: str, pretrained: str, device: str, experts_dir: str, modalities: list):
    expert_models = []
    for m in modalities:
        model = open_clip.create_model(model_name, pretrained=pretrained).to(device).eval()
        ckpt = os.path.join(experts_dir, f"expert_{m}_0.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing expert checkpoint: {ckpt}")
        loaded, missing, unexpected = load_expert_mlp(model, ckpt)
        print(f"[Expert:{m}] loaded_keys={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
        expert_models.append(model)
    return expert_models


# ----------------------------
# MoME entropy-min gating
# ----------------------------
def entropy_of_probs(p: torch.Tensor) -> torch.Tensor:
    # p: [B,C]
    return -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)  # [B]


def _prepare_single_label_targets(target):
    """
    Converts common target formats to class indices.
    - [B] or [B,1] class ids -> [B]
    - [B,C] one-hot / multi-hot -> argmax labels
    """
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
    """ECE in percentage points. Lower is better."""
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
    """Multiclass Brier score. Lower is better."""
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.sum((probs - one_hot) ** 2, dim=1).mean().item()


@torch.no_grad()
def negative_log_likelihood(probs, labels):
    """NLL. Lower is better."""
    return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels, reduction="mean").item()


def log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time_per_sample, args, run_time,
                       compute_stats=None):
    """Append accuracy + calibration/probabilistic error metrics to CSV."""
    file_exists = os.path.exists(results_path)
    fieldnames = [
        "run_time", "dataset", "accuracy", "ece_15bins", "brier", "nll",
        "avg_time_per_sample", "avg_selected_experts", "image_encoder_gflops",
        "gflops_per_sample", "tta_steps", "tta_lr", "gating_temp",
        "entropy_temp", "warm_start", "gate_reg", "gate_kl", "topk", "seed"
    ]
    compute_stats = compute_stats or {}
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not file_exists) or os.path.getsize(results_path) == 0:
            writer.writeheader()
        writer.writerow({
            "run_time": run_time,
            "dataset": dataset_name,
            "accuracy": float(acc),
            "ece_15bins": float(ece),
            "brier": float(brier),
            "nll": float(nll),
            "avg_time_per_sample": float(avg_time_per_sample),
            "avg_selected_experts": compute_stats.get("avg_selected_experts", ""),
            "image_encoder_gflops": compute_stats.get("image_encoder_gflops", ""),
            "gflops_per_sample": compute_stats.get("gflops_per_sample", ""),
            "tta_steps": int(args.tta_steps),
            "tta_lr": float(args.tta_lr),
            "gating_temp": float(args.gating_temp),
            "entropy_temp": float(args.entropy_temp),
            "warm_start": bool(args.warm_start),
            "gate_reg": float(args.gate_reg),
            "gate_kl": float(args.gate_kl),
            "topk": int(args.topk),
            "seed": int(args.seed),
        })


@torch.no_grad()
def initial_entropy_ranking(expert_logits):
    # expert_logits: list of [B,C]
    # returns entropies [E], low is better
    E = len(expert_logits)
    ent = []
    for e in range(E):
        p = expert_logits[e].softmax(dim=-1)
        ent.append(entropy_of_probs(p).mean())
    return torch.stack(ent, dim=0)  # [E]


def optimize_gating_weights(
    expert_logits: torch.Tensor,   # [E,C] (B=1 assumed; you can extend to [B,E,C])
    steps: int,
    lr: float,
    wd: float,
    gating_temp: float,
    entropy_temp: float,
    gate_reg: float,
    gate_kl: float,
    w_prev: torch.Tensor | None,
    device: str,
):
    """
    Optimize gating logits g (E,) to minimize entropy of mixed prediction:
      w = softmax(g / gating_temp)
      s_mix = sum_e w_e * s_e
      p = softmax(s_mix / entropy_temp)
      L = H(p) + gate_reg*||g||^2 + gate_kl*KL(w || w_prev)
    """
    E, C = expert_logits.shape
    g = torch.zeros(E, device=device, requires_grad=True)

    # Warm-start by inverse of previous weights if provided (or directly via logits)
    if w_prev is not None:
        # convert w_prev -> logits for warm start
        g0 = (w_prev.clamp_min(1e-12)).log()
        g = g0.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([g], lr=lr, weight_decay=wd)

    w_prev_detached = None
    if w_prev is not None:
        w_prev_detached = w_prev.detach().clamp_min(1e-12)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)

        w = torch.softmax(g / gating_temp, dim=0)  # [E]
        s_mix = (w[:, None] * expert_logits).sum(dim=0, keepdim=True)  # [1,C]
        p = torch.softmax(s_mix / entropy_temp, dim=-1)  # [1,C]
        loss = entropy_of_probs(p).mean()

        if gate_reg > 0:
            loss = loss + gate_reg * (g * g).mean()

        if gate_kl > 0 and w_prev_detached is not None:
            # KL(w || w_prev)
            loss = loss + gate_kl * (w * (w.clamp_min(1e-12).log() - w_prev_detached.log())).sum()

        loss.backward()
        opt.step()

    with torch.no_grad():
        w_final = torch.softmax(g / gating_temp, dim=0)
        s_mix = (w_final[:, None] * expert_logits).sum(dim=0, keepdim=True)  # [1,C]
    return w_final.detach(), s_mix.detach()


# ----------------------------
# Evaluation loop
# ----------------------------
# @torch.no_grad()
def run_test_mome_entropy(loader, expert_models, clip_weights, args):
    device = clip_weights.device
    E = len(expert_models)
    accuracies = []
    all_probs = []
    all_targets = []
    selected_expert_counts = []

    w_prev = None  # for warm-start temporal stability

    for i, (images, target) in enumerate(tqdm(loader, desc="MoME Entropy TTA")):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward all experts (frozen)
        with torch.no_grad():
            expert_logits = []
            for e in range(E):
                _, logits_e, _, _, _ = get_clip_logits(images, expert_models[e], clip_weights)  # logits: [B,C]
                expert_logits.append(logits_e)

        # Stack logits for B=1 (your setting)
        # expert_logits_mat: [E,C]
        expert_logits_mat = torch.cat([le for le in expert_logits], dim=0)  # [E,C] because B=1

        # Optional: restrict to top-k by *initial* entropy to speed up optimization
        if args.topk and args.topk > 0 and args.topk < E:
            ent = initial_entropy_ranking(expert_logits)  # [E]
            _, idx = torch.sort(ent, descending=False)
            sel = idx[:args.topk]
            expert_logits_sel = expert_logits_mat[sel]  # [k,C]
            selected_expert_counts.append(int(sel.numel()))

            # optimize on subset
            w_sub_prev = None
            if args.warm_start and (w_prev is not None):
                w_sub_prev = w_prev[sel]
                w_sub_prev = w_sub_prev / w_sub_prev.sum()

            w_sub, s_mix_sub = optimize_gating_weights(
                expert_logits=expert_logits_sel,
                steps=args.tta_steps,
                lr=args.tta_lr,
                wd=args.tta_wd,
                gating_temp=args.gating_temp,
                entropy_temp=args.entropy_temp,
                gate_reg=args.gate_reg,
                gate_kl=args.gate_kl,
                w_prev=w_sub_prev,
                device=device,
            )

            # lift weights back to E
            w = torch.zeros(E, device=device)
            w[sel] = w_sub
            s_mix = torch.zeros_like(expert_logits_mat[0:1])  # [1,C]
            for j, eidx in enumerate(sel):
                s_mix += w_sub[j] * expert_logits_mat[eidx:eidx+1]

        else:
            selected_expert_counts.append(E)
            # optimize on all experts
            w, s_mix = optimize_gating_weights(
                expert_logits=expert_logits_mat,
                steps=args.tta_steps,
                lr=args.tta_lr,
                wd=args.tta_wd,
                gating_temp=args.gating_temp,
                entropy_temp=args.entropy_temp,
                gate_reg=args.gate_reg,
                gate_kl=args.gate_kl,
                w_prev=(w_prev if args.warm_start else None),
                device=device,
            )

        # Evaluate
        acc = cls_acc(s_mix, target)
        accuracies.append(acc)

        with torch.no_grad():
            probs_eval = torch.softmax(s_mix / args.entropy_temp, dim=-1).detach().cpu()
            labels_eval = _prepare_single_label_targets(target.detach()).cpu()
            all_probs.append(probs_eval)
            all_targets.append(labels_eval)

        # Update warm-start state
        if args.warm_start:
            w_prev = w.detach()

        # Optional wandb
        if wandb.run is not None:
            # report mean entropy after adaptation
            p = torch.softmax(s_mix / args.entropy_temp, dim=-1)
            H = float(entropy_of_probs(p).mean().item())
            wandb.log({
                "Averaged test accuracy": sum(accuracies) / len(accuracies),
                "mome_entropy": H,
                "mome_w_max": float(w.max().item()),
                "mome_w_entropy": float((-(w * (w.clamp_min(1e-12)).log())).sum().item()),
            }, commit=True)

        if i % 150 == 0:
            print(f"[MoME] step={i} avg_acc={sum(accuracies)/len(accuracies):.2f} w_top={float(w.max().item()):.3f}")

    avg_acc = sum(accuracies) / len(accuracies)
    probs_all = torch.cat(all_probs, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    ece = expected_calibration_error(probs_all, targets_all, n_bins=15)
    brier = brier_score(probs_all, targets_all)
    nll = negative_log_likelihood(probs_all, targets_all)

    print(f"---- MoME avg acc: {avg_acc:.2f} ----")
    print(f"---- Metrics: ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} ----")
    avg_selected_experts = sum(selected_expert_counts) / max(1, len(selected_expert_counts))
    print(f"---- Avg selected experts: {avg_selected_experts:.2f} ----")

    if wandb.run is not None:
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
        "avg_selected_experts": avg_selected_experts,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    args = get_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.backbone
    if model_name == "ViT-B/16":
        model_name = "ViT-B-16"

    pretrained = args.pretrained
    if pretrained is None:
        pretrained = None if model_name.startswith("hf-hub:") else "openai"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Base model for text weights + preprocess/tokenizer
    base_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    base_model = base_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Experts
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    expert_models = build_expert_models(model_name, pretrained, device, args.experts_dir, modalities)

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.log_dir, "results.csv")
    init_log(results_path)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    datasets = args.datasets.split("/")
    summary = []

    for dataset_name in datasets:
        print(f"\n[Dataset] {dataset_name}")
        cfg = get_config_file(args.config, dataset_name)
        print(cfg)

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, base_model, tokenizer).to(device)
        sample_images, _ = next(iter(test_loader))
        image_encoder_gflops = estimate_image_encoder_gflops(
            expert_models[0], tuple(sample_images.shape[1:]), device=device
        )
        compute_stats = {
            "avg_selected_experts": "",
            "image_encoder_gflops": image_encoder_gflops,
            "gflops_per_sample": image_encoder_gflops * len(expert_models),
        }
        print(
            f"---- Compute estimate: image encoder={compute_stats['image_encoder_gflops']:.2f} GFLOPs | "
            f"GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} ----"
        )

        if args.wandb:
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=f"MoME_{dataset_name}")

        start = time.perf_counter()
        metrics = run_test_mome_entropy(test_loader, expert_models, clip_weights, args)
        acc = metrics["acc"]
        ece = metrics["ece"]
        brier = metrics["brier"]
        nll = metrics["nll"]
        end = time.perf_counter()

        num_samples = len(test_loader.dataset)
        avg_time = (end - start) / max(1, num_samples)
        compute_stats["avg_selected_experts"] = metrics.get("avg_selected_experts", "")

        summary.append((dataset_name, acc, ece, brier, nll, compute_stats))
        log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time, args, run_time, compute_stats)
        fig_stats = save_dataset_reliability_figure(
            metrics["probs"], metrics["labels"], dataset_name, args.log_dir, method_name="MoME"
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

    print("\nFinal metrics:")
    for name, acc, ece, brier, nll, compute_stats in summary:
        print(
            f"- {name}: Acc={acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} "
            f"| GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={float(compute_stats['avg_selected_experts']):.2f}"
        )


if __name__ == "__main__":
    main()
