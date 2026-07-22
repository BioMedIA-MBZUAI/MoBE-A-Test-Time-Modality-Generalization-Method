import argparse
import csv
import os
import sys
import random
import time
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import open_clip
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.utils import AugMixAugmenter
from utils import build_test_data_loader, cls_acc, save_dataset_reliability_figure


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_arguments():
    # 'hardbench_btmri', 'hardbench_busi', 'hardbench_chmnist', 'hardbench_ctkidney', 'hardbench_covid19', 'hardbench_dermamnist', 'hardbench_kneexray', 'hardbench_kvasir', 'hardbench_lungcolon', 'hardbench_retina', 'hardbench_octmnist'
    default_datasets = "hardbench_btmri/hardbench_covid19/hardbench_ctkidney/hardbench_kvasir/hardbench_chmnist/hardbench_lungcolon/hardbench_retina/hardbench_kneexray/hardbench_busi"
    default_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    default_dataroot = os.path.join(PROJECT_ROOT, "datasets_all")
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Test-time Prompt Tuning baseline for local medical datasets.")
    parser.add_argument("--wandb-log", dest="wandb", action="store_true")
    parser.add_argument("--datasets", type=str, default=default_datasets,
                        help="Datasets separated by '/'. Example: hardbench_btmri/hardbench_busi")
    parser.add_argument("--data-root", type=str, default=default_dataroot)
    parser.add_argument("--backbone", type=str, default=default_backbone)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--log-dir", type=str, default="./results/tpt")

    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of TPT views per image, including the clean view.")
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--selection-p", type=float, default=0.1,
                        help="Fraction of lowest-entropy augmented views used for adaptation.")
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument("--ctx-init", type=str, default="a medical image of",
                        help="Text used to initialize the learnable context tokens.")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_input_ids(tokens):
    if isinstance(tokens, Mapping):
        return tokens["input_ids"]
    return tokens


def _as_attention_mask(tokens, pad_token_id=0):
    if isinstance(tokens, Mapping) and "attention_mask" in tokens:
        return tokens["attention_mask"]
    input_ids = _as_input_ids(tokens)
    return (input_ids != pad_token_id).long()


def _tokenize(tokenizer, texts, device):
    tokens = tokenizer(texts)
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, Mapping):
        return {k: v.to(device) for k, v in tokens.items()}
    return tokens


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    n_selected = max(1, int(batch_entropy.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=False)[:n_selected]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    avg_logits = torch.clamp(avg_logits, min=torch.finfo(avg_logits.dtype).min)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


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
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    for b in range(n_bins):
        lo, hi = boundaries[b], boundaries[b + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if b == 0 else (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            ece += prop * torch.abs(accuracies[in_bin].float().mean() - confidences[in_bin].mean())
    return 100.0 * ece.item()


@torch.no_grad()
def brier_score(probs, labels):
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.sum((probs - one_hot) ** 2, dim=1).mean().item()


@torch.no_grad()
def negative_log_likelihood(probs, labels):
    return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels, reduction="mean").item()


class CLIPSoftPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, n_ctx, ctx_init):
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.n_cls = len(classnames)
        self.dtype = clip_model.token_embedding.weight.dtype
        device = next(clip_model.parameters()).device
        ctx_dim = clip_model.token_embedding.weight.shape[1]

        ctx = torch.empty(n_ctx, ctx_dim, dtype=self.dtype, device=device)
        nn.init.normal_(ctx, std=0.02)
        if ctx_init:
            init_tokens = _tokenize(tokenizer, [ctx_init], device)
            init_ids = _as_input_ids(init_tokens)
            with torch.no_grad():
                init_emb = clip_model.token_embedding(init_ids).type(self.dtype)
            eot = int(init_ids[0].argmax().item())
            usable = init_emb[0, 1:min(eot, 1 + n_ctx), :]
            ctx[:usable.size(0)] = usable

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name.replace('_', ' ')}." for name in classnames]
        tokenized = _tokenize(tokenizer, prompts, device)
        token_ids = _as_input_ids(tokenized)
        with torch.no_grad():
            embedding = clip_model.token_embedding(token_ids).type(self.dtype)

        self.ctx = nn.Parameter(ctx)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", token_ids)
        self.register_buffer("initial_ctx", ctx.detach().clone())

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.initial_ctx)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return encode_clip_text_from_embeddings(self.clip_model, prompts, self.tokenized_prompts)


class HFSoftPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, n_ctx, ctx_init):
        super().__init__()
        self.clip_model = clip_model
        self.text = clip_model.text
        self.n_ctx = n_ctx
        self.n_cls = len(classnames)
        device = next(clip_model.parameters()).device

        embeddings = self.text.transformer.get_input_embeddings()
        ctx_dim = embeddings.embedding_dim
        dtype = embeddings.weight.dtype
        ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(ctx, std=0.02)
        if ctx_init:
            init_tokens = _tokenize(tokenizer, [ctx_init], device)
            init_ids = _as_input_ids(init_tokens)
            init_mask = _as_attention_mask(init_tokens, getattr(self.text.config, "pad_token_id", 0))
            with torch.no_grad():
                init_emb = embeddings(init_ids).to(dtype=dtype)
            valid = torch.nonzero(init_mask[0], as_tuple=False).flatten()
            usable_positions = valid[1:1 + n_ctx] if valid.numel() > 2 else valid[:n_ctx]
            usable = init_emb[0, usable_positions, :]
            ctx[:usable.size(0)] = usable

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name.replace('_', ' ')}." for name in classnames]
        tokenized = _tokenize(tokenizer, prompts, device)
        self.tokenized = tokenized
        self.pad_token_id = getattr(self.text.config, "pad_token_id", 0)
        self.ctx = nn.Parameter(ctx)
        self.register_buffer("initial_ctx", ctx.detach().clone())

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.initial_ctx)

    def forward(self):
        return encode_hf_text_with_soft_prompt(
            self.clip_model,
            self.tokenized,
            self.ctx,
            self.n_ctx,
            self.pad_token_id,
        )


def encode_clip_text_from_embeddings(model, prompt_embeddings, tokenized_prompts):
    dtype = prompt_embeddings.dtype
    x = prompt_embeddings + model.positional_embedding.to(device=prompt_embeddings.device, dtype=dtype)
    x = x.permute(1, 0, 2)
    attn_mask = getattr(model, "attn_mask", None)
    try:
        x = model.transformer(x, attn_mask=attn_mask)
    except TypeError:
        x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).to(dtype)
    eot_indices = tokenized_prompts.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
    text_projection = getattr(model, "text_projection", None)
    if text_projection is not None:
        x = x @ text_projection
    return F.normalize(x, dim=-1)


def _apply_projection(proj, x):
    if proj is None:
        return x
    if isinstance(proj, nn.Module):
        return proj(x)
    return x @ proj


def encode_hf_text_with_soft_prompt(model, tokenized, ctx, n_ctx, pad_token_id):
    text = model.text
    input_ids = _as_input_ids(tokenized)
    attention_mask = _as_attention_mask(tokenized, pad_token_id)
    embeddings = text.transformer.get_input_embeddings()
    inputs_embeds = embeddings(input_ids).to(dtype=ctx.dtype)
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[:, 1:1 + n_ctx, :] = ctx.unsqueeze(0).expand(input_ids.size(0), -1, -1)

    outputs = text.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    pooled = text.pooler(outputs, attention_mask) if hasattr(text, "pooler") else outputs.last_hidden_state[:, 0]
    pooled = _apply_projection(getattr(text, "proj", None), pooled)
    return F.normalize(pooled, dim=-1)


def build_prompt_learner(classnames, clip_model, tokenizer, n_ctx, ctx_init):
    if hasattr(clip_model, "token_embedding") and hasattr(clip_model, "positional_embedding"):
        return CLIPSoftPromptLearner(classnames, clip_model, tokenizer, n_ctx, ctx_init)
    if hasattr(clip_model, "text") and hasattr(clip_model.text, "transformer"):
        return HFSoftPromptLearner(classnames, clip_model, tokenizer, n_ctx, ctx_init)
    raise TypeError("This open_clip text tower does not expose a supported prompt-tuning interface.")


def extract_normalize(preprocess):
    if hasattr(preprocess, "transforms"):
        for transform in preprocess.transforms:
            if isinstance(transform, transforms.Normalize):
                return transform
    return transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )


def build_tpt_preprocess(preprocess, resolution, batch_size):
    normalize = extract_normalize(preprocess)
    base_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
    ])
    image_preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return AugMixAugmenter(base_transform, image_preprocess, n_views=max(1, batch_size - 1), augmix=True)


def prepare_views(images, device):
    if isinstance(images, list):
        views = [view.to(device, non_blocking=True) for view in images]
        return torch.cat(views, dim=0), views[0]
    images = images.to(device, non_blocking=True)
    if images.dim() > 4 and images.size(0) == 1:
        images = images.squeeze(0)
    return images, images[:1]


def get_logit_scale(clip_model):
    if hasattr(clip_model, "logit_scale"):
        return clip_model.logit_scale.exp()
    return torch.tensor(100.0, device=next(clip_model.parameters()).device)


def compute_logits(clip_model, images, text_features):
    image_features = clip_model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)
    return get_logit_scale(clip_model) * image_features @ text_features.t()


def run_tpt(loader, clip_model, prompt_learner, args):
    device = next(clip_model.parameters()).device
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=args.lr)
    optim_state = optimizer.state_dict()
    accuracies = []
    all_probs, all_targets = [], []

    clip_model.eval()
    for i, (images, target) in enumerate(tqdm(loader, desc="Processed test images")):
        target = _prepare_single_label_targets(target).to(device)
        image_batch, clean_image = prepare_views(images, device)

        prompt_learner.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_batch)
            image_features = F.normalize(image_features, dim=-1)

        selected_idx = None
        for _ in range(args.tta_steps):
            text_features = prompt_learner()
            logits = get_logit_scale(clip_model) * image_features @ text_features.t()
            if selected_idx is None:
                selected_logits, selected_idx = select_confident_samples(logits, args.selection_p)
            else:
                selected_logits = logits[selected_idx]
            loss = avg_entropy(selected_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            text_features = prompt_learner()
            final_logits = compute_logits(clip_model, clean_image, text_features)
            probs = final_logits.softmax(dim=-1)
            accuracies.append(cls_acc(final_logits, target))
            all_probs.append(probs.detach().cpu())
            all_targets.append(target.detach().cpu())

        if wandb.run is not None:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % 150 == 0:
            print("---- TPT's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    avg_acc = sum(accuracies) / len(accuracies)
    probs_all = torch.cat(all_probs, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    ece = expected_calibration_error(probs_all, targets_all, n_bins=15)
    brier = brier_score(probs_all, targets_all)
    nll = negative_log_likelihood(probs_all, targets_all)

    print("---- TPT's test accuracy: {:.2f}. ----\n".format(avg_acc))
    print(f"---- TPT metrics: Acc={avg_acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} ----")

    return {
        "acc": avg_acc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "probs": probs_all,
        "labels": targets_all,
    }


def log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time_per_sample, run_time, args,
                       compute_stats=None):
    file_exists = os.path.exists(results_path)
    fieldnames = [
        "run_time", "method", "dataset", "accuracy", "ece_15bins", "brier", "nll",
        "avg_time_per_sample", "avg_selected_experts", "image_encoder_gflops",
        "gflops_per_sample", "backbone", "tta_steps", "lr", "selection_p", "n_ctx",
    ]
    compute_stats = compute_stats or {}
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not file_exists) or os.path.getsize(results_path) == 0:
            writer.writeheader()
        writer.writerow({
            "run_time": run_time,
            "method": "TPT",
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
            "tta_steps": args.tta_steps,
            "lr": args.lr,
            "selection_p": args.selection_p,
            "n_ctx": args.n_ctx,
        })


def main():
    args = get_arguments()
    set_seed(args.seed)

    model_name = "ViT-B-16" if args.backbone == "ViT-B/16" else args.backbone
    pretrained = args.pretrained
    if pretrained is None:
        pretrained = None if model_name.startswith("hf-hub:") else "openai"

    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    clip_model = clip_model.to(args.device).eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    tokenizer = open_clip.get_tokenizer(model_name)
    tpt_preprocess = build_tpt_preprocess(preprocess, args.resolution, args.batch_size)

    os.makedirs(args.log_dir, exist_ok=True)
    results_path = os.path.join(args.log_dir, "results.csv")
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    else:
        group_name = None

    summary = []
    for dataset_name in args.datasets.split("/"):
        print(f"Processing {dataset_name} dataset.")
        test_loader, classnames, _ = build_test_data_loader(dataset_name, args.data_root, tpt_preprocess)
        sample_images, _ = next(iter(test_loader))
        sample_image_batch, clean_image = prepare_views(sample_images, args.device)
        compute_stats = compute_single_encoder_stats(
            clip_model,
            clean_image,
            device=args.device,
            image_forwards_per_sample=sample_image_batch.shape[0] + 1,
        )
        print(
            f"---- Compute estimate: image encoder={compute_stats['image_encoder_gflops']:.2f} GFLOPs | "
            f"GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={compute_stats['avg_selected_experts']:.2f} ----"
        )

        prompt_learner = build_prompt_learner(
            classnames,
            clip_model,
            tokenizer,
            args.n_ctx,
            args.ctx_init,
        ).to(args.device)

        if args.wandb:
            run = wandb.init(project="ETTA-CLIP", group=group_name, name=dataset_name, config=vars(args))

        start_time = time.perf_counter()
        metrics = run_tpt(test_loader, clip_model, prompt_learner, args)
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / max(1, len(test_loader.dataset))
        log_metric_results(
            results_path,
            dataset_name,
            metrics["acc"],
            metrics["ece"],
            metrics["brier"],
            metrics["nll"],
            avg_time,
            run_time,
            args,
            compute_stats,
        )
        summary.append((dataset_name, metrics["acc"], metrics["ece"], metrics["brier"], metrics["nll"], compute_stats))
        fig_stats = save_dataset_reliability_figure(
            metrics["probs"], metrics["labels"], dataset_name, args.log_dir, method_name="TPT"
        )
        print(f"---- Saved reliability figure: {fig_stats['figure_path']} ----")

        if args.wandb:
            wandb.log({
                f"{dataset_name}/acc": metrics["acc"],
                f"{dataset_name}/ece": metrics["ece"],
                f"{dataset_name}/brier": metrics["brier"],
                f"{dataset_name}/nll": metrics["nll"],
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
