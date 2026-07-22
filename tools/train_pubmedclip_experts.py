# train_experts_hf.py
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel


# ----------------------------
# Modality inference (optional fallback)
# ----------------------------
def infer_modality_from_caption(caption: str):
    c = (caption or "").lower()

    if "computed tomography" in c or " ct " in f" {c} ":
        return "CT"
    if "magnetic resonance" in c or " mri " in f" {c} ":
        return "MRI"
    if "x-ray" in c or "xray" in c or "radiograph" in c:
        return "Xray"
    if "ultrasound" in c or "sonograph" in c or "us-guided" in c or "doppler" in c:
        return "Ultrasound"
    if "angiogram" in c or "angiography" in c or "angio" in c or "digital subtraction" in c or " dsa" in f" {c} ":
        return "Angiogram"

    return None


# ----------------------------
# ROCOv2 local dataset (returns PIL images + text)
# ----------------------------
class ROCOv2ModalityDataset(Dataset):
    """
    Returns:
      (PIL.Image RGB, caption_str)
    Collation + preprocessing is done by CLIPProcessor in collate_fn.
    """
    def __init__(
        self,
        roco_root: str,
        modality: str,
        split: str = "train",
        meta_filename_candidates: List[str] = ("metadata_fixed.jsonl", "metadata.jsonl"),
        infer_if_missing: bool = True,
    ):
        super().__init__()
        self.modality = modality
        self.split = split

        split_dir = os.path.join(roco_root, split)
        img_dir = os.path.join(split_dir, "images")

        meta_path = None
        for fn in meta_filename_candidates:
            p = os.path.join(split_dir, fn)
            if os.path.exists(p):
                meta_path = p
                break
        if meta_path is None:
            raise FileNotFoundError(
                f"Could not find metadata jsonl in {split_dir}. Tried: {meta_filename_candidates}"
            )
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        self.samples: List[Tuple[str, str]] = []

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item: Dict[str, Any] = json.loads(line)

                cap = item.get("caption", "")
                mod = item.get("modality", None)

                if mod is None and infer_if_missing:
                    mod = infer_modality_from_caption(cap)

                if mod != modality:
                    continue

                img_name = item.get("image", None)
                if not img_name:
                    continue

                img_path = os.path.join(img_dir, img_name)
                if not os.path.exists(img_path):
                    continue

                self.samples.append((img_path, cap))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[ROCOv2] Found 0 samples for split='{split}', modality='{modality}'. "
                f"Check modality names or enable inference."
            )

        print(f"[ROCOv2] split={split} modality={modality}: {len(self.samples)} samples (meta={os.path.basename(meta_path)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, caption


# ----------------------------
# Param selection: HF CLIP Vision MLP only
# ----------------------------
def is_hf_vision_mlp_param(name: str) -> bool:
    """
    HF CLIP vision transformer MLP params typically look like:
      vision_model.encoder.layers.{i}.mlp.fc1.weight
      vision_model.encoder.layers.{i}.mlp.fc2.weight
    """
    return ("vision_model.encoder.layers" in name) and (".mlp." in name)


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_hf_vision_mlp(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if is_hf_vision_mlp_param(n):
            p.requires_grad = True


def extract_hf_vision_mlp_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    full = model.state_dict()
    mlp_sd = {k: v.detach().cpu() for k, v in full.items() if is_hf_vision_mlp_param(k)}
    if len(mlp_sd) == 0:
        raise RuntimeError(
            "No HF vision MLP params found to save. "
            "Print model.named_parameters() and verify name patterns."
        )
    return mlp_sd


# ----------------------------
# CLIP contrastive loss from model logits (symmetric)
# ----------------------------
def clip_contrastive_loss_from_logits(logits_per_image, logits_per_text):
    # logits_per_image: [B, B], logits_per_text: [B, B]
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


# ----------------------------
# Collate fn using CLIPProcessor
# ----------------------------
def make_collate_fn(processor: CLIPProcessor, text_max_length: int):
    def collate(batch):
        images, texts = zip(*batch)  # list of PIL, list of str
        enc = processor(
            images=list(images),
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        # enc: dict with pixel_values [B,3,H,W], input_ids [B,L], attention_mask [B,L]
        return enc
    return collate


# ----------------------------
# Train one modality expert
# ----------------------------
def train_one_modality(modality: str, args, expert_idx: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = args.backbone  # HF repo id
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPModel.from_pretrained(model_id).to(device)
    text_max_length = int(model.config.text_config.max_position_embeddings)

    model.train()
    freeze_all(model)
    unfreeze_hf_vision_mlp(model)

    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable params after unfreeze_hf_vision_mlp(). "
            "Check is_hf_vision_mlp_param() for your model."
        )

    print(f"\n[Expert {expert_idx}] Modality={modality} trainable_params={len(trainable_params)}")
    # for sanity:
    # for n, p in trainable_params[:6]:
    #     print("  ", n, tuple(p.shape))

    # Datasets
    train_ds = ROCOv2ModalityDataset(args.roco_root, modality, split="train")
    val_ds   = ROCOv2ModalityDataset(args.roco_root, modality, split="val")

    collate_fn = make_collate_fn(processor, text_max_length=text_max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW([p for _, p in trainable_params], lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        running = 0.0
        nsteps = 0

        pbar = tqdm(train_loader, desc=f"[{modality}] train {epoch+1}/{args.epochs}")
        for enc in pbar:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            outputs = model(**enc)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            loss = clip_contrastive_loss_from_logits(logits_per_image, logits_per_text)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            nsteps += 1
            pbar.set_postfix(loss=running / max(1, nsteps))

        # ---- Val ----
        model.eval()
        v_running = 0.0
        v_steps = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"[{modality}] val   {epoch+1}/{args.epochs}")
            for enc in vbar:
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                outputs = model(**enc)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                vloss = clip_contrastive_loss_from_logits(logits_per_image, logits_per_text)

                v_running += float(vloss.item())
                v_steps += 1
                vbar.set_postfix(val_loss=v_running / max(1, v_steps))

        val_loss = v_running / max(1, v_steps)
        print(f"[Expert {expert_idx}] {modality} epoch {epoch+1}: val_loss={val_loss:.4f}")

        # Save best MLP-only checkpoint
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"expert_{modality}_0.pt")
            mlp_sd = extract_hf_vision_mlp_state_dict(model)
            torch.save(mlp_sd, out_path)
            print(f"[Expert {expert_idx}] Saved BEST HF vision-MLP weights: {out_path} (best_val={best_val:.4f})")


def get_args():
    project_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser()
    p.add_argument("--roco-root", default=str(project_root / "dataset_roco"))
    p.add_argument("--out-dir", default=str(project_root / "experts_pubmedclip_hf"))
    p.add_argument("--modalities", default="Angiogram,CT,MRI,Ultrasound,Xray")
    p.add_argument("--backbone", default="flaviagiammarino/pubmed-clip-vit-base-patch32")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def main():
    args = get_args()
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    for idx, modality in enumerate(mods):
        train_one_modality(modality, args, expert_idx=idx)


if __name__ == "__main__":
    main()
