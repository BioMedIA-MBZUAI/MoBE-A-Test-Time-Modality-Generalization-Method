# train_experts.py
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

import open_clip


# ----------------------------
# Modality inference (optional fallback)
# ----------------------------
def infer_modality_from_caption(caption: str):
    """
    If metadata modality is null, try to infer from caption.
    Returns one of: CT, MRI, Xray, Ultrasound, Angiogram, or None.
    """
    c = (caption or "").lower()

    # CT
    if "computed tomography" in c or " ct " in f" {c} ":
        return "CT"

    # MRI
    if "magnetic resonance" in c or " mri " in f" {c} ":
        return "MRI"

    # X-ray
    if "x-ray" in c or "xray" in c or "radiograph" in c:
        return "Xray"

    # Ultrasound
    if "ultrasound" in c or "sonograph" in c or "us-guided" in c or "doppler" in c:
        return "Ultrasound"

    # Angiogram
    if "angiogram" in c or "angiography" in c or "angio" in c or "digital subtraction" in c or " dsa" in f" {c}":
        return "Angiogram"

    return None


# ----------------------------
# ROCOv2 local dataset (reads JSONL + local images)
# ----------------------------
class ROCOv2ModalityDataset(Dataset):
    """
    Expects:
      {roco_root}/train/images/*.jpg and {roco_root}/train/metadata_fixed.jsonl  (or metadata.jsonl)
      {roco_root}/val/images/*.jpg   and {roco_root}/val/metadata_fixed.jsonl   (or metadata.jsonl)

    Each JSONL line should be valid JSON now:
      {"image":"000001.jpg","caption":"...","modality":null,"source":null,"idx":1}
    """
    def __init__(
        self,
        roco_root: str,
        modality: str,
        preprocess,
        split: str = "val",
        meta_filename_candidates: List[str] = ("metadata_fixed.jsonl", "metadata.jsonl"),
        infer_if_missing: bool = True,
    ):
        super().__init__()
        self.preprocess = preprocess
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
                    # skip missing files rather than crashing
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
        img = self.preprocess(img)
        return img, caption


# ----------------------------
# Param selection: vision MLP only (MoME-style experts)
# ----------------------------
def is_visual_mlp_param(name: str) -> bool:
    """
    Match MLP params inside the vision transformer blocks.

    Supports common OpenCLIP naming variants:
      - visual.transformer.resblocks.*.mlp.*   (older)
      - visual.trunk.blocks.*.mlp.*            (your BiomedCLIP)
    """
    if ".mlp." not in name:
        return False

    # Your variant
    if "visual.trunk.blocks." in name:
        return True

    # Other common variant
    if "visual.transformer" in name and "resblocks" in name:
        return True

    # Some builds use "visual.transformer.blocks"
    if "visual.transformer.blocks" in name:
        return True

    return False


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_visual_mlp(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if is_visual_mlp_param(n):
            p.requires_grad = True

def extract_visual_mlp_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    full = model.state_dict()
    mlp_sd = {k: v.detach().cpu() for k, v in full.items() if is_visual_mlp_param(k)}
    if len(mlp_sd) == 0:
        raise RuntimeError(
            "No visual MLP params found to save. "
            "Check is_visual_mlp_param() naming against your model.named_parameters()."
        )
    return mlp_sd


# ----------------------------
# CLIP contrastive loss
# ----------------------------
def clip_contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * (image_features @ text_features.t())
    logits_per_text = logits_per_image.t()
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


# ----------------------------
# Train one modality expert
# ----------------------------
from transformers import CLIPProcessor, CLIPModel

def train_one_modality(modality: str, args, expert_idx: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.backbone
    if model_name == "ViT-B/16":
        model_name = "ViT-B-16"

    if args.pretrained is None:
        pretrained = None if model_name.startswith("hf-hub:") else "openai"
    else:
        pretrained = args.pretrained

    # model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    preprocess = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.train()

    freeze_all(model)
    unfreeze_visual_mlp(model)

    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable params after unfreeze_visual_mlp(). "
            "Check is_visual_mlp_param() for your model."
        )

    print(f"\n[Expert {expert_idx}] Modality={modality} trainable_params={len(trainable_params)}")
    # Print a few names for sanity
    # for n, p in trainable_params[:8]:
    #     print("  ", n, tuple(p.shape))
    # if len(trainable_params) > 8:
    #     print("  ...")

    # Build datasets
    train_ds = ROCOv2ModalityDataset(args.roco_root, modality, preprocess, split="train")
    val_ds = ROCOv2ModalityDataset(args.roco_root, modality, preprocess, split="val")
    print(f"[Expert {expert_idx}/{modality}] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    optimizer = torch.optim.AdamW([p for _, p in trainable_params], lr=args.lr, weight_decay=args.wd)

    def get_logit_scale():
        if hasattr(model, "logit_scale"):
            return model.logit_scale.exp()
        return torch.tensor(args.logit_scale, device=device)

    best_val = float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        running = 0.0
        nsteps = 0

        pbar = tqdm(train_loader, desc=f"[{modality}] train epoch {epoch+1}/{args.epochs}")
        for images, texts in pbar:
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(texts)).to(device)

            image_features = F.normalize(model.encode_image(images), dim=-1)
            text_features = F.normalize(model.encode_text(tokens), dim=-1)

            loss = clip_contrastive_loss(image_features, text_features, get_logit_scale())

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
            vbar = tqdm(val_loader, desc=f"[{modality}] val   epoch {epoch+1}/{args.epochs}")
            for images, texts in vbar:
                images = images.to(device, non_blocking=True)
                tokens = tokenizer(list(texts)).to(device)

                image_features = F.normalize(model.encode_image(images), dim=-1)
                text_features = F.normalize(model.encode_text(tokens), dim=-1)

                vloss = clip_contrastive_loss(image_features, text_features, get_logit_scale())
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
            mlp_sd = extract_visual_mlp_state_dict(model)
            torch.save(mlp_sd, out_path)
            print(f"[Expert {expert_idx}] Saved BEST MLP weights to: {out_path} (best_val={best_val:.4f})")


def get_args():

    # CUDA_VISIBLE_DEVICES=0 python train_experts.py \
    # --roco-root ./dataset_roco \
    # --out-dir experts \
    # --modalities "CT,MRI,Xray,Ultrasound,Angiogram" \
    # --backbone hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

    project_root = Path(__file__).resolve().parents[1]
    default_roco_root = project_root / "dataset_roco"
    default_out_dir = project_root / "experts"
    default_modalities = "Angiogram,CT,MRI,Ultrasound,Xray"
    # default_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    default_backbone = "flaviagiammarino/pubmed-clip-vit-base-patch32"  # LAION pretrain

    p = argparse.ArgumentParser()
    p.add_argument("--roco-root", default=str(default_roco_root), help="Path to ROCOv2 root / metadata")
    p.add_argument("--out-dir", default=str(default_out_dir), help="Where to save expert_<modality>_0.pt")
    p.add_argument("--modalities", default=default_modalities,
                   help="Comma-separated modality list (order defines expert index).")
    p.add_argument("--backbone", default=default_backbone,
                   help="e.g. hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--pretrained", default=None)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--logit-scale", type=float, default=100.0, help="fallback if model has no logit_scale")
    return p.parse_args()

def main():
    args = get_args()
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    for idx, modality in enumerate(mods):
        train_one_modality(modality, args, expert_idx=idx)


if __name__ == "__main__":
    main()
