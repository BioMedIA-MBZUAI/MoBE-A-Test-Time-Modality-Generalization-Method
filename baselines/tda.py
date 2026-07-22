import os
import sys
import csv
import time
import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import open_clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def get_arguments():
    # 'hardbench_btmri', 'hardbench_busi', 'hardbench_chmnist', 'hardbench_ctkidney', 'hardbench_covid19', 'hardbench_dermamnist', 'hardbench_kneexray', 'hardbench_kvasir', 'hardbench_lungcolon', 'hardbench_retina', 'hardbench_octmnist'
    default_datasets = "dermamnist"
    default_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    default_dataroot = os.path.join(PROJECT_ROOT, "datasets_all")

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, default=default_datasets, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default=default_dataroot, help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default=default_backbone, help='open_clip model name (e.g., RN50, ViT-B-16, hf-hub:...).')
    parser.add_argument('--pretrained', dest='pretrained', type=str, default=None, help='open_clip pretrained tag (e.g., openai, laion2b_s34b_b79k). Leave unset for hf-hub models.')
    parser.add_argument('--log-dir', type=str, default='./results/tda', help='Directory to save final metric results.')

    parser.add_argument('--positive-enabled', type=str_to_bool, nargs='?', const=True, default=False, help='Enable the positive TDA cache.')
    parser.add_argument('--positive-shot-capacity', type=int, default=3, help='Positive cache shot capacity.')
    parser.add_argument('--positive-alpha', type=float, default=2.0, help='Positive cache alpha.')
    parser.add_argument('--positive-beta', type=float, default=7.0, help='Positive cache beta.')

    parser.add_argument('--negative-enabled', type=str_to_bool, nargs='?', const=True, default=False, help='Enable the negative TDA cache.')
    parser.add_argument('--negative-shot-capacity', type=int, default=2, help='Negative cache shot capacity.')
    parser.add_argument('--negative-alpha', type=float, default=0.117, help='Negative cache alpha.')
    parser.add_argument('--negative-beta', type=float, default=1.0, help='Negative cache beta.')
    parser.add_argument('--negative-entropy-lower', type=float, default=0.2, help='Negative cache lower entropy threshold.')
    parser.add_argument('--negative-entropy-upper', type=float, default=0.5, help='Negative cache upper entropy threshold.')
    parser.add_argument('--negative-mask-lower', type=float, default=0.03, help='Negative cache lower mask threshold.')
    parser.add_argument('--negative-mask-upper', type=float, default=1.0, help='Negative cache upper mask threshold.')
    return parser.parse_args()


def build_tda_config(args):
    return {
        'positive': {
            'enabled': args.positive_enabled,
            'shot_capacity': args.positive_shot_capacity,
            'alpha': args.positive_alpha,
            'beta': args.positive_beta,
        },
        'negative': {
            'enabled': args.negative_enabled,
            'shot_capacity': args.negative_shot_capacity,
            'alpha': args.negative_alpha,
            'beta': args.negative_beta,
            'entropy_threshold': {
                'lower': args.negative_entropy_lower,
                'upper': args.negative_entropy_upper,
            },
            'mask_threshold': {
                'lower': args.negative_mask_lower,
                'upper': args.negative_mask_upper,
            },
        },
    }


# ----------------------------
# Calibration / probabilistic error metrics
# ----------------------------
def _prepare_single_label_targets(target):
    """
    Converts common dataset target formats to class indices.
    - [B] or [B,1] class ids -> [B]
    - [B,C] one-hot / multi-hot -> argmax labels for multiclass calibration
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
    return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels, reduction='mean').item()


def log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time_per_sample, run_time, args,
                       compute_stats=None):
    file_exists = os.path.exists(results_path)
    fieldnames = [
        'run_time', 'method', 'dataset', 'accuracy', 'ece_15bins', 'brier', 'nll',
        'avg_time_per_sample', 'avg_selected_experts', 'image_encoder_gflops',
        'gflops_per_sample', 'backbone', 'config'
    ]
    compute_stats = compute_stats or {}
    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not file_exists) or os.path.getsize(results_path) == 0:
            writer.writeheader()
        writer.writerow({
            'run_time': run_time,
            'method': 'TDA',
            'dataset': dataset_name,
            'accuracy': float(acc),
            'ece_15bins': float(ece),
            'brier': float(brier),
            'nll': float(nll),
            'avg_time_per_sample': float(avg_time_per_sample),
            'avg_selected_experts': compute_stats.get('avg_selected_experts', ''),
            'image_encoder_gflops': compute_stats.get('image_encoder_gflops', ''),
            'gflops_per_sample': compute_stats.get('gflops_per_sample', ''),
            'backbone': args.backbone,
            'config': 'cli_args',
        })


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        device = image_features.device
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        affinity = image_features @ cache_keys
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).to(device=device, dtype=affinity.dtype)
        else:
            cache_values = F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1)).to(device=device, dtype=affinity.dtype)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        all_probs, all_targets = [], []
        device = clip_weights.device

        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target = target.to(device)
            prop_entropy = get_entropy(loss, clip_weights)

            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features,
                    neg_cache,
                    neg_params['alpha'],
                    neg_params['beta'],
                    clip_weights,
                    (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                )

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            probs_eval = final_logits.softmax(dim=-1).detach().cpu()
            labels_eval = _prepare_single_label_targets(target.detach()).cpu()
            all_probs.append(probs_eval)
            all_targets.append(labels_eval)

            if wandb.run is not None:
                wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

            if i % 150 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

        avg_acc = sum(accuracies) / len(accuracies)
        probs_all = torch.cat(all_probs, dim=0)
        targets_all = torch.cat(all_targets, dim=0)
        ece = expected_calibration_error(probs_all, targets_all, n_bins=15)
        brier = brier_score(probs_all, targets_all)
        nll = negative_log_likelihood(probs_all, targets_all)

        print("---- TDA's test accuracy: {:.2f}. ----\n".format(avg_acc))
        print(f"---- TDA metrics: Acc={avg_acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} ----")

        if wandb.run is not None:
            wandb.log({
                'Final accuracy': avg_acc,
                'ECE_15bins': ece,
                'Brier_score': brier,
                'NLL': nll,
            }, commit=True)

        return {
            'acc': avg_acc,
            'ece': ece,
            'brier': brier,
            'nll': nll,
            'probs': probs_all,
            'labels': targets_all,
        }


def load_expert_mlp(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = len([k for k in sd.keys() if k in model.state_dict()])
    return loaded, missing, unexpected


def main():
    args = get_arguments()
    cfg = build_tda_config(args)

    model_name = args.backbone
    if model_name == 'ViT-B/16':
        model_name = 'ViT-B-16'
    if args.pretrained is None:
        pretrained = None if model_name.startswith('hf-hub:') else 'openai'
    else:
        pretrained = args.pretrained

    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime('%b%d_%H-%M-%S')
        group_name = f'{args.backbone}_{args.datasets}_{date}'

    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.log_dir, 'results.csv')

    datasets = args.datasets.split('/')
    summary = []
    for dataset_name in datasets:
        print(f'Processing {dataset_name} dataset.')

        print('\nRunning shared TDA configuration:')
        print(cfg, '\n')

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model, tokenizer).to(device)
        sample_images, _ = next(iter(test_loader))
        compute_stats = compute_single_encoder_stats(clip_model, sample_images, device=device)
        print(
            f"---- Compute estimate: image encoder={compute_stats['image_encoder_gflops']:.2f} GFLOPs | "
            f"GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={compute_stats['avg_selected_experts']:.2f} ----"
        )

        if args.wandb:
            run_name = f'{dataset_name}'
            run = wandb.init(project='ETTA-CLIP', config=cfg, group=group_name, name=run_name)

        start_time = time.perf_counter()
        metrics = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
        end_time = time.perf_counter()

        acc = metrics['acc']
        ece = metrics['ece']
        brier = metrics['brier']
        nll = metrics['nll']
        avg_time = (end_time - start_time) / max(1, len(test_loader.dataset))

        summary.append((dataset_name, acc, ece, brier, nll, compute_stats))
        log_metric_results(results_path, dataset_name, acc, ece, brier, nll, avg_time, run_time, args, compute_stats)
        fig_stats = save_dataset_reliability_figure(
            metrics['probs'], metrics['labels'], dataset_name, args.log_dir, method_name='TDA'
        )
        print(f"---- Saved reliability figure: {fig_stats['figure_path']} ----")

        if args.wandb:
            wandb.log({
                f'{dataset_name}/acc': acc,
                f'{dataset_name}/ece': ece,
                f'{dataset_name}/brier': brier,
                f'{dataset_name}/nll': nll,
            })
            run.finish()

    if summary:
        print('\nFinal metrics:')
        for name, acc, ece, brier, nll, compute_stats in summary:
            print(
                f'- {name}: Acc={acc:.2f} | ECE={ece:.2f} | Brier={brier:.4f} | NLL={nll:.4f} '
                f"| GFLOPs/sample={compute_stats['gflops_per_sample']:.2f} | Avg #Exp={compute_stats['avg_selected_experts']:.2f}"
            )


if __name__ == '__main__':
    main()
