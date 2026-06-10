"""Fine-tune the CoR classifier head on labeled tomographic slices.

Single-GPU only; no distributed, no AMP, no wandb.

NOTE: only the `nn.Linear(embed_dim, 2)` head is trained — the vendored
`ClassificationModel.forward` hardcodes `self.model.eval()` + `torch.no_grad()`
around the DINOv2 backbone, so backbone gradients never flow even if you set
`requires_grad=True`. Full backbone fine-tuning is out of scope for this repo.

Data layout:
    LABELS_DIR/centered/*.tif        # well-centered reconstructions  (label 1)
    LABELS_DIR/off_centered/*.tif    # off-centered reconstructions   (label 0)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from tomo_center import logging as tca_logging
from tomo_center.ai.model_archs import ClassificationModel, _make_dinov2_model

log = tca_logging.getLogger(__name__)

_TIFF_EXTS = (".tif", ".tiff")


# ---------- dataset -----------------------------------------------------------

def _collect_pairs(labels_dir: Path) -> List[Tuple[Path, int]]:
    centered_dir = labels_dir / "centered"
    off_dir = labels_dir / "off_centered"
    for d in (centered_dir, off_dir):
        if not d.is_dir():
            raise SystemExit(
                f"Missing required subfolder: {d}\n"
                f"Expected layout: {labels_dir}/centered/*.tif and "
                f"{labels_dir}/off_centered/*.tif"
            )
    centered = sorted(p for p in centered_dir.iterdir()
                      if p.suffix.lower() in _TIFF_EXTS)
    off = sorted(p for p in off_dir.iterdir()
                 if p.suffix.lower() in _TIFF_EXTS)
    if not centered or not off:
        raise SystemExit(
            f"Need at least one TIFF in both centered/ ({len(centered)}) and "
            f"off_centered/ ({len(off)})."
        )
    return [(p, 1) for p in centered] + [(p, 0) for p in off]


class CoRDataset(Dataset):
    """Yields (image_tensor, label).

    Tensor shape: (1, 1, sz, sz) — matches what `ClassificationModel` expects
    when indexed as `sample['images'][:, 0]` in the single-window branch.
    """

    def __init__(self, pairs: List[Tuple[Path, int]], window_size: int, augment: bool):
        self.pairs = list(pairs)
        self.window_size = window_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label = self.pairs[idx]
        img = tifffile.imread(str(path))
        if img.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D image, got {img.shape}")
        img = img.astype(np.float32, copy=False)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        h, w = img.shape
        sz = self.window_size
        if h < sz or w < sz:
            raise ValueError(f"{path.name}: image {img.shape} smaller than window {sz}")

        cy = (h - sz) // 2
        cx = (w - sz) // 2
        if self.augment:
            # Small random offset, capped at sz//8 (~28 px for 224), then clipped to image.
            max_off = sz // 8
            cy += random.randint(-max_off, max_off)
            cx += random.randint(-max_off, max_off)
            cy = max(0, min(cy, h - sz))
            cx = max(0, min(cx, w - sz))

        crop = img[cy:cy + sz, cx:cx + sz]

        if self.augment and random.random() < 0.5:
            crop = crop[:, ::-1]

        # (sz, sz) -> (1, 1, sz, sz): (channel=1, window-index slot used as channel-of-window)
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).unsqueeze(0)
        return tensor, int(label)


def _split_pairs(pairs, val_split: float, seed: int):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_split))) if val_split > 0 else 0
    return pairs[n_val:], pairs[:n_val]


# ---------- model build / load ------------------------------------------------

def _build_model(args, device: torch.device) -> ClassificationModel:
    backbone = _make_dinov2_model()
    if args.resume is None:
        log.info("Loading backbone from torch.hub (%s) — requires internet.", args.base_model)
        try:
            hub_model = torch.hub.load("facebookresearch/dinov2", args.base_model)
        except Exception as e:
            raise SystemExit(
                f"--resume not given. Loading {args.base_model} from torch.hub requires "
                f"internet (failed: {e}). Either pass --resume <existing checkpoint.pt> "
                f"or run on a machine with internet access first."
            ) from e
        backbone.load_state_dict(hub_model.state_dict(), strict=False)

    # Backbone gradients can't flow through the vendored ClassificationModel anyway
    # (forward() forces torch.no_grad on it). Pin requires_grad=False so the
    # optimizer is built only over trainable head params and we report correct counts.
    for p in backbone.parameters():
        p.requires_grad = False

    model = ClassificationModel(
        backbone,
        embed_dim=backbone.embed_dim,
        num_windows=[1],
        multi_instances=False,
    )

    if args.resume is not None:
        log.info("Resuming full classifier from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        states = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        states = {(k.replace("module.", "") if k.startswith("module.") else k): v
                  for k, v in states.items()}
        msg = model.load_state_dict(states, strict=False)
        if msg.missing_keys:
            log.warning("Missing keys when loading --resume: %d (showing first 5: %s)",
                        len(msg.missing_keys), msg.missing_keys[:5])
        if msg.unexpected_keys:
            log.warning("Unexpected keys when loading --resume: %d (showing first 5: %s)",
                        len(msg.unexpected_keys), msg.unexpected_keys[:5])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("Model built: %d trainable params (head only) / %d total.",
             n_trainable, n_total)

    model.to(device)
    return model


# ---------- optimizer / scheduler ---------------------------------------------

def _make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no weight decay on gain/bias/norm parameters (matches Polaris script)."""
    def is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
        return p.ndim < 2 or "bias" in name or "ln" in name or "bn" in name

    no_decay, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if is_no_decay(n, p) else decay).append(p)

    return torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0},
         {"params": decay,    "weight_decay": weight_decay}],
        lr=lr,
    )


def _lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


# ---------- train / eval steps ------------------------------------------------

def _epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None):
    """One pass over `loader`. If optimizer is given, train; otherwise eval."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)     # (B, 1, 1, sz, sz)
            labels = labels.to(device, non_blocking=True)
            # ClassificationModel.forward takes a list of dicts (one per scale).
            logits = model([{"images": imgs}])            # (B, 2)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += bs
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ---------- entry point -------------------------------------------------------

def run_training(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU (slow).")
        args.device = "cpu"
    device = torch.device(args.device)

    log.info("Scanning %s for labeled TIFFs ...", args.labels_dir)
    pairs = _collect_pairs(args.labels_dir)
    log.info("  found %d slices total (centered=%d, off_centered=%d)",
             len(pairs),
             sum(1 for _, y in pairs if y == 1),
             sum(1 for _, y in pairs if y == 0))

    train_pairs, val_pairs = _split_pairs(pairs, args.val_split, args.seed)
    log.info("  split: train=%d  val=%d", len(train_pairs), len(val_pairs))

    train_ds = CoRDataset(train_pairs, args.window_size, augment=not args.no_augment)
    val_ds = CoRDataset(val_pairs, args.window_size, augment=False) if val_pairs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
                  if val_ds is not None else None)

    model = _build_model(args, device)
    optimizer = _make_optimizer(model, args.lr, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_lr_lambda(args.warmup_steps, total_steps))
    loss_fn = torch.nn.CrossEntropyLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0  # val_acc if we have val, else -train_loss
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _epoch(model, train_loader, loss_fn, device,
                                       optimizer=optimizer, scheduler=scheduler)
        if val_loader is not None:
            val_loss, val_acc = _epoch(model, val_loader, loss_fn, device)
            log.info("epoch %2d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f",
                     epoch, args.epochs, train_loss, train_acc, val_loss, val_acc)
            metric = val_acc
        else:
            log.info("epoch %2d/%d  train_loss=%.4f train_acc=%.3f  (no val)",
                     epoch, args.epochs, train_loss, train_acc)
            metric = -train_loss

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_acc": (metric if val_loader is not None else None),
                },
                args.out,
            )
            log.info("  saved best -> %s%s",
                     args.out,
                     f" (val_acc={metric:.3f})" if val_loader is not None else "")

    log.info("Training done. Best epoch=%d. Checkpoint: %s",
             best_epoch, args.out)
    return 0
