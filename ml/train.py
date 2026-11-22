"""End-to-end PyTorch (MPS/CPU) training and evaluation pipeline for FaultSense-DL."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from ml.dataset import (
    MultiDomainImageDataset,
    MultiDomainVideoDataset,
    get_eval_transform,
    get_train_transform,
    materialize_dataset_and_samples,
)
from ml.model import FaultSenseTemporalNet
from ml.taxonomy import FAULT_CLASSES, NUM_CLASSES


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_spatial(
    model: FaultSenseTemporalNet,
    loader: DataLoader,
    device: torch.device,
    criterion_cls: nn.Module,
    criterion_sev: nn.Module,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels, severities in loader:
            images = images.to(device)
            labels = labels.to(device)
            severities = severities.float().to(device)

            out = model.forward_image(images)
            loss = criterion_cls(out["logits"], labels) + 0.35 * criterion_sev(
                out["severity"], severities
            )
            total_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = max(1, len(all_labels))
    acc = float(accuracy_score(all_labels, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {
        "loss": round(total_loss / n, 4),
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "preds": all_preds,
        "labels": all_labels,
    }


def train_multidomain(
    root_dir: Path,
    epochs_spatial: int = 8,
    epochs_temporal: int = 4,
    batch_size: int = 32,
    lr: float = 3e-4,
    backbone: str = "mobilenet_v3_small",
    samples_per_class: int = 110,
) -> Dict[str, Any]:
    """Trains the full FaultSenseTemporalNet on all domains and saves checkpoint + metrics."""
    t_start = time.time()
    device = get_device()
    print(f"[FaultSense] Training on device: {device} | backbone: {backbone}")

    train_samples, val_samples, test_samples = materialize_dataset_and_samples(
        root_dir, samples_per_class=samples_per_class
    )

    train_ds = MultiDomainImageDataset(train_samples, transform=get_train_transform(224))
    val_ds = MultiDomainImageDataset(val_samples, transform=get_eval_transform(224))
    test_ds = MultiDomainImageDataset(test_samples, transform=get_eval_transform(224))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FaultSenseTemporalNet(
        num_classes=NUM_CLASSES,
        backbone_name=backbone,
        pretrained=True,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_sev = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.spatial.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_spatial)

    ckpt_dir = root_dir / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "faultsense_best.pt"

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0

    # Stage 1: Spatial Encoder + Severity Head
    for epoch in range(1, epochs_spatial + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, severities in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            severities = severities.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model.forward_image(images)
            loss = criterion_cls(out["logits"], labels) + 0.35 * criterion_sev(
                out["severity"], severities
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.spatial.parameters(), 2.0)
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(out["logits"], dim=-1)
            correct += int((preds == labels).sum().item())
            total += images.size(0)

        scheduler.step()
        train_loss = round(running_loss / max(1, total), 4)
        train_acc = round(correct / max(1, total), 4)
        val_stats = evaluate_spatial(model, val_loader, device, criterion_cls, criterion_sev)

        row = {
            "epoch": epoch,
            "stage": "spatial",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["accuracy"],
            "val_f1": val_stats["f1_macro"],
        }
        history.append(row)
        print(
            f"[Spatial Epoch {epoch}/{epochs_spatial}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} val_f1={val_stats['f1_macro']:.4f}"
        )

        if val_stats["f1_macro"] >= best_val_f1:
            best_val_f1 = val_stats["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": backbone,
                    "num_classes": NUM_CLASSES,
                    "class_ids": [c["id"] for c in FAULT_CLASSES],
                    "best_val_f1": best_val_f1,
                },
                best_ckpt_path,
            )

    # Load best spatial weights before Temporal training
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Stage 2: Temporal Attention Aggregator on Video Clips
    for p in model.spatial.parameters():
        p.requires_grad = False
    video_train_ds = MultiDomainVideoDataset(num_clips=160, num_frames=12, seed_offset=2000)
    video_val_ds = MultiDomainVideoDataset(num_clips=40, num_frames=12, seed_offset=9000)
    video_train_loader = DataLoader(video_train_ds, batch_size=8, shuffle=True, num_workers=0)
    video_val_loader = DataLoader(video_val_ds, batch_size=8, shuffle=False, num_workers=0)

    t_opt = torch.optim.AdamW(model.temporal.parameters(), lr=4e-4, weight_decay=1e-4)
    temporal_history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs_temporal + 1):
        model.train()
        t_loss = 0.0
        t_correct = 0
        t_total = 0
        for clips, labels, severities in video_train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            severities = severities.float().to(device)

            t_opt.zero_grad(set_to_none=True)
            out = model.forward_video(clips)
            loss = criterion_cls(out["clip_logits"], labels) + 0.35 * criterion_sev(
                out["clip_severity"], severities
            )
            loss.backward()
            t_opt.step()

            t_loss += float(loss.item()) * clips.size(0)
            preds = torch.argmax(out["clip_logits"], dim=-1)
            t_correct += int((preds == labels).sum().item())
            t_total += clips.size(0)

        # Validate video clips
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for clips, labels, _ in video_val_loader:
                clips = clips.to(device)
                labels = labels.to(device)
                out = model.forward_video(clips)
                preds = torch.argmax(out["clip_logits"], dim=-1)
                v_correct += int((preds == labels).sum().item())
                v_total += clips.size(0)

        t_acc = round(t_correct / max(1, t_total), 4)
        v_acc = round(v_correct / max(1, v_total), 4)
        t_row = {
            "epoch": epoch,
            "stage": "temporal",
            "train_loss": round(t_loss / max(1, t_total), 4),
            "train_acc": t_acc,
            "val_acc": v_acc,
        }
        temporal_history.append(t_row)
        print(
            f"[Temporal Epoch {epoch}/{epochs_temporal}] "
            f"clip_train_acc={t_acc:.4f} clip_val_acc={v_acc:.4f}"
        )

    # Unfreeze for inference / future custom fine-tuning and save final best state
    for p in model.spatial.parameters():
        p.requires_grad = True

    test_stats = evaluate_spatial(model, test_loader, device, criterion_cls, criterion_sev)
    cm = confusion_matrix(
        test_stats["labels"],
        test_stats["preds"],
        labels=list(range(NUM_CLASSES)),
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        test_stats["labels"],
        test_stats["preds"],
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    per_class = []
    for i, cinfo in enumerate(FAULT_CLASSES):
        per_class.append(
            {
                "id": cinfo["id"],
                "label_en": cinfo["label_en"],
                "label_ar": cinfo["label_ar"],
                "domain": cinfo["domain"],
                "precision": round(float(p_cls[i]), 4),
                "recall": round(float(r_cls[i]), 4),
                "f1": round(float(f1_cls[i]), 4),
                "support": int(sup_cls[i]),
            }
        )

    elapsed = round(time.time() - t_start, 2)
    metrics_payload = {
        "device": str(device),
        "backbone": backbone,
        "num_classes": NUM_CLASSES,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "training_seconds": elapsed,
        "test_accuracy": test_stats["accuracy"],
        "test_f1_macro": test_stats["f1_macro"],
        "test_precision_macro": test_stats["precision_macro"],
        "test_recall_macro": test_stats["recall_macro"],
        "video_val_accuracy": temporal_history[-1]["val_acc"] if temporal_history else 0.0,
        "spatial_history": history,
        "temporal_history": temporal_history,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_ids": [c["id"] for c in FAULT_CLASSES],
        "class_labels": [c["label_en"] for c in FAULT_CLASSES],
    }

    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": backbone,
            "num_classes": NUM_CLASSES,
            "class_ids": [c["id"] for c in FAULT_CLASSES],
            "metrics": {
                "test_accuracy": test_stats["accuracy"],
                "test_f1_macro": test_stats["f1_macro"],
                "video_val_accuracy": metrics_payload["video_val_accuracy"],
            },
        },
        best_ckpt_path,
    )

    metrics_path = root_dir / "artifacts" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(
        f"[FaultSense] Training complete in {elapsed}s! "
        f"Test Acc={test_stats['accuracy']:.4f}, F1={test_stats['f1_macro']:.4f}, "
        f"Video Val Acc={metrics_payload['video_val_accuracy']:.4f}"
    )
    return metrics_payload


def train_custom_domain(
    custom_dir: Path,
    ckpt_out_path: Path,
    class_names: List[str],
    samples: List[Tuple[str, int]],
    epochs: int = 8,
    lr: float = 4e-4,
    backbone: str = "mobilenet_v3_small",
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fine-tunes a custom FaultSense model on user-uploaded classes."""
    device = get_device()
    num_classes = len(class_names)

    # Split 80/20 (ensuring at least 1 val sample per class if possible)
    rng = np.random.default_rng(42)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split = max(1, int(len(samples) * 0.8))
    train_idx = indices[:split]
    val_idx = indices[split:] if len(indices) > split else indices[:1]

    class _CustomDS(torch.utils.data.Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            from PIL import Image
            p, y = self.items[i]
            img = Image.open(p).convert("RGB")
            return self.transform(img), y

    train_items = [samples[i] for i in train_idx]
    val_items = [samples[i] for i in val_idx]

    train_loader = DataLoader(
        _CustomDS(train_items, get_train_transform(224)),
        batch_size=min(16, max(2, len(train_items))),
        shuffle=True,
    )
    val_loader = DataLoader(
        _CustomDS(val_items, get_eval_transform(224)),
        batch_size=min(16, max(1, len(val_items))),
        shuffle=False,
    )

    model = FaultSenseTemporalNet(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.spatial.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_image(imgs)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * imgs.size(0)
            correct += int((torch.argmax(out["logits"], dim=-1) == labels).sum().item())
            total += imgs.size(0)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model.forward_image(imgs)
                v_correct += int((torch.argmax(out["logits"], dim=-1) == labels).sum().item())
                v_total += imgs.size(0)

        train_acc = round(correct / max(1, total), 4)
        val_acc = round(v_correct / max(1, v_total), 4)
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, total), 4),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": backbone,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "best_val_acc": best_acc,
                },
                ckpt_out_path,
            )
        if progress_cb is not None:
            progress_cb({"epoch": epoch, "total_epochs": epochs, "history": history, "best_val_acc": best_acc})

    return {
        "class_names": class_names,
        "best_val_acc": best_acc,
        "history": history,
        "checkpoint": str(ckpt_out_path),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_multidomain(project_root)
