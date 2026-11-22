"""Multi-domain image and video dataset synthesis, loading, and sample generation."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from ml.taxonomy import FAULT_CLASSES, CLASS_BY_ID, CLASS_NAMES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transform(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size + 16, img_size + 16)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.12),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _multi_octave_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    for scale, weight in [(4, 0.45), (12, 0.30), (32, 0.25)]:
        sh, sw = max(4, h // scale), max(4, w // scale)
        low = rng.normal(0.0, 1.0, (sh, sw)).astype(np.float32)
        up = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += up * weight
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    return acc


def _brushed_metal(size: int, rng: np.random.Generator) -> np.ndarray:
    base_val = rng.integers(145, 195)
    img = np.full((size, size, 3), base_val, dtype=np.float32)
    # Directional brushed streaks
    streak = rng.normal(0, 12, (size, max(8, size // 6))).astype(np.float32)
    streak = cv2.resize(streak, (size, size), interpolation=cv2.INTER_LINEAR)
    angle = rng.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    streak = cv2.warpAffine(streak, M, (size, size), borderMode=cv2.BORDER_REFLECT)

    # Specular lighting gradient
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * size, rng.uniform(0.2, 0.8) * size
    glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.45) ** 2)) * 28.0
    fine = rng.normal(0, 3.5, (size, size)).astype(np.float32)

    for c, tint in enumerate([1.0, 0.99, 0.96]):  # Warm neutral steel, no blue cast
        img[:, :, c] = (img[:, :, c] + streak + glow + fine) * tint
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_fractal_crack(
    canvas: np.ndarray,
    rng: np.random.Generator,
    dark_color: Tuple[int, int, int] = (32, 30, 28),
    highlight_color: Tuple[int, int, int] = (210, 204, 196),
) -> None:
    h, w = canvas.shape[:2]
    x = float(rng.uniform(0.15 * w, 0.85 * w))
    y = float(rng.uniform(0.15 * h, 0.85 * h))
    angle = float(rng.uniform(0, 2 * np.pi))
    steps = int(rng.integers(24, 42))
    pts = [(int(x), int(y))]
    for step in range(steps):
        angle += float(rng.uniform(-0.55, 0.55))
        step_len = float(rng.uniform(4.5, 9.5))
        x = np.clip(x + np.cos(angle) * step_len, 4, w - 5)
        y = np.clip(y + np.sin(angle) * step_len, 4, h - 5)
        pts.append((int(x), int(y)))
        # Occasional micro-branch
        if step in (8, 16, 24) and rng.random() < 0.75:
            bx, by = x, y
            ba = angle + float(rng.choice([-1, 1])) * float(rng.uniform(0.5, 1.1))
            b_pts = [(int(bx), int(by))]
            for _ in range(int(rng.integers(6, 12))):
                ba += float(rng.uniform(-0.4, 0.4))
                bx = np.clip(bx + np.cos(ba) * 5.0, 4, w - 5)
                by = np.clip(by + np.sin(ba) * 5.0, 4, h - 5)
                b_pts.append((int(bx), int(by)))
            for i in range(len(b_pts) - 1):
                cv2.line(canvas, b_pts[i], b_pts[i + 1], dark_color, 1, cv2.LINE_AA)

    for i in range(len(pts) - 1):
        thick = 3 if i < len(pts) // 2 else 2
        p1, p2 = pts[i], pts[i + 1]
        cv2.line(canvas, (p1[0] + 1, p1[1] + 1), (p2[0] + 1, p2[1] + 1), highlight_color, 1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, dark_color, thick, cv2.LINE_AA)


def _draw_pcb_board(size: int, defective: bool, rng: np.random.Generator) -> np.ndarray:
    # Rich sage-emerald solder mask
    mask_rgb = np.array([42, 92, 66], dtype=np.float32)
    noise = _multi_octave_noise(size, size, rng)[..., None] * 18.0 - 9.0
    img = np.clip(mask_rgb + noise, 0, 255).astype(np.uint8)

    trace_color = (96, 148, 112)
    gold_pad = (212, 182, 108)
    ic_body = (44, 44, 42)

    # Copper traces grid
    for y in range(28, size - 24, int(rng.integers(22, 34))):
        x1 = int(rng.integers(12, 40))
        x2 = int(rng.integers(size - 45, size - 12))
        cv2.line(img, (x1, y), (x2, y), trace_color, 2, cv2.LINE_AA)
        cv2.circle(img, (x1, y), 4, gold_pad, -1, cv2.LINE_AA)
        cv2.circle(img, (x2, y), 4, gold_pad, -1, cv2.LINE_AA)

    # IC packages
    for _ in range(int(rng.integers(2, 4))):
        ix = int(rng.integers(45, size - 90))
        iy = int(rng.integers(45, size - 90))
        iw = int(rng.integers(34, 54))
        ih = int(rng.integers(28, 44))
        cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), ic_body, -1)
        cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), (88, 86, 80), 1)
        for px in range(ix + 5, ix + iw - 3, 8):
            cv2.rectangle(img, (px, iy - 5), (px + 4, iy), gold_pad, -1)
            cv2.rectangle(img, (px, iy + ih), (px + 4, iy + ih + 5), gold_pad, -1)

    if defective:
        dx = int(rng.integers(56, size - 56))
        dy = int(rng.integers(56, size - 56))
        # Scorched burn halo + solder bridge blob
        cv2.circle(img, (dx, dy), int(rng.integers(18, 28)), (58, 36, 28), -1)
        cv2.circle(img, (dx, dy), int(rng.integers(10, 16)), (184, 92, 68), -1)
        cv2.ellipse(
            img,
            (dx + 4, dy - 2),
            (int(rng.integers(10, 18)), int(rng.integers(5, 9))),
            float(rng.uniform(0, 180)),
            0,
            360,
            (196, 194, 188),
            -1,
        )
        # Broken trace gap
        cv2.line(img, (dx - 16, dy + 10), (dx + 16, dy - 10), (32, 24, 22), 3, cv2.LINE_AA)
    return img


def _draw_asphalt(size: int, pothole: bool, rng: np.random.Generator) -> np.ndarray:
    coarse = _multi_octave_noise(size, size, rng)
    grain = rng.normal(0, 14, (size, size)).astype(np.float32)
    base = 76.0 + coarse * 34.0 + grain
    img = np.stack([base * 1.02, base * 1.0, base * 0.96], axis=-1)
    img = np.clip(img, 20, 210).astype(np.uint8)

    # Occasional warm road stripe on edge
    if rng.random() < 0.55:
        sx = int(rng.choice([24, size - 32]))
        cv2.rectangle(img, (sx, 0), (sx + 10, size), (218, 198, 134), -1)

    if pothole:
        cx = int(rng.integers(64, size - 64))
        cy = int(rng.integers(64, size - 64))
        rx = int(rng.integers(26, 48))
        ry = int(rng.integers(20, 38))
        # Outer crumbling rim
        pts = []
        for a in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            r_mod = float(rng.uniform(0.75, 1.25))
            pts.append([int(cx + np.cos(a) * rx * r_mod), int(cy + np.sin(a) * ry * r_mod)])
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [poly], True, (132, 124, 114), 4, cv2.LINE_AA)
        cv2.fillPoly(img, [poly], (34, 31, 29))
        # Inner deep shadow + loose stones
        inner = (poly * 0.72 + np.array([cx * 0.28, cy * 0.28])).astype(np.int32)
        cv2.fillPoly(img, [inner], (20, 18, 17))
        for _ in range(14):
            sx = int(cx + rng.integers(-rx // 2, rx // 2))
            sy = int(cy + rng.integers(-ry // 2, ry // 2))
            cv2.circle(img, (sx, sy), int(rng.integers(2, 4)), (92, 86, 78), -1)
    return img


def _draw_corrosion(size: int, rng: np.random.Generator) -> np.ndarray:
    img = _brushed_metal(size, rng).astype(np.float32)
    rust_map = _multi_octave_noise(size, size, rng)
    thresh = float(rng.uniform(0.50, 0.62))
    mask = np.clip((rust_map - thresh) / (1.0 - thresh + 1e-6), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    # Rich rust tones (sienna / terracotta / ochre — RGB)
    rust_dark = np.array([116, 52, 32], dtype=np.float32)
    rust_bright = np.array([186, 98, 48], dtype=np.float32)
    detail = _multi_octave_noise(size, size, rng)[..., None]
    rust_tex = rust_dark * (1 - detail) + rust_bright * detail

    alpha = np.clip(mask[..., None] * 1.45, 0.0, 0.95)
    blended = img * (1.0 - alpha) + rust_tex * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _draw_safety_scene(
    size: int,
    variant: str,
    rng: np.random.Generator,
    worker_offset: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    # Clean warm concrete factory floor
    noise = _multi_octave_noise(size, size, rng)
    floor = 188.0 + noise * 22.0
    img = np.stack([floor, floor * 0.99, floor * 0.96], axis=-1).astype(np.uint8)

    # Restricted hazard zone on right/bottom portion
    hz_x = int(size * 0.58)
    cv2.rectangle(img, (hz_x, 0), (size, size), (242, 224, 218), -1)
    # Diagonal warning stripes along boundary
    for y in range(-size, size * 2, 22):
        cv2.line(img, (hz_x - 8, y), (hz_x + 8, y + 16), (188, 96, 74), 5)
    cv2.line(img, (hz_x, 0), (hz_x, size), (176, 86, 68), 3)

    # Machine block inside hazard zone
    cv2.rectangle(img, (hz_x + 24, int(size * 0.25)), (size - 16, int(size * 0.72)), (110, 114, 108), -1)
    cv2.rectangle(img, (hz_x + 24, int(size * 0.25)), (size - 16, int(size * 0.72)), (74, 76, 72), 2)

    # Safe walkway indicator on left
    cv2.rectangle(img, (18, 18), (hz_x - 24, size - 18), (225, 236, 228), 2)

    # Worker position
    if worker_offset is not None:
        wx, wy = worker_offset
    elif variant == "safety_hazard_zone":
        wx = int(rng.integers(hz_x + 12, size - 36))
        wy = int(rng.integers(int(size * 0.35), int(size * 0.72)))
    else:
        wx = int(rng.integers(int(size * 0.18), hz_x - 38))
        wy = int(rng.integers(int(size * 0.36), int(size * 0.74)))

    # Shadow
    cv2.ellipse(img, (wx, wy + 44), (20, 8), 0, 0, 360, (145, 142, 136), -1)

    # Legs (dark charcoal trousers)
    cv2.line(img, (wx - 7, wy + 16), (wx - 10, wy + 42), (54, 56, 52), 7, cv2.LINE_AA)
    cv2.line(img, (wx + 7, wy + 16), (wx + 10, wy + 42), (54, 56, 52), 7, cv2.LINE_AA)

    # Torso: high-vis vest (warm amber/sage-lime) or plain grey if hazard breach
    vest_color = (224, 178, 64) if variant != "safety_hazard_zone" else (142, 138, 130)
    cv2.rectangle(img, (wx - 15, wy - 14), (wx + 15, wy + 18), vest_color, -1)
    if variant != "safety_hazard_zone":
        # Reflective strips
        cv2.line(img, (wx - 14, wy - 2), (wx + 14, wy - 2), (245, 245, 240), 2)
        cv2.line(img, (wx - 14, wy + 8), (wx + 14, wy + 8), (245, 245, 240), 2)

    # Arms
    cv2.line(img, (wx - 15, wy - 10), (wx - 23, wy + 12), vest_color, 5, cv2.LINE_AA)
    cv2.line(img, (wx + 15, wy - 10), (wx + 23, wy + 12), vest_color, 5, cv2.LINE_AA)

    # Head
    cv2.circle(img, (wx, wy - 26), 11, (214, 176, 146), -1)

    if variant == "safety_no_helmet":
        # Bare dark hair, NO safety helmet
        cv2.ellipse(img, (wx, wy - 31), (11, 7), 0, 180, 360, (48, 38, 32), -1)
    else:
        # Bright safety helmet (yellow-amber)
        cv2.ellipse(img, (wx, wy - 31), (13, 10), 0, 180, 360, (236, 194, 52), -1)
        cv2.line(img, (wx - 15, wy - 31), (wx + 15, wy - 31), (218, 168, 36), 3)

    return img


def synthesize_image(
    class_id: str,
    size: int = 224,
    seed: Optional[int] = None,
    worker_offset: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Synthesizes an RGB uint8 image for any of the 10 built-in multi-domain classes."""
    rng = np.random.default_rng(seed)
    if class_id == "industrial_normal":
        return _brushed_metal(size, rng)
    if class_id == "industrial_scratch":
        img = _brushed_metal(size, rng)
        for _ in range(int(rng.integers(1, 3))):
            x1, y1 = int(rng.integers(24, size - 24)), int(rng.integers(24, size - 24))
            angle = float(rng.uniform(-0.9, 0.9))
            length = int(rng.integers(55, 115))
            x2 = int(np.clip(x1 + np.cos(angle) * length, 10, size - 10))
            y2 = int(np.clip(y1 + np.sin(angle) * length, 10, size - 10))
            cv2.line(img, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (235, 228, 214), 2, cv2.LINE_AA)
            cv2.line(img, (x1, y1), (x2, y2), (56, 48, 44), int(rng.integers(2, 4)), cv2.LINE_AA)
        return img
    if class_id == "industrial_crack":
        img = _brushed_metal(size, rng)
        _draw_fractal_crack(img, rng)
        return img
    if class_id == "industrial_pcb_defect":
        return _draw_pcb_board(size, defective=True, rng=rng)
    if class_id == "infra_normal":
        return _draw_asphalt(size, pothole=False, rng=rng)
    if class_id == "infra_pothole":
        return _draw_asphalt(size, pothole=True, rng=rng)
    if class_id == "infra_corrosion":
        return _draw_corrosion(size, rng)
    if class_id in ("safety_compliant", "safety_no_helmet", "safety_hazard_zone"):
        return _draw_safety_scene(size, class_id, rng, worker_offset=worker_offset)
    return _brushed_metal(size, rng)


def synthesize_video_frames(
    class_id: str,
    num_frames: int = 16,
    size: int = 224,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Synthesizes a temporal sequence of frames simulating real camera/conveyor/worker motion.
    For fault classes, the fault translates smoothly across the view or triggers mid-sequence.
    """
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, 10_000_000))

    if class_id.startswith("safety_"):
        frames = []
        hz_x = int(size * 0.58)
        start_x = int(size * 0.28)
        end_x = int(size * 0.78) if class_id == "safety_hazard_zone" else int(size * 0.44)
        wy = int(size * 0.56)
        for t in range(num_frames):
            alpha = t / max(1, num_frames - 1)
            wx = int(start_x + (end_x - start_x) * alpha + np.sin(alpha * np.pi) * 6)
            # If hazard_zone, worker crosses boundary mid-video
            frame_variant = (
                "safety_hazard_zone"
                if (class_id == "safety_hazard_zone" and wx >= hz_x - 4)
                else ("safety_no_helmet" if class_id == "safety_no_helmet" else "safety_compliant")
            )
            f = synthesize_image(frame_variant, size=size, seed=base_seed, worker_offset=(wx, wy))
            frames.append(f)
        return frames

    # Panning surface / road / PCB across conveyor or dashcam
    big = int(size * 1.45)
    canvas = synthesize_image(class_id, size=big, seed=base_seed)
    frames = []
    max_shift = big - size
    dx_total = int(rng.integers(max_shift // 2, max_shift))
    dy_start = int(rng.integers(0, max(1, max_shift // 2)))
    for t in range(num_frames):
        alpha = t / max(1, num_frames - 1)
        ox = int(alpha * dx_total)
        oy = int(np.clip(dy_start + np.sin(alpha * np.pi) * 10, 0, max_shift))
        crop = canvas[oy : oy + size, ox : ox + size].copy()
        # Subtle temporal sensor noise
        noise = rng.normal(0, 1.8, crop.shape).astype(np.float32)
        crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(crop)
    return frames


class MultiDomainImageDataset(Dataset):
    """On-disk or memory-cached multi-domain fault image dataset."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform or get_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        severity = FAULT_CLASSES[label]["base_severity"] / 100.0
        return tensor, label, float(severity)


class MultiDomainVideoDataset(Dataset):
    """Generates or loads T-frame video sequences for Temporal Attention training."""

    def __init__(
        self,
        num_clips: int = 240,
        num_frames: int = 12,
        img_size: int = 224,
        seed_offset: int = 1000,
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.img_size = img_size
        self.seed_offset = seed_offset
        self.transform = get_eval_transform(img_size)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        label = idx % len(FAULT_CLASSES)
        cls_info = FAULT_CLASSES[label]
        frames_rgb = synthesize_video_frames(
            cls_info["id"],
            num_frames=self.num_frames,
            size=self.img_size,
            seed=self.seed_offset + idx * 17,
        )
        tensors = [self.transform(Image.fromarray(f)) for f in frames_rgb]
        clip = torch.stack(tensors, dim=0)  # [T, C, H, W]
        severity = cls_info["base_severity"] / 100.0
        return clip, label, float(severity)


def materialize_dataset_and_samples(
    root_dir: Path,
    samples_per_class: int = 110,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Creates clean train/val/test image splits on disk and ready-to-test image/video samples.
    """
    data_dir = root_dir / "data" / "multidomain_images"
    sample_dir = root_dir / "artifacts" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    n_train = int(samples_per_class * 0.72)
    n_val = int(samples_per_class * 0.14)

    for cls_info in FAULT_CLASSES:
        cid = cls_info["id"]
        cidx = cls_info["index"]
        for split_name, count, bucket, seed_base in [
            ("train", n_train, train_samples, 10_000 + cidx * 1000),
            ("val", n_val, val_samples, 50_000 + cidx * 1000),
            ("test", samples_per_class - n_train - n_val, test_samples, 90_000 + cidx * 1000),
        ]:
            split_dir = data_dir / split_name / cid
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                fpath = split_dir / f"{cid}_{i:03d}.png"
                if not fpath.exists():
                    rgb = synthesize_image(cid, size=256, seed=seed_base + i)
                    Image.fromarray(rgb).save(fpath)
                bucket.append((str(fpath), cidx))

        # Save one crisp interactive sample image per class
        sample_img = sample_dir / f"sample_{cid}.png"
        if not sample_img.exists():
            rgb = synthesize_image(cid, size=384, seed=777_000 + cidx)
            Image.fromarray(rgb).save(sample_img)

    # Save 4 realistic H.264 sample videos across all domains
    video_specs = [
        ("video_industrial_scratch.mp4", "industrial_scratch", 48),
        ("video_road_pothole.mp4", "infra_pothole", 48),
        ("video_safety_hazard_breach.mp4", "safety_hazard_zone", 56),
        ("video_industrial_clean.mp4", "industrial_normal", 48),
    ]
    for fname, cid, n_frames in video_specs:
        vpath = sample_dir / fname
        if not vpath.exists():
            frames = synthesize_video_frames(cid, num_frames=n_frames, size=320, seed=888_000 + hash(cid) % 1000)
            writer = cv2.VideoWriter(
                str(vpath),
                cv2.VideoWriter_fourcc(*"avc1"),
                12.0,
                (320, 320),
            )
            for fr in frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()

    return train_samples, val_samples, test_samples


def load_custom_image_folder(folder: Path) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Loads any custom folder-per-class image dataset."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    classes = sorted([d.name for d in folder.iterdir() if d.is_dir() and not d.name.startswith(".")])
    samples: List[Tuple[str, int]] = []
    for idx, cname in enumerate(classes):
        for f in sorted((folder / cname).iterdir()):
            if f.suffix.lower() in exts:
                samples.append((str(f), idx))
    return samples, classes
