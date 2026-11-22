"""Grad-CAM visual explainability and fault localization with calm zero-blue colormaps."""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.model import FaultSenseSpatialEncoder


def build_calm_lut() -> np.ndarray:
    """
    256-entry RGB lookup table with ZERO blue dominance:
    0.00 -> Warm Cream (#FAF9F6)
    0.35 -> Soft Honey Amber (#E6BE73)
    0.70 -> Warm Terracotta (#C86A54)
    1.00 -> Deep Crimson-Plum (#732937)
    """
    anchors = np.array([0.0, 0.35, 0.70, 1.0], dtype=np.float32)
    # RGB tuples
    colors = np.array(
        [
            [250, 249, 246],
            [230, 190, 115],
            [200, 106, 84],
            [115, 41, 55],
        ],
        dtype=np.float32,
    )
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(x, anchors, colors[:, c]), 0, 255).astype(np.uint8)
    return lut


CALM_LUT_RGB = build_calm_lut()


class GradCAMExplainer:
    """Computes Grad-CAM heatmaps and bounding-box fault regions for spatial predictions."""

    def __init__(self, spatial_model: FaultSenseSpatialEncoder) -> None:
        self.model = spatial_model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        target_layer = self.model.get_target_layer()
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        input_tensor: [1, 3, H, W] on model device.
        Returns: (heatmap_01 [H, W], predicted_class_idx, logits [1, C])
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = input_tensor.clone().detach().requires_grad_(True)
        out = self.model(x)
        logits = out["logits"]
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        cls_idx = pred_idx if target_class is None else int(target_class)

        score = logits[0, cls_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32), pred_idx, logits.detach()

        # Global average pooling of gradients across spatial axes
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, h', w']
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
        max_val = float(cam_np.max())
        if max_val > 1e-8:
            cam_np = (cam_np - float(cam_np.min())) / (max_val - float(cam_np.min()) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np, pred_idx, logits.detach()


def render_calm_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.52,
) -> np.ndarray:
    """Blends Grad-CAM heatmap over RGB image using our calm zero-blue LUT."""
    h, w = image_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, 1.0)
    idx = (hm * 255).astype(np.uint8)
    colored_rgb = CALM_LUT_RGB[idx]

    # Modulate blending weight by activation intensity so clean areas stay crisp
    weight_map = (hm[..., None] ** 0.75) * alpha
    blended = image_rgb.astype(np.float32) * (1.0 - weight_map) + colored_rgb.astype(np.float32) * weight_map
    return np.clip(blended, 0, 255).astype(np.uint8)


def extract_and_draw_regions(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    is_fault: bool,
    threshold: float = 0.48,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Extracts salient defect bounding regions from Grad-CAM and renders calm corner-bracket overlays.
    Returns: (annotated_rgb, regions_list)
    """
    h, w = image_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    canvas = image_rgb.copy()
    regions: List[Dict[str, float]] = []

    if not is_fault:
        # Subtle sage border frame indicating pass/normal inspection
        sage_rgb = (78, 122, 101)
        cv2.rectangle(canvas, (6, 6), (w - 7, h - 7), sage_rgb, 2)
        return canvas, regions

    mask = (hm >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by peak activation
    scored = []
    min_area = max(32, int(0.004 * h * w))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        peak = float(hm[y : y + bh, x : x + bw].max())
        scored.append((peak, area, x, y, bw, bh))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Terracotta accent for fault regions (zero blue dominance)
    terracotta_rgb = (184, 93, 78)
    cream_rgb = (250, 249, 246)

    for idx_r, (peak, area, x, y, bw, bh) in enumerate(scored[:4]):
        regions.append(
            {
                "id": idx_r + 1,
                "x_norm": round(x / w, 4),
                "y_norm": round(y / h, 4),
                "w_norm": round(bw / w, 4),
                "h_norm": round(bh / h, 4),
                "peak_activation": round(peak, 4),
                "area_pct": round((area / (h * w)) * 100.0, 2),
            }
        )
        # Draw soft semi-transparent box fill + sleek corner brackets
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), terracotta_rgb, -1)
        canvas = cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0)
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), terracotta_rgb, 1)

        corner_len = max(10, min(bw, bh) // 4)
        thick = 3
        # Top-left
        cv2.line(canvas, (x, y), (x + corner_len, y), terracotta_rgb, thick)
        cv2.line(canvas, (x, y), (x, y + corner_len), terracotta_rgb, thick)
        # Top-right
        cv2.line(canvas, (x + bw, y), (x + bw - corner_len, y), terracotta_rgb, thick)
        cv2.line(canvas, (x + bw, y), (x + bw, y + corner_len), terracotta_rgb, thick)
        # Bottom-left
        cv2.line(canvas, (x, y + bh), (x + corner_len, y + bh), terracotta_rgb, thick)
        cv2.line(canvas, (x, y + bh), (x, y + bh - corner_len), terracotta_rgb, thick)
        # Bottom-right
        cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), terracotta_rgb, thick)
        cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), terracotta_rgb, thick)

        tag = f"R{idx_r + 1} {int(peak * 100)}%"
        tw = len(tag) * 7 + 10
        ty = max(18, y - 4)
        cv2.rectangle(canvas, (x, ty - 14), (x + tw, ty + 2), terracotta_rgb, -1)
        cv2.putText(
            canvas,
            tag,
            (x + 4, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            cream_rgb,
            1,
            cv2.LINE_AA,
        )

    return canvas, regions
