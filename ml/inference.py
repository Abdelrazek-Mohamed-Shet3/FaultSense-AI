"""Real-time Spatial & Temporal Deep Learning Inference Engine for Images & Videos."""

import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.dataset import get_eval_transform
from ml.gradcam import GradCAMExplainer, extract_and_draw_regions, render_calm_overlay
from ml.model import FaultSenseTemporalNet
from ml.taxonomy import CLASS_BY_INDEX, FAULT_CLASSES, NUM_CLASSES
from ml.train import get_device


class FaultSenseEngine:
    """Thread-safe inference engine supporting built-in multi-domain and custom models."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.device = get_device()
        self.transform = get_eval_transform(224)
        self._lock = threading.Lock()

        self.builtin_ckpt = root_dir / "artifacts" / "checkpoints" / "faultsense_best.pt"
        self.custom_ckpt = root_dir / "artifacts" / "checkpoints" / "faultsense_custom.pt"

        self.models: Dict[str, FaultSenseTemporalNet] = {}
        self.custom_class_names: List[str] = []
        self.load_builtin()
        self.load_custom_if_available()

    def load_builtin(self) -> None:
        if self.builtin_ckpt.exists():
            ckpt = torch.load(self.builtin_ckpt, map_location=self.device, weights_only=False)
            backbone = ckpt.get("backbone", "mobilenet_v3_small")
            num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
            model = FaultSenseTemporalNet(
                num_classes=num_classes,
                backbone_name=backbone,
                pretrained=False,
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.models["builtin"] = model
        else:
            model = FaultSenseTemporalNet(num_classes=NUM_CLASSES, pretrained=True).to(self.device)
            model.eval()
            self.models["builtin"] = model

    def load_custom_if_available(self) -> bool:
        if not self.custom_ckpt.exists():
            return False
        ckpt = torch.load(self.custom_ckpt, map_location=self.device, weights_only=False)
        backbone = ckpt.get("backbone", "mobilenet_v3_small")
        class_names = ckpt.get("class_names", [])
        if not class_names:
            return False
        model = FaultSenseTemporalNet(
            num_classes=len(class_names),
            backbone_name=backbone,
            pretrained=False,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.models["custom"] = model
        self.custom_class_names = list(class_names)
        return True

    def _get_model_and_meta(self, active_model: str) -> Tuple[FaultSenseTemporalNet, bool]:
        if active_model == "custom" and "custom" in self.models:
            return self.models["custom"], True
        return self.models["builtin"], False

    def _class_meta(self, idx: int, is_custom: bool) -> Dict[str, Any]:
        if is_custom:
            cname = self.custom_class_names[idx]
            lower = cname.lower()
            is_fault = not any(k in lower for k in ["normal", "clean", "good", "ok", "pass", "intact", "compliant"])
            sev_level = "pass" if not is_fault else "critical"
            return {
                "id": f"custom_{cname}",
                "index": idx,
                "domain": "custom",
                "label_en": cname.replace("_", " ").title(),
                "label_ar": cname.replace("_", " "),
                "severity_level": sev_level,
                "base_severity": 6 if not is_fault else 82,
                "is_fault": is_fault,
                "color": "#4E7A65" if not is_fault else "#B85D4E",
                "bg_tint": "#EFF5F1" if not is_fault else "#FDF0EE",
                "description_en": f"Custom-trained class '{cname}' predicted by fine-tuned FaultSense model.",
                "description_ar": f"تم التصنيف كفئة مخصصة ({cname}) بواسطة النموذج المدرب على بياناتك.",
                "recommendation_en": "Review localized region and follow custom SOP." if is_fault else "Passed custom inspection.",
                "recommendation_ar": "مراجعة المنطقة المحددة واتباع إجراءات الفحص المخصصة." if is_fault else "اجتاز الفحص المخصص بنجاح.",
            }
        return CLASS_BY_INDEX[idx]

    def _apply_domain_mask(
        self,
        logits: torch.Tensor,
        domain_filter: Optional[str],
        is_custom: bool,
    ) -> torch.Tensor:
        if is_custom or not domain_filter or domain_filter == "all":
            return logits
        masked = logits.clone()
        allowed = [c["index"] for c in FAULT_CLASSES if c["domain"] == domain_filter]
        if not allowed:
            return logits
        mask = torch.full_like(masked, -1e9)
        mask[:, allowed] = 0.0
        return masked + mask

    def _compute_severity_score(
        self,
        meta: Dict[str, Any],
        confidence: float,
        head_sev: float,
        regions: List[Dict[str, float]],
    ) -> int:
        if not meta["is_fault"]:
            return int(np.clip(round((1.0 - confidence) * 20.0 + head_sev * 10.0), 1, 18))
        area_bonus = min(12.0, sum(r["area_pct"] for r in regions) * 0.45)
        raw = 0.55 * meta["base_severity"] + 0.30 * (head_sev * 100.0) + 0.15 * (confidence * 100.0) + area_bonus * 0.25
        return int(np.clip(round(raw), 35, 99))

    def inspect_image(
        self,
        image_path: Path,
        out_dir: Path,
        inspection_id: Optional[str] = None,
        domain_filter: str = "all",
        active_model: str = "builtin",
    ) -> Dict[str, Any]:
        inspection_id = inspection_id or uuid.uuid4().hex[:12]
        out_dir.mkdir(parents=True, exist_ok=True)

        pil_img = Image.open(image_path).convert("RGB")
        orig_rgb = np.array(pil_img)
        # Keep display image at crisp bounded size (max 768px)
        h, w = orig_rgb.shape[:2]
        if max(h, w) > 768:
            scale = 768.0 / max(h, w)
            orig_rgb = cv2.resize(orig_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with self._lock:
            model, is_custom = self._get_model_and_meta(active_model)
            explainer = GradCAMExplainer(model.spatial)
            try:
                with torch.no_grad():
                    out_init = model.forward_image(tensor)
                    masked_logits = self._apply_domain_mask(out_init["logits"], domain_filter, is_custom)
                    probs = F.softmax(masked_logits, dim=-1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    head_sev = float(out_init["severity"][0].item())

                heatmap, _, _ = explainer.generate(tensor, target_class=pred_idx)
            finally:
                explainer.close()

        probs_np = probs.detach().cpu().numpy()
        meta = self._class_meta(pred_idx, is_custom)
        confidence = float(probs_np[pred_idx])

        gradcam_rgb = render_calm_overlay(orig_rgb, heatmap, alpha=0.54)
        localized_rgb, regions = extract_and_draw_regions(
            orig_rgb, heatmap, is_fault=meta["is_fault"], threshold=0.46
        )
        severity_score = self._compute_severity_score(meta, confidence, head_sev, regions)

        orig_out = out_dir / f"{inspection_id}_original.png"
        cam_out = out_dir / f"{inspection_id}_gradcam.png"
        loc_out = out_dir / f"{inspection_id}_localized.png"

        Image.fromarray(orig_rgb).save(orig_out)
        Image.fromarray(gradcam_rgb).save(cam_out)
        Image.fromarray(localized_rgb).save(loc_out)

        # Build sorted probability list
        num_c = len(probs_np)
        prob_items = []
        for idx in range(num_c):
            cm = self._class_meta(idx, is_custom)
            if not is_custom and domain_filter != "all" and cm["domain"] != domain_filter:
                continue
            prob_items.append(
                {
                    "class_id": cm["id"],
                    "label_en": cm["label_en"],
                    "label_ar": cm["label_ar"],
                    "domain": cm["domain"],
                    "probability": round(float(probs_np[idx]), 4),
                    "severity_level": cm["severity_level"],
                    "color": cm["color"],
                }
            )
        prob_items.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "id": inspection_id,
            "media_type": "image",
            "model_mode": "custom" if is_custom else "builtin",
            "domain_filter": domain_filter,
            "predicted_class": meta["id"],
            "domain": meta["domain"],
            "label_en": meta["label_en"],
            "label_ar": meta["label_ar"],
            "is_fault": meta["is_fault"],
            "severity_level": meta["severity_level"],
            "severity_score": severity_score,
            "confidence": round(confidence, 4),
            "description_en": meta["description_en"],
            "description_ar": meta["description_ar"],
            "recommendation_en": meta["recommendation_en"],
            "recommendation_ar": meta["recommendation_ar"],
            "regions": regions,
            "probabilities": prob_items[:6],
            "assets": {
                "original": f"/artifacts/outputs/{orig_out.name}",
                "gradcam": f"/artifacts/outputs/{cam_out.name}",
                "localized": f"/artifacts/outputs/{loc_out.name}",
            },
            "file_paths": {
                "original": str(orig_out),
                "gradcam": str(cam_out),
                "localized": str(loc_out),
            },
        }

    def inspect_video(
        self,
        video_path: Path,
        out_dir: Path,
        inspection_id: Optional[str] = None,
        sample_frames: int = 16,
        domain_filter: str = "all",
        active_model: str = "builtin",
    ) -> Dict[str, Any]:
        inspection_id = inspection_id or uuid.uuid4().hex[:12]
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.0)
        if fps <= 0.1 or fps > 120:
            fps = 12.0
        raw_frames_rgb: List[np.ndarray] = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raw_frames_rgb.append(rgb)
        cap.release()

        if not raw_frames_rgb:
            raise ValueError("Uploaded video contains zero readable frames.")

        total_frames = len(raw_frames_rgb)
        duration_sec = round(total_frames / fps, 2)
        sample_count = min(sample_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()

        sampled_rgb = [raw_frames_rgb[i] for i in indices]
        tensors = [self.transform(Image.fromarray(f)) for f in sampled_rgb]
        clip_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        with self._lock:
            model, is_custom = self._get_model_and_meta(active_model)
            with torch.no_grad():
                v_out = model.forward_video(clip_tensor)
                clip_logits = self._apply_domain_mask(v_out["clip_logits"], domain_filter, is_custom)
                clip_probs = F.softmax(clip_logits, dim=-1)[0]
                pred_idx = int(torch.argmax(clip_probs).item())
                clip_head_sev = float(v_out["clip_severity"][0].item())
                temp_weights = v_out["temporal_weights"][0].detach().cpu().numpy()

                frame_logits = self._apply_domain_mask(
                    v_out["frame_logits"][0], domain_filter, is_custom
                )
                frame_probs = F.softmax(frame_logits, dim=-1).detach().cpu().numpy()
                frame_sev = v_out["frame_severity"][0].detach().cpu().numpy()

        meta = self._class_meta(pred_idx, is_custom)
        clip_probs_np = clip_probs.detach().cpu().numpy()
        confidence = float(clip_probs_np[pred_idx])

        # Build timeline per sampled frame
        timeline: List[Dict[str, Any]] = []
        for s_idx, orig_idx in enumerate(indices):
            f_pred = int(np.argmax(frame_probs[s_idx]))
            f_meta = self._class_meta(f_pred, is_custom)
            f_conf = float(frame_probs[s_idx, f_pred])
            target_prob = float(frame_probs[s_idx, pred_idx])
            timeline.append(
                {
                    "sample_index": s_idx,
                    "frame_index": int(orig_idx),
                    "timestamp_sec": round(orig_idx / fps, 2),
                    "predicted_class": f_meta["id"],
                    "label_en": f_meta["label_en"],
                    "label_ar": f_meta["label_ar"],
                    "is_fault": f_meta["is_fault"],
                    "confidence": round(f_conf, 4),
                    "target_class_prob": round(target_prob, 4),
                    "temporal_attention": round(float(temp_weights[s_idx]), 4),
                    "severity_score": int(np.clip(round(frame_sev[s_idx] * 100), 1, 99)),
                }
            )

        # Identify top 3 peak frames (attention * target probability) for Grad-CAM keyframes
        rank_scores = [
            (s_idx, float(temp_weights[s_idx]) * float(frame_probs[s_idx, pred_idx]) + (0.15 if timeline[s_idx]["is_fault"] else 0.0))
            for s_idx in range(sample_count)
        ]
        rank_scores.sort(key=lambda x: x[1], reverse=True)
        top_sample_indices = [x[0] for x in rank_scores[:3]]
        peak_sample_idx = top_sample_indices[0]

        keyframes: List[Dict[str, Any]] = []
        peak_regions: List[Dict[str, float]] = []
        peak_orig_path = ""
        peak_cam_path = ""
        peak_loc_path = ""

        with self._lock:
            model, _ = self._get_model_and_meta(active_model)
            explainer = GradCAMExplainer(model.spatial)
            try:
                for rank_i, s_idx in enumerate(top_sample_indices):
                    fr_rgb = sampled_rgb[s_idx]
                    fh, fw = fr_rgb.shape[:2]
                    if max(fh, fw) > 640:
                        sc = 640.0 / max(fh, fw)
                        fr_rgb = cv2.resize(fr_rgb, (int(fw * sc), int(fh * sc)), interpolation=cv2.INTER_AREA)

                    fr_t = tensors[s_idx].unsqueeze(0).to(self.device)
                    hm, _, _ = explainer.generate(fr_t, target_class=pred_idx)
                    cam_rgb = render_calm_overlay(fr_rgb, hm, alpha=0.52)
                    loc_rgb, regs = extract_and_draw_regions(
                        fr_rgb, hm, is_fault=meta["is_fault"], threshold=0.46
                    )

                    kf_orig = out_dir / f"{inspection_id}_kf{rank_i}_orig.png"
                    kf_cam = out_dir / f"{inspection_id}_kf{rank_i}_cam.png"
                    kf_loc = out_dir / f"{inspection_id}_kf{rank_i}_loc.png"
                    Image.fromarray(fr_rgb).save(kf_orig)
                    Image.fromarray(cam_rgb).save(kf_cam)
                    Image.fromarray(loc_rgb).save(kf_loc)

                    if rank_i == 0:
                        peak_regions = regs
                        peak_orig_path = str(kf_orig)
                        peak_cam_path = str(kf_cam)
                        peak_loc_path = str(kf_loc)

                    keyframes.append(
                        {
                            "rank": rank_i + 1,
                            "frame_index": timeline[s_idx]["frame_index"],
                            "timestamp_sec": timeline[s_idx]["timestamp_sec"],
                            "attention_weight": timeline[s_idx]["temporal_attention"],
                            "confidence": timeline[s_idx]["confidence"],
                            "regions": regs,
                            "original_url": f"/artifacts/outputs/{kf_orig.name}",
                            "gradcam_url": f"/artifacts/outputs/{kf_cam.name}",
                            "localized_url": f"/artifacts/outputs/{kf_loc.name}",
                        }
                    )
            finally:
                explainer.close()

        # Render annotated H.264 video with calm zero-blue HUD
        annotated_mp4 = out_dir / f"{inspection_id}_annotated.mp4"
        self._render_annotated_video(
            raw_frames_rgb=raw_frames_rgb,
            sample_indices=indices,
            timeline=timeline,
            overall_meta=meta,
            overall_conf=confidence,
            fps=fps,
            out_path=annotated_mp4,
        )

        severity_score = self._compute_severity_score(meta, confidence, clip_head_sev, peak_regions)

        prob_items = []
        for idx in range(len(clip_probs_np)):
            cm = self._class_meta(idx, is_custom)
            if not is_custom and domain_filter != "all" and cm["domain"] != domain_filter:
                continue
            prob_items.append(
                {
                    "class_id": cm["id"],
                    "label_en": cm["label_en"],
                    "label_ar": cm["label_ar"],
                    "domain": cm["domain"],
                    "probability": round(float(clip_probs_np[idx]), 4),
                    "severity_level": cm["severity_level"],
                    "color": cm["color"],
                }
            )
        prob_items.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "id": inspection_id,
            "media_type": "video",
            "model_mode": "custom" if is_custom else "builtin",
            "domain_filter": domain_filter,
            "predicted_class": meta["id"],
            "domain": meta["domain"],
            "label_en": meta["label_en"],
            "label_ar": meta["label_ar"],
            "is_fault": meta["is_fault"],
            "severity_level": meta["severity_level"],
            "severity_score": severity_score,
            "confidence": round(confidence, 4),
            "description_en": meta["description_en"],
            "description_ar": meta["description_ar"],
            "recommendation_en": meta["recommendation_en"],
            "recommendation_ar": meta["recommendation_ar"],
            "video_stats": {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "sampled_frames": sample_count,
                "peak_timestamp_sec": timeline[peak_sample_idx]["timestamp_sec"],
                "peak_frame_index": timeline[peak_sample_idx]["frame_index"],
            },
            "regions": peak_regions,
            "probabilities": prob_items[:6],
            "timeline": timeline,
            "keyframes": keyframes,
            "assets": {
                "annotated_video": f"/artifacts/outputs/{annotated_mp4.name}",
                "original": keyframes[0]["original_url"],
                "gradcam": keyframes[0]["gradcam_url"],
                "localized": keyframes[0]["localized_url"],
            },
            "file_paths": {
                "original": peak_orig_path,
                "gradcam": peak_cam_path,
                "localized": peak_loc_path,
                "annotated_video": str(annotated_mp4),
            },
        }

    def _render_annotated_video(
        self,
        raw_frames_rgb: List[np.ndarray],
        sample_indices: List[int],
        timeline: List[Dict[str, Any]],
        overall_meta: Dict[str, Any],
        overall_conf: float,
        fps: float,
        out_path: Path,
    ) -> None:
        h, w = raw_frames_rgb[0].shape[:2]
        # Ensure even dimensions >= 360px for crisp HUD & libx264 compatibility
        scale = max(1.0, 420.0 / max(h, w))
        vw = int(w * scale) // 2 * 2
        vh = int(h * scale) // 2 * 2

        tmp_raw = out_path.with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))

        sage_rgb = (78, 122, 101)
        amber_rgb = (179, 131, 62)
        terracotta_rgb = (184, 93, 78)
        cream_rgb = (250, 249, 246)
        charcoal_rgb = (36, 38, 34)

        sample_arr = np.array(sample_indices)
        total = len(raw_frames_rgb)

        for f_idx, fr_rgb in enumerate(raw_frames_rgb):
            frame = cv2.resize(fr_rgb, (vw, vh), interpolation=cv2.INTER_AREA)
            nearest_pos = int(np.argmin(np.abs(sample_arr - f_idx)))
            t_info = timeline[nearest_pos]

            badge_color = (
                terracotta_rgb
                if t_info["is_fault"] and overall_meta["severity_level"] == "critical"
                else (amber_rgb if t_info["is_fault"] else sage_rgb)
            )

            # Calm top HUD strip (cream semi-transparent bar, zero blue)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (vw, 38), cream_rgb, -1)
            frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
            cv2.line(frame, (0, 38), (vw, 38), badge_color, 2)

            # Status dot + label
            cv2.circle(frame, (14, 19), 5, badge_color, -1, cv2.LINE_AA)
            hud_text = f"{t_info['label_en']} | {int(t_info['confidence'] * 100)}%"
            cv2.putText(
                frame,
                hud_text[:38],
                (26, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                charcoal_rgb,
                1,
                cv2.LINE_AA,
            )

            # Right side temporal attention badge
            att_txt = f"ATTN {int(t_info['temporal_attention'] * 100)}%"
            cv2.putText(
                frame,
                att_txt,
                (max(vw - 95, 10), 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                badge_color,
                1,
                cv2.LINE_AA,
            )

            # Bottom progress scrub bar
            prog_w = int(((f_idx + 1) / max(1, total)) * vw)
            cv2.rectangle(frame, (0, vh - 5), (vw, vh), (234, 231, 224), -1)
            cv2.rectangle(frame, (0, vh - 5), (prog_w, vh), badge_color, -1)

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

        # Transcode to H.264 yuv420p for universal HTML5 <video> compatibility
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_raw),
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-loglevel",
                    "error",
                    str(out_path),
                ],
                check=True,
                timeout=30,
            )
            if tmp_raw.exists():
                tmp_raw.unlink()
        except Exception:
            if tmp_raw.exists():
                tmp_raw.rename(out_path)
