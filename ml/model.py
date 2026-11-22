"""Spatio-Temporal Deep Learning Architecture for Multi-Domain Fault Classification."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for subtle defect feature amplification."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class FaultSenseSpatialEncoder(nn.Module):
    """Spatial backbone with SE attention, classification head, and severity regression head."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            feat_dim = 512
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            feat_dim = 576

        self.feat_dim = feat_dim
        self.se_refine = SEBlock(feat_dim, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embed_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.severity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def get_target_layer(self) -> nn.Module:
        """Returns the final spatial convolutional feature layer for Grad-CAM."""
        return self.features[-1]

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.features(x)
        fmap_refined = self.se_refine(fmap)
        pooled = self.pool(fmap_refined).flatten(1)
        emb = self.embed_proj(pooled)
        return emb, fmap_refined

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb, _ = self.forward_features(x)
        logits = self.classifier(emb)
        severity = self.severity_head(emb).squeeze(-1)
        return {
            "logits": logits,
            "severity": severity,
            "embedding": emb,
        }


class TemporalAttentionAggregator(nn.Module):
    """Transformer-based temporal encoder with attention pooling for video clips."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 10,
        max_frames: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.clip_classifier = nn.Linear(embed_dim, num_classes)
        self.clip_severity = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # frame_embeddings: [B, T, D]
        B, T, D = frame_embeddings.shape
        pos = self.pos_embed[:, :T, :]
        seq = self.temporal_encoder(frame_embeddings + pos)

        # Attention pooling highlights fault-bearing frames over benign frames
        raw_weights = self.attn_score(seq).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(raw_weights, dim=-1)  # [B, T]
        pooled = torch.bmm(attn_weights.unsqueeze(1), seq).squeeze(1)  # [B, D]

        clip_logits = self.clip_classifier(pooled)
        if frame_logits is not None:
            # Blend attention-weighted per-frame evidence with temporal sequence reasoning
            weighted_frame_logits = torch.bmm(attn_weights.unsqueeze(1), frame_logits).squeeze(1)
            clip_logits = 0.65 * clip_logits + 0.35 * weighted_frame_logits

        clip_severity = self.clip_severity(pooled).squeeze(-1)
        return {
            "clip_logits": clip_logits,
            "clip_severity": clip_severity,
            "temporal_weights": attn_weights,
        }


class FaultSenseTemporalNet(nn.Module):
    """Unified Image + Video Deep Learning Fault Classification Model."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.spatial = FaultSenseSpatialEncoder(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        self.temporal = TemporalAttentionAggregator(
            embed_dim=256,
            num_classes=num_classes,
        )

    def forward_image(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.spatial(x)

    def forward_video(self, clips: torch.Tensor) -> Dict[str, torch.Tensor]:
        # clips: [B, T, C, H, W]
        B, T, C, H, W = clips.shape
        flat = clips.view(B * T, C, H, W)
        spatial_out = self.spatial(flat)
        frame_emb = spatial_out["embedding"].view(B, T, -1)
        frame_logits = spatial_out["logits"].view(B, T, -1)
        frame_severity = spatial_out["severity"].view(B, T)

        temp_out = self.temporal(frame_emb, frame_logits)
        return {
            "clip_logits": temp_out["clip_logits"],
            "clip_severity": temp_out["clip_severity"],
            "temporal_weights": temp_out["temporal_weights"],
            "frame_logits": frame_logits,
            "frame_severity": frame_severity,
        }
