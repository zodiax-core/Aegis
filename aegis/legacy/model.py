import torch.nn as nn
import torchvision.models as tvm

def build_backbone(name: str, freeze: bool = True):
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    else:
        # fine-tune only the last residual block, keep everything else frozen
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True
    return m, feat_dim


class BaselinePoolingModel(nn.Module):
    """Baseline A: backbone -> mean-pool over time -> classifier heads."""

    def __init__(self, cfg: AegisConfig):
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg.backbone_name, cfg.freeze_backbone)
        self.event_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, cfg.num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, clip):  # clip: (B, T, C, H, W)
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)
        feats = self.backbone(x).view(B, T, -1)   # (B, T, feat_dim)
        pooled = feats.mean(dim=1)                # (B, feat_dim)
        event_logits = self.event_head(pooled)
        anomaly_logit = self.anomaly_head(pooled).squeeze(-1)
        return event_logits, anomaly_logit


print("BaselinePoolingModel defined.")



# CELL NO 9

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):  # x: (B, T, dim)
        return x + self.pe[:, :x.size(1), :]


class AegisTemporalModel(nn.Module):
    """Primary AEGIS event/anomaly detector."""

    def __init__(self, cfg: AegisConfig):
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg.backbone_name, cfg.freeze_backbone)
        self.project = nn.Linear(feat_dim, cfg.temporal_dim)
        self.pos_enc = PositionalEncoding(cfg.temporal_dim, max_len=cfg.clip_length + 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.temporal_dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.temporal_ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.temporal_layers)

        # learned CLS-style token that attends to the whole clip and is used for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.temporal_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.event_head = nn.Sequential(
            nn.LayerNorm(cfg.temporal_dim),
            nn.Linear(cfg.temporal_dim, 256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, cfg.num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(cfg.temporal_dim),
            nn.Linear(cfg.temporal_dim, 128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, clip, object_features=None):
        """clip: (B, T, C, H, W). object_features: optional (B, obj_dim) to concat post-hoc
        by the fusion model in Cell Group 12 — this model stays clip-only by design so it can
        be evaluated standalone as Baseline A/C in the ablation table."""
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            feats = self.backbone(x).view(B, T, -1)
        tokens = self.project(feats)                      # (B, T, temporal_dim)
        tokens = self.pos_enc(tokens)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, T+1, temporal_dim)

        encoded = self.temporal_encoder(tokens)
        pooled = encoded[:, 0, :]                           # take the CLS token's output

        event_logits = self.event_head(pooled)
        anomaly_logit = self.anomaly_head(pooled).squeeze(-1)
        return event_logits, anomaly_logit, pooled


print("AegisTemporalModel defined "
      f"(backbone={cfg.backbone_name}, temporal_dim={cfg.temporal_dim}, "
      f"layers={cfg.temporal_layers}, heads={cfg.temporal_heads}).")
