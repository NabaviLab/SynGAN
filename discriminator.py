import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

class Discriminator(nn.Module):
    """
    Swin Transformer–based discriminator (as specified).
    Input is a 3-channel image formed by concatenating:
        [X_prior, X_current, X_{current or synthetic}]
    A 1×1 conv is applied first for channel adjustment. If timm is
    unavailable, we fall back to a small CNN (kept minimal).
    """
    def __init__(self, swin_name: str = "swin_tiny_patch4_window7_224"):
        super().__init__()
        self.pre_1x1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

        if _HAS_TIMM:
            self.backbone = timm.create_model(swin_name, pretrained=True, num_classes=1)
        else:
            # Fallback: tiny CNN classifier
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 1)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x3: torch.Tensor) -> torch.Tensor:
        """
        x3: (B, 3, H, W)
        returns y ∈ (0,1): probability of 'real'
        """
        x = self.pre_1x1(x3)
        logits = self.backbone(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        y = self.sigmoid(logits)
        return y.view(-1, 1)