"""
models/wavkan_head.py
─────────────────────
MODULE 3 — WavKAN_Head (pixel-wise KAN classifier).

Attempts to use efficient-kan's KANLinear for a learnable
non-linear feature→class mapping applied independently to every
spatial position.  Falls back gracefully to a plain 1×1 Conv2d
if efficient-kan is not installed.

Input : (B, in_features, H, W)    — dense feature map
Output: (B, num_classes, H, W)    — logits at the same resolution
"""

import warnings
import torch
import torch.nn as nn

# Unified KAN symbol resolution: try both known import paths
_KANLinear = None
_KAN_AVAILABLE = False
try:
    from efficient_kan import KANLinear as _KANLinear  # primary path
    _KAN_AVAILABLE = True
except ImportError:
    try:
        from kan import KANLinear as _KANLinear        # alternative path
        _KAN_AVAILABLE = True
    except ImportError:
        try:
            from kan import KAN as _KANLinear           # KAN([sizes]) API fallback
            _KAN_AVAILABLE = True
        except ImportError:
            _KAN_AVAILABLE = False
            warnings.warn(
                "[WavKAN_Head] efficient-kan not found.  "
                "Falling back to a standard 1×1 Conv2d classifier.  "
                "Install with:  pip install efficient-kan",
                stacklevel=2,
            )


class _KANPixelHead(nn.Module):
    """Pixel-wise KAN head using efficient-kan's KANLinear."""

    def __init__(self, in_features: int, num_classes: int, grid_size: int = 5):
        super().__init__()
        # Try KANLinear-style API (takes positional in/out + grid_size)
        try:
            self.kan = _KANLinear(in_features, num_classes, grid_size=grid_size)
        except TypeError:
            # Some versions use KAN([layer_sizes]) list-of-widths API
            self.kan = _KANLinear([in_features, num_classes])
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        returns: (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        # Reshape to (B*H*W, C), apply KAN, reshape back
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
        out_flat = self.kan(x_flat)                       # (B*H*W, num_classes)
        out = out_flat.reshape(B, H, W, self.num_classes) \
                      .permute(0, 3, 1, 2)                # (B, num_classes, H, W)
        return out


class _ConvPixelHead(nn.Module):
    """Fallback: plain 1×1 convolution classifier."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WavKAN_Head(nn.Module):
    """
    Pixel-wise classification head.

    Parameters
    ----------
    in_features : number of input feature channels (e.g. 64)
    num_classes : number of segmentation classes
    grid_size   : KAN grid size (only used when KAN is available)
    """

    def __init__(
        self,
        in_features: int = 64,
        num_classes: int = 6,
        grid_size: int = 5,
    ):
        super().__init__()
        if _KAN_AVAILABLE:
            try:
                self.head = _KANPixelHead(in_features, num_classes, grid_size)
                self.using_kan = True
            except Exception as e:
                warnings.warn(
                    f"[WavKAN_Head] KAN instantiation failed ({e}).  "
                    "Falling back to 1×1 Conv2d.",
                    stacklevel=2,
                )
                self.head = _ConvPixelHead(in_features, num_classes)
                self.using_kan = False
        else:
            self.head = _ConvPixelHead(in_features, num_classes)
            self.using_kan = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, in_features, H, W)
        returns: (B, num_classes, H, W)
        """
        return self.head(x)
