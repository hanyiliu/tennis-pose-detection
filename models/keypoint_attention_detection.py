"""
Keypoint Detection Model — SOTA Redesign
Architecture: HRNet-inspired multi-scale U-Net with residual blocks,
CBAM attention, deep supervision, and no pretrained backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with BatchNorm.

    Uses a 1x1 projection shortcut when channel dimensions differ.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution for efficient decoding."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            ConvBnRelu(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            # Pointwise
            ConvBnRelu(in_ch, out_ch, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention).

    Reference: Woo et al., ECCV 2018.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # --- Channel attention ---
        b, c, h, w = x.shape
        avg = x.mean(dim=[2, 3])               # (B, C)
        mx = x.amax(dim=[2, 3])               # (B, C)
        ca = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        x = x * ca.view(b, c, 1, 1)

        # --- Spatial attention ---
        avg_s = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_s = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Four-stage encoder producing skip features at 1/2, 1/4, 1/8, 1/16 scale."""

    CHANNELS = [64, 128, 256, 512]

    def __init__(self):
        super().__init__()
        # Stem: 3 → 64 at 1/2 resolution
        self.stem = ConvBnRelu(3, self.CHANNELS[0], stride=2)

        # Each stage doubles channels and halves resolution
        self.stage1 = self._make_stage(self.CHANNELS[0], self.CHANNELS[1], stride=2)
        self.stage2 = self._make_stage(self.CHANNELS[1], self.CHANNELS[2], stride=2)
        self.stage3 = self._make_stage(self.CHANNELS[2], self.CHANNELS[3], stride=2)

    @staticmethod
    def _make_stage(in_ch, out_ch, stride):
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride=stride),
            ResidualBlock(out_ch, out_ch),
        )

    def forward(self, x):
        e1 = self.stem(x)           # 1/2
        e2 = self.stage1(e1)        # 1/4
        e3 = self.stage2(e2)        # 1/8
        e4 = self.stage3(e3)        # 1/16
        return e1, e2, e3, e4


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """Two residual blocks at 1/32 scale with ASPP-lite for multi-scale context."""

    def __init__(self, in_ch=512, out_ch=1024):
        super().__init__()
        self.downsample = ResidualBlock(in_ch, out_ch, stride=2)  # → 1/32

        # Atrous Spatial Pyramid Pooling (lite: 3 dilations + pooling branch)
        mid = out_ch // 4
        self.d1 = ConvBnRelu(out_ch, mid, kernel_size=1, padding=0)
        self.d6 = nn.Sequential(
            nn.Conv2d(out_ch, mid, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.d12 = nn.Sequential(
            nn.Conv2d(out_ch, mid, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.global_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_ch, mid, kernel_size=1, padding=0),
        )
        self.project = ConvBnRelu(mid * 4, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.downsample(x)
        b_h, b_w = x.shape[2], x.shape[3]
        gp = F.interpolate(self.global_pool_branch(x), size=(b_h, b_w),
                           mode="bilinear", align_corners=False)
        x = self.project(torch.cat([self.d1(x), self.d6(x), self.d12(x), gp], dim=1))
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderStage(nn.Module):
    """Single decoder stage: upsample → concat skip → CBAM → depthwise-sep conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = DepthwiseSeparableConv(out_ch + skip_ch, out_ch)
        self.attn = CBAM(out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.attn(x)


class Decoder(nn.Module):
    """Four-stage decoder restoring resolution from 1/32 back to full scale."""

    def __init__(self, bottleneck_ch=1024, encoder_channels=(64, 128, 256, 512)):
        super().__init__()
        e1, e2, e3, e4 = encoder_channels

        # 1/32 → 1/16
        self.stage1 = DecoderStage(bottleneck_ch, e4, 512)
        # 1/16 → 1/8
        self.stage2 = DecoderStage(512, e3, 256)
        # 1/8 → 1/4
        self.stage3 = DecoderStage(256, e2, 128)
        # 1/4 → 1/2
        self.stage4 = DecoderStage(128, e1, 64)
        # 1/2 → full resolution (no skip)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 32),
        )

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        d1 = self.stage1(bottleneck, e4)
        d2 = self.stage2(d1, e3)
        d3 = self.stage3(d2, e2)
        d4 = self.stage4(d3, e1)
        d5 = self.final_upsample(d4)
        return d1, d2, d3, d4, d5   # return all scales for deep supervision


# ---------------------------------------------------------------------------
# Deep Supervision Heads
# ---------------------------------------------------------------------------

class SupervisionHead(nn.Module):
    """Lightweight 1×1 head that maps intermediate features to keypoint heatmaps."""

    def __init__(self, in_ch, num_keypoints):
        super().__init__()
        self.head = nn.Conv2d(in_ch, num_keypoints, kernel_size=1)

    def forward(self, x, target_size):
        x = self.head(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class KeypointAttentionDetectionModel(nn.Module):
    """SOTA Encoder-Decoder model for player keypoint detection.

    Architecture highlights:
    - Pre-activation residual encoder with BatchNorm throughout
    - ASPP-lite bottleneck for multi-scale context at lowest resolution
    - CBAM attention gates in every decoder stage
    - Depthwise-separable convolutions in the decoder for efficiency
    - Deep supervision: auxiliary heatmap heads at 1/8, 1/4, 1/2 scales
      (auxiliary heads are only active during training)

    Model Input:
        Cropped player image as (batch_size, 3, H, W).
        H and W must be divisible by 32. Minimum size: 32×32.

    Model Output (training):
        Tuple of (main_heatmap, aux1, aux2, aux3) — all at full input resolution.
        Use a weighted sum for loss: L = L_main + 0.4*L_aux1 + 0.2*L_aux2 + 0.1*L_aux3

    Model Output (eval):
        Single heatmap tensor of shape (batch_size, num_keypoints, H, W).
        Apply sigmoid to convert logits to [0, 1] confidence maps.

    Usage:
        # Inference
        model = KeypointDetectionModel()
        model.eval()
        with torch.no_grad():
            heatmap = model(input)   # (B, 18, H, W)
            confidence = torch.sigmoid(heatmap)

        # Training
        model = KeypointDetectionModel()
        model.train()
        outputs = model(input)       # tuple of 4 heatmaps
        main, aux1, aux2, aux3 = outputs
        loss = (criterion(main, target)
                + 0.4 * criterion(aux1, target)
                + 0.2 * criterion(aux2, target)
                + 0.1 * criterion(aux3, target))
    """

    def __init__(self, num_keypoints: int = 18):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck(in_ch=512, out_ch=1024)
        self.decoder = Decoder(
            bottleneck_ch=1024,
            encoder_channels=(64, 128, 256, 512),
        )

        # Main output head (full resolution)
        self.main_head = nn.Conv2d(32, num_keypoints, kernel_size=1)

        # Auxiliary supervision heads (intermediate decoder scales)
        self.aux_head1 = SupervisionHead(512, num_keypoints)   # 1/16
        self.aux_head2 = SupervisionHead(256, num_keypoints)   # 1/8
        self.aux_head3 = SupervisionHead(128, num_keypoints)   # 1/4

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W). H and W must be divisible by 32.

        Returns:
            Training mode: tuple (main, aux1, aux2, aux3) — all (B, K, H, W).
            Eval mode:     main heatmap tensor (B, K, H, W).

        Raises:
            ValueError: If input dimensions are invalid.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )
        _, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"Expected 3 input channels (RGB), got {c}")
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"Height and width must be divisible by 32, got ({h}, {w})"
            )

        target_size = (h, w)

        # Encode
        skips = self.encoder(x)           # e1…e4

        # Bottleneck
        b = self.bottleneck(skips[-1])

        # Decode — returns (d1@1/16, d2@1/8, d3@1/4, d4@1/2, d5@1/1)
        d1, d2, d3, d4, d5 = self.decoder(b, skips)

        # Main prediction
        main = self.main_head(d5)

        if not self.training:
            return main

        # Auxiliary predictions (deep supervision — training only)
        aux1 = self.aux_head1(d1, target_size)
        aux2 = self.aux_head2(d2, target_size)
        aux3 = self.aux_head3(d3, target_size)

        return main, aux1, aux2, aux3