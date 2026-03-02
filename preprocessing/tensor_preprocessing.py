# utils/keypoints.py

import torch
from typing import Tuple
import numpy as np


def heatmaps_to_keypoints(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Convert heatmaps (B, K, H, W) -> keypoints (B, K, 3) with [x, y, visibility].

    - x, y are pixel coordinates in heatmap space
    - visibility is the maximum heatmap value for that keypoint channel

    Args:
        heatmaps: torch.Tensor of shape (B, K, H, W)
        B = batch size, K = number of keypoints, H/W = heatmap height/width

    Returns:
        keypoints: torch.Tensor of shape (B, K, 3)
        B = batch size, K = number of keypoints, 3 = (x, y, visibility)
    """
    if heatmaps.dim() != 4: # ensures heatmap is correct shape.
        raise ValueError(f"Expected heatmaps shape (B,K,H,W). Got {tuple(heatmaps.shape)}")

    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)          # (B,K,H*W), flatten into 1D vector of length H*W for each keypoint channel
    vis, idx = flat.max(dim=-1)             # (B,K), (B,K), get max value and index for each keypoint channel (most likely joint location)

    y = (idx // W).float() # reshape back into 2D grid. 
    x = (idx %  W).float()

    keypoints = torch.stack([x, y, vis], dim=-1)  # (B,K,3), where last dim is (x,y,visibility) for each keypoint channel
    return keypoints # (x,y,visibility) for each keypoint channel, where visibility is the max heatmap value (confidence) for that keypoint.


def normalize_keypoints_xy(keypoints: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Normalize x,y of keypoints from pixel coords to [0,1].

    Args:
        keypoints: torch.Tensor of shape (B,K,3) or (K,3)
        H, W: heatmap height/width used for x,y scaling

    Returns:
        normalized keypoints tensor with same shape as input
    """
    if keypoints.dim() == 2:
        keypoints = keypoints.unsqueeze(0)  # (1,K,3), allows single sample input of shape (K,3) to be processed as batch of size 1

    if keypoints.dim() != 3 or keypoints.size(-1) != 3: # ensures keypoints have correct shape and last dimension is (x,y,visibility)
        raise ValueError(f"Expected keypoints shape (B,K,3) or (K,3). Got {tuple(keypoints.shape)}")

    out = keypoints.clone() # normalizing x and y, dividing x and y respectively by width range and height range.
    out[..., 0] = out[..., 0] / max(W - 1, 1)  # x
    out[..., 1] = out[..., 1] / max(H - 1, 1)  # y
    return out # normalized keypoints with x,y in [0,1] range, visibility unchanged.

class KeypointsToHeatmaps:
    """Convert bbox-local keypoints into gaussian heatmap targets.

    This transform maps normalized keypoints from cropped-player coordinates to
    a letterboxed output canvas, then renders one gaussian peak per visible
    keypoint channel.
    """

    def __init__(self, out_size: Tuple[int, int], num_keypoints: int = 18, sigma: float = 2.0):
        """Initialize heatmap transform settings.

        Args:
            out_size: Output heatmap size as (height, width).
            num_keypoints: Number of keypoint channels to generate.
            sigma: Standard deviation for each gaussian peak in pixels.
        """
        self.out_h, self.out_w = out_size
        self.num_keypoints = num_keypoints
        self.sigma = sigma

    def _draw_gaussian(self, heatmap: np.ndarray, center_x: float, center_y: float):
        """Draw a gaussian peak at the specified center on a single heatmap.

        Args:
            heatmap: 2D heatmap array to update in-place.
            center_x: X coordinate of gaussian center in heatmap space.
            center_y: Y coordinate of gaussian center in heatmap space.
        """
        radius = max(int(3 * self.sigma), 1)
        x0 = max(int(center_x) - radius, 0)
        x1 = min(int(center_x) + radius + 1, self.out_w)
        y0 = max(int(center_y) - radius, 0)
        y1 = min(int(center_y) + radius + 1, self.out_h)

        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        gaussian = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * (self.sigma ** 2)))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

    def __call__(self, keypoints: np.ndarray, bbox_w: float, bbox_h: float) -> np.ndarray:
        """Build target heatmaps from normalized bbox-local keypoints.

        Args:
            keypoints: Array of shape (K, 3) with [x_norm, y_norm, visibility].
            bbox_w: Width of the uncropped bbox region in pixels.
            bbox_h: Height of the uncropped bbox region in pixels.

        Returns:
            Heatmaps as a numpy array of shape (num_keypoints, out_h, out_w).
        """
        heatmaps = np.zeros((self.num_keypoints, self.out_h, self.out_w), dtype=np.float32)
        if bbox_w <= 0 or bbox_h <= 0:
            return heatmaps

        scale = min(self.out_w / bbox_w, self.out_h / bbox_h)
        resized_w = bbox_w * scale
        resized_h = bbox_h * scale
        pad_x = (self.out_w - resized_w) / 2.0
        pad_y = (self.out_h - resized_h) / 2.0

        count = min(self.num_keypoints, keypoints.shape[0])
        for i in range(count):
            x_norm, y_norm, visibility = keypoints[i]
            if visibility <= 0:
                continue

            x_crop = float(np.clip(x_norm, 0.0, 1.0)) * bbox_w
            y_crop = float(np.clip(y_norm, 0.0, 1.0)) * bbox_h

            x_out = x_crop * scale + pad_x
            y_out = y_crop * scale + pad_y
            self._draw_gaussian(heatmaps[i], x_out, y_out)

        return heatmaps

