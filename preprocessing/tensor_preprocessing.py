# utils/keypoints.py

import torch


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
