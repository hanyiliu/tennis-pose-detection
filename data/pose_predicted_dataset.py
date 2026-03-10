# data/pose_predicted_dataset.py

'''
loads the cached .pt file built by the previous script and returns predicted keypoints, and labels.
'''

import torch
from torch.utils.data import Dataset


class pose_predicted_dataset(Dataset):
    """
    Loads cached Stage 2 predicted keypoints for Stage 3 training.

    Expected saved file format:
        {
            "keypoints": Tensor (N, 18, 3),
            "labels": Tensor (N,)
        }

    Returns:
        keypoints: FloatTensor (18,3)
        label: LongTensor scalar
    """

    def __init__(self, feature_path: str):
        data = torch.load(feature_path, map_location="cpu")

        self.keypoints = data["keypoints"].float()
        self.labels = data["labels"].long()

        if self.keypoints.dim() != 3 or self.keypoints.size(1) != 18 or self.keypoints.size(2) != 3:
            raise ValueError(f"Expected keypoints shape (N,18,3), got {tuple(self.keypoints.shape)}")

        if self.labels.dim() != 1:
            raise ValueError(f"Expected labels shape (N,), got {tuple(self.labels.shape)}")

        if self.keypoints.size(0) != self.labels.size(0):
            raise ValueError("Keypoints and labels have mismatched number of samples.")

    def __len__(self):
        return self.keypoints.size(0)

    def __getitem__(self, index):
        return self.keypoints[index], self.labels[index]