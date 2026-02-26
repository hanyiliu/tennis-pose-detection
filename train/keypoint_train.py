import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.keypoint_dataset import TennisKeypointDataset
from models.keypoint_detection import KeypointDetectionModel
from preprocessing.pil_preprocessing import letterbox_resize


def get_device():
    """Return the best available PyTorch device for training.

    Priority order is CUDA, then Apple MPS, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Compute average loss over one evaluation dataloader.

    Args:
        model: Keypoint model that outputs heatmaps.
        loader: Validation or test dataloader.
        criterion: Loss function used to compare predicted vs target heatmaps.
        device: torch.device where tensors and model are placed.

    Returns:
        Mean loss across all samples in the loader.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    for images, target_heatmaps in loader:
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)

        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, target_heatmaps)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def train_keypoint_model(args):
    """Train, validate, checkpoint, and test the keypoint detection model.

    Args:
        args: Parsed CLI args namespace from `main()`.

    Raises:
        ValueError: If split ratios do not sum to 1.
        RuntimeError: If model output shape and target heatmap shape mismatch.
    """
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    image_size = (args.image_height, args.image_width)
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.ToTensor(),
    ])
    keypoint_transform = KeypointsToHeatmaps(
        out_size=image_size,
        num_keypoints=args.num_keypoints,
        sigma=args.heatmap_sigma,
    )

    dataset = TennisKeypointDataset(
        root_dir=args.root_dir,
        annotation_dir=args.annotation_dir,
        image_transform=image_transform,
        keypoint_transform=keypoint_transform,
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_split)
    val_size = int(dataset_size * args.val_split)
    test_size = dataset_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print("Device:", device)
    print("Train/Val/Test sizes:", len(train_ds), len(val_ds), len(test_ds))

    model = KeypointDetectionModel(num_keypoints=args.num_keypoints).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    images0, heatmaps0 = next(iter(train_loader))
    images0 = images0.to(device)
    heatmaps0 = heatmaps0.to(device)
    pred0 = model(images0)
    if pred0.shape != heatmaps0.shape:
        raise RuntimeError(
            f"Model output shape {tuple(pred0.shape)} does not match target shape {tuple(heatmaps0.shape)}. "
            f"Check image size and keypoint heatmap transform."
        )

    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/keypoint_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for images, target_heatmaps in train_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d}/{args.epochs} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )

    print("Best val loss:", best_val_loss)
    print("Saved checkpoint:", best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.5f}")


def main():
    """Parse CLI arguments and launch keypoint model training.

    Arguments:
        --root_dir: Dataset root containing image and annotation subfolders.
        --annotation_dir: Annotation directory relative to root_dir.
        --epochs: Number of full passes over the training split.
        --batch_size: Number of samples per optimization step.
        --lr: Learning rate for Adam optimizer.
        --weight_decay: L2 regularization factor for optimizer.
        --train_split: Fraction of dataset for training.
        --val_split: Fraction of dataset for validation.
        --test_split: Fraction of dataset for final testing.
        --seed: Random seed for deterministic splitting and initialization.
        --num_keypoints: Number of keypoint heatmap channels predicted by model.
        --image_height: Letterboxed training image/heatmap height in pixels.
        --image_width: Letterboxed training image/heatmap width in pixels.
        --heatmap_sigma: Gaussian spread (pixels) for target keypoint heatmaps.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default="annotations")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_keypoints", type=int, default=18)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)

    args = parser.parse_args()
    train_keypoint_model(args)


if __name__ == "__main__":
    main()