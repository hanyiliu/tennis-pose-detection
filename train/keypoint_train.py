import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from data.keypoint_dataset import TennisKeypointDataset
from models.keypoint_detection import KeypointDetectionModel
from preprocessing.pil_preprocessing import letterbox_resize
from preprocessing.tensor_preprocessing import KeypointsToHeatmaps


def get_device():
    """Return the best available PyTorch device for training.

    Priority order is CUDA, then Apple MPS, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def heatmap_keypoint_accuracy(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    normalized_distance_threshold: float = 0.05,
) -> float:
    """Compute keypoint localization accuracy from predicted and target heatmaps.

    A keypoint is counted correct if its predicted argmax location is within
    `normalized_distance_threshold` (in normalized image space) of the target
    argmax location. Only visible keypoints (target heatmap max > 0) are scored.

    Args:
        pred_heatmaps: Predicted heatmaps of shape (B, K, H, W).
        target_heatmaps: Ground-truth heatmaps of shape (B, K, H, W).
        normalized_distance_threshold: Max normalized Euclidean distance for a
            keypoint prediction to be considered correct.

    Returns:
        Accuracy in [0, 1] over visible keypoints.
    """
    if pred_heatmaps.shape != target_heatmaps.shape:
        raise ValueError(
            f"pred_heatmaps and target_heatmaps must have same shape. "
            f"Got {tuple(pred_heatmaps.shape)} vs {tuple(target_heatmaps.shape)}"
        )

    bsz, num_keypoints, heat_h, heat_w = pred_heatmaps.shape
    pred_flat = pred_heatmaps.view(bsz, num_keypoints, -1)
    target_flat = target_heatmaps.view(bsz, num_keypoints, -1)

    pred_idx = pred_flat.argmax(dim=-1)
    target_idx = target_flat.argmax(dim=-1)
    visible = target_flat.max(dim=-1).values > 0.0

    pred_x = (pred_idx % heat_w).float()
    pred_y = (pred_idx // heat_w).float()
    target_x = (target_idx % heat_w).float()
    target_y = (target_idx // heat_w).float()

    dx = (pred_x - target_x) / max(heat_w - 1, 1)
    dy = (pred_y - target_y) / max(heat_h - 1, 1)
    dist = torch.sqrt(dx * dx + dy * dy)
    correct = dist <= normalized_distance_threshold

    visible_count = visible.sum().item()
    if visible_count == 0:
        return 0.0

    correct_visible = (correct & visible).sum().item()
    return correct_visible / visible_count


@torch.no_grad()
def compute_channelwise_pos_weight(loader, num_keypoints: int, max_pos_weight: float = 100.0) -> torch.Tensor:
    """Compute per-channel positive class weights for sparse heatmap targets.

    Args:
        loader: Train dataloader returning (images, target_heatmaps).
        num_keypoints: Number of heatmap channels.
        max_pos_weight: Upper clamp for numerical stability.

    Returns:
        Tensor of shape (num_keypoints,) with pos_weight = neg/pos.
    """
    pos_count = torch.zeros(num_keypoints, dtype=torch.float64)
    total_count = torch.zeros(num_keypoints, dtype=torch.float64)

    for _, target_heatmaps in loader:
        target = target_heatmaps.float()
        batch_size, channels, height, width = target.shape
        if channels != num_keypoints:
            raise ValueError(
                f"Expected {num_keypoints} keypoint channels, got {channels} while computing pos_weight."
            )

        per_channel_pos = target.sum(dim=(0, 2, 3)).to(torch.float64)
        per_channel_total = torch.full(
            (channels,),
            fill_value=batch_size * height * width,
            dtype=torch.float64,
        )

        pos_count += per_channel_pos
        total_count += per_channel_total

    neg_count = total_count - pos_count
    pos_weight = neg_count / torch.clamp(pos_count, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_pos_weight)
    return pos_weight.to(torch.float32)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Compute average loss and accuracy over one evaluation dataloader.

    Args:
        model: Keypoint model that outputs heatmaps.
        loader: Validation or test dataloader.
        criterion: Loss function used to compare predicted vs target heatmaps.
        device: torch.device where tensors and model are placed.

    Returns:
        Tuple of (mean loss, keypoint accuracy) over the loader.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    weighted_acc_sum = 0.0

    for images, target_heatmaps in loader:
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)

        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, target_heatmaps)
        batch_acc = heatmap_keypoint_accuracy(pred_heatmaps, target_heatmaps)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        weighted_acc_sum += batch_acc * bs

    mean_loss = total_loss / max(total_count, 1)
    mean_acc = weighted_acc_sum / max(total_count, 1)
    return mean_loss, mean_acc


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

    if args.image_height % 32 != 0 or args.image_width % 32 != 0:
        raise ValueError(
            "image_height and image_width must be divisible by 32 for KeypointDetectionModel. "
            f"Got image_height={args.image_height}, image_width={args.image_width}."
        )

    image_size = (args.image_height, args.image_width)
    target_heatmap_size = image_size
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.ToTensor(),
    ])
    keypoint_transform = KeypointsToHeatmaps(
        out_size=target_heatmap_size,
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
    print("Image size:", image_size, "Target heatmap size:", target_heatmap_size)

    model = KeypointDetectionModel(num_keypoints=args.num_keypoints).to(device)
    pos_weight = compute_channelwise_pos_weight(
        train_loader,
        num_keypoints=args.num_keypoints,
        max_pos_weight=args.max_pos_weight,
    ).to(device)
    print("BCE pos_weight per channel:", pos_weight.detach().cpu().numpy())
    pos_weight_broadcast = pos_weight.view(args.num_keypoints, 1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_broadcast)
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
    os.makedirs(args.export_dir, exist_ok=True)
    best_path = "checkpoints/keypoint_best.pt"
    deploy_weights_path = os.path.join(args.export_dir, "keypoint_best_state_dict.pt")
    deploy_torchscript_path = os.path.join(args.export_dir, "keypoint_best_torchscript.pt")
    best_val_loss = float("inf")
    best_cv_acc = 0.0
    start_epoch = 1

    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")

        try:
            resume_checkpoint = torch.load(args.resume_from, map_location=device)
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to resume training from the provided file. "
                "Use a training checkpoint/state_dict path such as checkpoints/keypoint_best.pt "
                "or exports/keypoint_best_state_dict.pt, not exports/keypoint_best_torchscript.pt. "
                f"Original error: {error}"
            ) from error
        if isinstance(resume_checkpoint, dict) and "model_state" in resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model_state"])

            if "optimizer_state" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])

            if "epoch" in resume_checkpoint:
                start_epoch = int(resume_checkpoint["epoch"]) + 1

            if "best_val_loss" in resume_checkpoint:
                best_val_loss = float(resume_checkpoint["best_val_loss"])

            if "best_cv_acc" in resume_checkpoint:
                best_cv_acc = float(resume_checkpoint["best_cv_acc"])
        else:
            model.load_state_dict(resume_checkpoint)

        print(f"Resumed from: {args.resume_from}")
        print(f"Resume start epoch: {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        train_acc_sum = 0.0

        for images, target_heatmaps in train_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps)
            batch_acc = heatmap_keypoint_accuracy(pred_heatmaps, target_heatmaps)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            train_acc_sum += batch_acc * bs

        train_loss = total_loss / max(total_count, 1)
        train_acc = train_acc_sum / max(total_count, 1)
        val_loss, cv_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss: {train_loss:.5f} acc: {train_acc:.5f} | "
            f"cv loss: {val_loss:.5f} acc: {cv_acc:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cv_acc = cv_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_cv_acc": best_cv_acc,
                    "args": vars(args),
                },
                best_path,
            )

    print("Best val loss:", best_val_loss)
    print("Best cv accuracy:", best_cv_acc)
    print("Saved checkpoint:", best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.5f}")

    deploy_payload = {
        "model_state": model.state_dict(),
        "num_keypoints": args.num_keypoints,
        "image_height": args.image_height,
        "image_width": args.image_width,
    }
    torch.save(deploy_payload, deploy_weights_path)

    model_cpu = KeypointDetectionModel(num_keypoints=args.num_keypoints)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    example_input = torch.randn(1, 3, args.image_height, args.image_width)
    scripted_model = torch.jit.trace(model_cpu, example_input)
    torch.jit.save(scripted_model, deploy_torchscript_path)

    print("Saved deployable weights:", deploy_weights_path)
    print("Saved deployable TorchScript:", deploy_torchscript_path)


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
        --max_pos_weight: Upper clamp for BCE positive-class weighting.
        --resume_from: Optional checkpoint/model path to continue training from.
        --export_dir: Directory to store final deployable model artifacts.
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
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--max_pos_weight", type=float, default=100.0)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--export_dir", type=str, default="exports")

    args = parser.parse_args()
    train_keypoint_model(args)


if __name__ == "__main__":
    main()