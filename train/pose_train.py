# train/pose_train.py

from html import parser
import os
import argparse 
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.pose_dataset import tennis_pose_dataset 
from models.pose_classification import PoseClassificationModel

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# do we need this???
def get_device(): # checks if we have access to a GPU (NVIDIA or Apple Silicon) and returns the appropriate device for PyTorch computations. If no GPU is available, it falls back to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def augment_keypoints(keypoints: torch.Tensor, xy_noise_std: float = 0.0, joint_dropout: float = 0.0) -> torch.Tensor:
    """
    Apply Stage-2-like noise to GT keypoints.
    keypoints: (B, 18, 3) where last dim is [x, y, visibility], x/y assumed normalized to [0,1]

    xy_noise_std: std dev of Gaussian noise added to x,y (ex: 0.01)
    joint_dropout: probability of dropping a joint (ex: 0.10)
    """
    if xy_noise_std <= 0.0 and joint_dropout <= 0.0:
        return keypoints

    keypoint_clone = keypoints.clone()

    # 1) Add Gaussian noise to x,y only
    if xy_noise_std > 0.0:
        noise = torch.randn_like(keypoint_clone[:, :, 0:2]) * xy_noise_std
        keypoint_clone[:, :, 0:2] = keypoint_clone[:, :, 0:2] + noise
        keypoint_clone[:, :, 0:2] = keypoint_clone[:, :, 0:2].clamp(0.0, 1.0)  # keep normalized range stable

    # 2) Randomly drop joints (simulate missing / low-confidence joints from Stage 2)
    if joint_dropout > 0.0:
        # mask: 1 means keep, 0 means drop
        keep = (torch.rand(keypoint_clone.size(0), keypoint_clone.size(1), 1, device=keypoint_clone.device) > joint_dropout).float()
        keypoint_clone = keypoint_clone * keep  # zeros x,y,vis for dropped joints

    return keypoint_clone

@torch.no_grad() # don't track gradients during evaluation, this makes it more memory efficient and faster since we don't need to compute gradients or keep track of computation graph.
def evaluate(model, loader, device):
    model.eval() # disables dropout, uses deterministic behavior
    criterion = nn.CrossEntropyLoss() # cross entropy loss for multi-class classification.

    total_loss = 0.0
    total = 0
    correct = 0

    for keypoints, label in loader: # loop through batches, move tensor to GPU/CPU.
        keypoints = keypoints.to(device)
        label = label.to(device)

        logits = model(keypoints, return_logits=True) # forward pass
        loss = criterion(logits, label) # computes loss for this batch
        total_loss += loss.item() * label.size(0)
        total += label.size(0) # accumulate total number of samples seen so far (for averaging loss and accuracy)
        correct += (logits.argmax(dim=1) == label).sum().item() # compute number of correct predictions in this batch and accumulate

    return total_loss / max(total, 1), correct / max(total, 1) # return average loss and accuracy over the whole dataset

@torch.no_grad()
def evaluate_with_confusion_matrix(model, loader, device, label_names=None, save_path="evaluation_outputs/stage3_confusion_matrix.png"):
    """
    Evaluate Stage 3 on the given loader and create a confusion matrix.

    Args:
        model: trained pose classification model
        loader: DataLoader for evaluation
        device: cuda / mps / cpu
        label_names: optional list of class names
        save_path: where to save confusion matrix image
    """
    model.eval()

    all_preds = []
    all_labels = []

    for keypoints, label in loader:
        keypoints = keypoints.to(device)
        label = label.to(device)

        logits = model(keypoints, return_logits=True)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== STAGE 3 TEST CLASSIFICATION REPORT =====")
    if label_names is not None:
        print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))
    else:
        print(classification_report(all_labels, all_preds, digits=4))

    os.makedirs("evaluation_outputs", exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Stage 3 Confusion Matrix (Ground Truth Keypoints)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    # Hyperparameters and settings
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # dataset and splits
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--visibility_threshold", type=float, default=0.0)

    # optional class order for consistent label mapping across different annotation files (if not provided, will infer from category ids in the JSON files)
    parser.add_argument("--class_order", nargs="*", default=None)

    # data augmentation hyperparameters (for training only, val/test remain clean)
    # adding noise for more realistic data
    parser.add_argument("--xy_noise_std", type=float, default=0.0)
    parser.add_argument("--joint_dropout", type=float, default=0.0)

    # dataset access (if root_dir is not provided, will download/reuse cached dataset the same way bbox_train does)
    args = parser.parse_args()
    
    # reproducibility, seeds everything.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # catches mistakes in split ratios, ensures they sum to 1.
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total}")
    
    # relative paths for data.
    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]

    # dataset access
    dataset = tennis_pose_dataset(root_dir=None, annotation_files=annotation_files, class_order=args.class_order)

    # sanity check, print dataset size and label order if available.
    print("Total Samples:", len(dataset))
    if hasattr(dataset, "label_names"):
        print("Label order (0...3):", dataset.label_names)

    # create train/val/test splits, use random_split with a fixed seed for reproducibility. This will give us three separate datasets for training, validation, and testing.
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_split)
    val_size = int(dataset_size * args.val_split)
    test_size = dataset_size - train_size - val_size

    # random_split returns subsets of the original dataset, we can then create DataLoaders for each split to handle batching and shuffling during training and evaluation.
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(args.seed),)

    # train shuffle = true for better learning
    # val/test shuffle = False (deterministic order for consistent evaluation)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # device info.
    device = get_device()
    print("Device:", device)
    print("Train/Val/Test sizes:", len(train_ds), len(val_ds), len(test_ds))

    # creates MLP classifier and moves it to GPU/CPU device.
    model = PoseClassificationModel(
        num_keypoints=18,
        num_classes=4,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        visibility_threshold=args.visibility_threshold,
    ).to(device)

    # optimiser and loss, 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # checkpoint setup, stores best model here, track best validation accuracy.
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/pose_best.pt"
    best_val_acc = -1.0

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train() # enables dropout, training mode

        # accumulates training stats.
        total_loss = 0.0
        total_count = 0
        correct = 0

        # loads batch and move to device.
        for keypoints, label in train_loader:
            keypoints = keypoints.to(device)
            label = label.to(device)

            # training sees noisy inputs (more realistic) val/test remain clean.
            keypoints = augment_keypoints(
                keypoints,
                xy_noise_std=args.xy_noise_std,
                joint_dropout=args.joint_dropout,
            )

            # clears old gradients, forward pass, compute loss, backward pass, and update weights.
            optimizer.zero_grad(set_to_none=True)
            logits = model(keypoints, return_logits=True) # forward pass
            loss = criterion(logits, label) # calc. loss
            loss.backward() # backward pass, computes gradients
            optimizer.step() # updates weights based on gradients

            # computes average loss and running accuracy for training.
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            correct += (logits.argmax(dim=1) == label).sum().item()
        
        # epoch metrics
        train_loss = total_loss / max(total_count, 1)
        train_acc = correct / max(total_count, 1)

        # runs evaluation.
        val_loss, val_acc = evaluate(model, val_loader, device)

        # print stats for this epoch.
        print(f"Epoch {epoch:03d} | " f"train loss: {train_loss:.4f} acc {train_acc:.4f} | " f"val loss: {val_loss:.4f} acc {val_acc:.4f}")
        
        # if validation accuracy improved: update best, save checkpoint.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label_names": getattr(dataset, "label_names", None),
                "args": vars(args),
            }, best_path,)

    # final results
    print("Best val acc:", best_val_acc)
    print("Saved checkpoints:", best_path)

    # test best checkpoint on test set
    checkpoint = torch.load(best_path, map_location=device) # load best model checkpoint back into model, ensures you test the best one.
    model.load_state_dict(checkpoint["model_state"])

    # evaluates on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} acc {test_acc:.4f}")

    label_names = checkpoint.get("label_names", None)
    if label_names is not None:
        print("Labels:", label_names)

    evaluate_with_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        label_names=label_names,
        save_path="evaluation_outputs/stage3_confusion_matrix.png",
    )
        
if __name__ == "__main__":
    main()



