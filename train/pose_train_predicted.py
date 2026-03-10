# train/pose_train_predicted.py

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from data.pose_predicted_dataset import pose_predicted_dataset
from models.pose_classification import PoseClassificationModel


# checks if we have access to a GPU (NVIDIA or Apple Silicon) and returns the appropriate device for PyTorch computations. If no GPU is available, it falls back to CPU.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment_keypoints(keypoints: torch.Tensor, xy_noise_std: float = 0.0, joint_dropout: float = 0.0) -> torch.Tensor:
    """
    Apply light noise to already-predicted keypoints.

    Since these keypoints already came from Stage 2 predictions, they already contain real pipeline noise.
    So augmentation here should stay lighter than what we would use for clean GT keypoints.

    keypoints: (B, 18, 3) where last dim is [x, y, visibility], x/y assumed normalized to [0,1]

    xy_noise_std: std dev of Gaussian noise added to x,y (ex: 0.003)
    joint_dropout: probability of dropping a joint (ex: 0.03)
    """
    if xy_noise_std <= 0.0 and joint_dropout <= 0.0:
        return keypoints

    keypoint_clone = keypoints.clone()

    # 1) Add small Gaussian noise to x,y only
    if xy_noise_std > 0.0:
        noise = torch.randn_like(keypoint_clone[:, :, 0:2]) * xy_noise_std
        keypoint_clone[:, :, 0:2] = keypoint_clone[:, :, 0:2] + noise
        keypoint_clone[:, :, 0:2] = keypoint_clone[:, :, 0:2].clamp(0.0, 1.0)  # keep normalized range stable

    # 2) Randomly drop joints (simulate occasional failed detections)
    if joint_dropout > 0.0:
        # mask: 1 means keep, 0 means drop
        keep = (torch.rand(keypoint_clone.size(0), keypoint_clone.size(1), 1, device=keypoint_clone.device) > joint_dropout).float()
        keypoint_clone = keypoint_clone * keep  # zeros x,y,vis for dropped joints

    return keypoint_clone


@torch.no_grad()  # don't track gradients during evaluation, this makes it more memory efficient and faster since we don't need to compute gradients or keep track of computation graph.
def evaluate(model, loader, device):
    model.eval()  # disables dropout, uses deterministic behavior
    criterion = nn.CrossEntropyLoss()  # plain cross entropy for evaluation reporting

    total_loss = 0.0
    total = 0
    correct = 0

    for keypoints, label in loader:  # loop through batches, move tensor to GPU/CPU.
        keypoints = keypoints.to(device)
        label = label.to(device)

        logits = model(keypoints, return_logits=True)  # forward pass
        loss = criterion(logits, label)  # computes loss for this batch
        total_loss += loss.item() * label.size(0)
        total += label.size(0)  # accumulate total number of samples seen so far (for averaging loss and accuracy)
        correct += (logits.argmax(dim=1) == label).sum().item()  # compute number of correct predictions in this batch and accumulate

    return total_loss / max(total, 1), correct / max(total, 1)  # return average loss and accuracy over the whole dataset


@torch.no_grad()
def evaluate_pose_metrics(model, loader, device, num_classes=4):
    """
    Evaluate Stage 3 pose classification on a given subset and return:
      - accuracy
      - per-class accuracy
      - raw correct / total counts per class
    """
    model.eval()

    total = 0
    correct = 0
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    for keypoints, label in loader:
        keypoints = keypoints.to(device)
        label = label.to(device)

        logits = model(keypoints, return_logits=True)
        preds = torch.argmax(logits, dim=1)

        total += label.size(0)
        correct += (preds == label).sum().item()

        label_cpu = label.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()

        for true_label, pred_label in zip(label_cpu, preds_cpu):
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1

    accuracy = correct / max(total, 1)
    class_accuracy = np.divide(
        class_correct,
        np.maximum(class_total, 1),
        dtype=np.float64,
    )

    return {
        "num_samples": total,
        "accuracy": float(accuracy),
        "class_correct": class_correct,
        "class_total": class_total,
        "class_accuracy": class_accuracy,
    }


@torch.no_grad()
def evaluate_with_confusion_matrix(model, loader, device, save_path="evaluation_outputs/stage3_predicted_confusion_matrix.png"):
    """
    Evaluate predicted-keypoint Stage 3 model and create a confusion matrix + classification report.
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

    # hardcoded label names based on your class order
    label_names = ["Backhand", "Forehand", "Ready_Position", "Serve"]

    print("\n===== STAGE 3 PREDICTED-KEYPOINT CLASSIFICATION REPORT =====")
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))

    os.makedirs("evaluation_outputs", exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Stage 3 Confusion Matrix (Predicted Keypoints)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


def save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="evaluation_outputs"):
    """
    Saves Stage 3 training curves:
        - loss vs epoch
        - accuracy vs epoch
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # -------- loss curve --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 3 Predicted-Keypoint Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage3_predicted_training_loss.png"), dpi=200)
    plt.close()

    # -------- accuracy curve --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Stage 3 Predicted-Keypoint Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage3_predicted_training_accuracy.png"), dpi=200)
    plt.close()

    print(f"Saved training curves to: {save_dir}")

    # -------- accuracy bar graph --------
def plot_pose_accuracy_bar_chart(train_acc, test_acc, output_dir="evaluation_outputs"):
    """
    Save a Stage 3 bar chart comparing train vs test classification accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = ["Training", "Test"]
    values = [train_acc, test_acc]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Pose Classification Model Performance")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.4f}",
            ha="center"
        )

    save_path = os.path.join(output_dir, "pose_accuracy_bar_chart.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

    print(f"Saved pose accuracy bar chart to: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    # path to cached Stage 2 predicted keypoints (hyperparameters set for lower training accuracy to match test accuracy, don't want overfitting? need better keypoint input.)
    parser.add_argument("--feature_path", type=str, required=True)

    # Hyperparameters and settings
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-2)

    # dataset and splits
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--visibility_threshold", type=float, default=0.15)

    # data augmentation hyperparameters (for training only, val/test remain clean)
    # predicted keypoints already contain real Stage 2 noise, so keep these light
    parser.add_argument("--xy_noise_std", type=float, default=0.006)
    parser.add_argument("--joint_dropout", type=float, default=0.06)

    # optimization stability improvements
    parser.add_argument("--label_smoothing", type=float, default=0.05)  # softens targets slightly so the model does not become overconfident too early
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)    # clips gradient norm to prevent exploding updates
    parser.add_argument("--early_stop_patience", type=int, default=12)  # stop if validation accuracy does not improve for many epochs

    args = parser.parse_args()

    # reproducibility, seeds everything.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # catches mistakes in split ratios, ensures they sum to 1.
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total}")

    # load cached predicted keypoints built from Stage 2 outputs
    dataset = pose_predicted_dataset(args.feature_path)

    # sanity check, print dataset size.
    print("Total Samples:", len(dataset))

    # create train/val/test splits, use random_split with a fixed seed for reproducibility. This will give us three separate datasets for training, validation, and testing.
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_split)
    val_size = int(dataset_size * args.val_split)
    test_size = dataset_size - train_size - val_size

    # random_split returns subsets of the original dataset, we can then create DataLoaders for each split to handle batching and shuffling during training and evaluation.
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # train shuffle = true for better learning
    # val/test shuffle = False (deterministic order for consistent evaluation)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # separate eval loaders so train accuracy is measured cleanly without shuffle / augmentation
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_eval_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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

    # optimiser and loss
    # AdamW handles weight decay better than plain Adam for regularized training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # label smoothing slightly softens class targets to improve stability/generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Reduce learning rate when validation accuracy plateaus so training settles more smoothly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    # checkpoint setup, stores best model here, track best validation accuracy.
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/pose_best_predicted.pt"
    best_val_acc = -1.0
    epochs_without_improvement = 0

    # stores per-epoch metrics so we can plot training curves at the end
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()  # enables dropout, training mode

        # accumulates training stats.
        total_loss = 0.0
        total_count = 0
        correct = 0

        # loads batch and move to device.
        for keypoints, label in train_loader:
            keypoints = keypoints.to(device)
            label = label.to(device)

            # training sees lightly augmented predicted inputs (more realistic) val/test remain clean.
            keypoints = augment_keypoints(
                keypoints,
                xy_noise_std=args.xy_noise_std,
                joint_dropout=args.joint_dropout,
            )

            # clears old gradients, forward pass, compute loss, backward pass, and update weights.
            optimizer.zero_grad(set_to_none=True)
            logits = model(keypoints, return_logits=True)  # forward pass
            loss = criterion(logits, label)  # calc. loss
            loss.backward()  # backward pass, computes gradients

            # clip gradient norm so very large updates do not destabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()  # updates weights based on gradients

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

        # save metrics for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # scheduler uses validation accuracy to decide when to reduce learning rate
        scheduler.step(val_acc)

        # print stats for this epoch.
        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc {val_acc:.4f}"
        )

        # if validation accuracy improved: update best, save checkpoint, reset patience counter.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
        else:
            # count how many epochs in a row failed to improve validation accuracy
            epochs_without_improvement += 1

        # stop training if validation accuracy has not improved for a while
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # final results
    print("Best val acc:", best_val_acc)
    print("Saved checkpoints:", best_path)

    # test best checkpoint on test set
    checkpoint = torch.load(best_path, map_location=device)  # load best model checkpoint back into model, ensures you test the best one.
    model.load_state_dict(checkpoint["model_state"])

    # evaluates on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} acc {test_acc:.4f}")

    # evaluate train/test metrics for the final bar chart
    train_metrics = evaluate_pose_metrics(
        model=model,
        loader=train_eval_loader,
        device=device,
        num_classes=4,
    )
    test_metrics = evaluate_pose_metrics(
        model=model,
        loader=test_eval_loader,
        device=device,
        num_classes=4,
    )

    print("\nTRAINING EVALUATION RESULTS")
    print(f"Num samples: {train_metrics['num_samples']}")
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")

    print("\nTRAIN per-class accuracy")
    label_names = ["Backhand", "Forehand", "Ready_Position", "Serve"]
    for i, name in enumerate(label_names):
        print(
            f"{name}: {train_metrics['class_accuracy'][i]:.4f} "
            f"({train_metrics['class_correct'][i]}/{train_metrics['class_total'][i]})"
        )

    print("\nTEST EVALUATION RESULTS")
    print(f"Num samples: {test_metrics['num_samples']}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")

    print("\nTEST per-class accuracy")
    for i, name in enumerate(label_names):
        print(
            f"{name}: {test_metrics['class_accuracy'][i]:.4f} "
            f"({test_metrics['class_correct'][i]}/{test_metrics['class_total'][i]})"
        )

    # builds confusion matrix on the predicted-keypoint test set
    evaluate_with_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        save_path="evaluation_outputs/stage3_predicted_confusion_matrix.png",
    )

    # save training/validation curves
    save_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        save_dir="evaluation_outputs",
    )

    # save final train vs test accuracy bar chart
    plot_pose_accuracy_bar_chart(
        train_acc=train_metrics["accuracy"],
        test_acc=test_metrics["accuracy"],
        output_dir="evaluation_outputs",
    )


if __name__ == "__main__":
    main()