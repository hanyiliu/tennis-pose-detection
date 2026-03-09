# inference/full_pipeline_eval.py
# runs the full 3-stage pipeline on a held-out test split,
# computes final classification metrics,
# saves confusion matrix, per-class f1 graph, predictions csv, and metrics text file.

import os
import sys
import json
import csv
from typing import List, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torchvision import transforms

# makes sure project root is on python path so imports work when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import kagglehub
from models.bbox_detection import BBoxDetectionModel
from models.keypoint_detection import KeypointDetectionModel
from models.pose_classification import PoseClassificationModel
from preprocessing.pil_preprocessing import letterbox_resize
from preprocessing.tensor_preprocessing import heatmaps_to_keypoints, normalize_keypoints_xy


# helper function to pick best available device.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# normalizes class names so label mapping is more consistent.
# this helps if one file says "Ready Position" and another says "ready_position".
def normalize_label_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


# downloads dataset (or reuses cached version) and handles kagglehub nested folder layout.
def resolve_dataset_root(root_dir: Optional[str] = None) -> str:
    if root_dir is None or root_dir == "":
        root_dir = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")

        # kagglehub sometimes wraps files in one extra folder
        items = os.listdir(root_dir)
        if len(items) == 1:
            maybe_folder = os.path.join(root_dir, items[0])
            if os.path.isdir(maybe_folder):
                root_dir = maybe_folder

    return root_dir


# loads all image paths + class labels from the coco annotation files.
# this is what we use to create a held-out test split for the full pipeline.
def load_pipeline_samples(root_dir: str, annotation_files: List[str], class_order: Optional[List[str]] = None):
    samples: List[Dict] = []

    # collect category names across all annotation files
    cat_id_to_name = {}
    json_datas = []

    for rel_path in annotation_files:
        full_path = os.path.join(root_dir, rel_path)
        with open(full_path, "r") as f:
            data = json.load(f)

        json_datas.append(data)

        for c in data.get("categories", []):
            cat_id_to_name[c["id"]] = normalize_label_name(c.get("name", str(c["id"])))

    if not cat_id_to_name:
        raise RuntimeError("No categories found in annotation files.")

    # build label order
    if class_order is not None:
        normalized_order = [normalize_label_name(name) for name in class_order]
        label_names = normalized_order
    else:
        ordered_ids = sorted(cat_id_to_name.keys())
        label_names = [cat_id_to_name[cid] for cid in ordered_ids]

    label_to_index = {name: i for i, name in enumerate(label_names)}

    # build actual sample list from annotations
    for data in json_datas:
        id_to_img = {img["id"]: img for img in data.get("images", [])}

        # build category id -> normalized category name for this file
        local_cat_map = {
            c["id"]: normalize_label_name(c.get("name", str(c["id"])))
            for c in data.get("categories", [])
        }

        for ann in data.get("annotations", []):
            image_id = ann.get("image_id", None)
            category_id = ann.get("category_id", None)

            if image_id not in id_to_img or category_id not in local_cat_map:
                continue

            label_name = local_cat_map[category_id]
            if label_name not in label_to_index:
                continue

            img_info = id_to_img[image_id]
            rel_path = img_info.get("path", img_info["file_name"])
            rel_path = rel_path.replace("../", "")
            img_path = os.path.join(root_dir, rel_path)

            samples.append(
                {
                    "img_path": img_path,
                    "label_name": label_name,
                    "label_index": label_to_index[label_name],
                }
            )

    if len(samples) == 0:
        raise RuntimeError("Loaded 0 samples for full pipeline evaluation.")

    return samples, label_names


# loads bbox model and weights.
def load_bbox_model(checkpoint_path: str, device):
    model = BBoxDetectionModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# loads keypoint model.
# this uses the deployable state_dict file saved in exports from keypoint_train.py
def load_keypoint_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_keypoints = checkpoint.get("num_keypoints", 18)
    image_height = checkpoint.get("image_height", 128)
    image_width = checkpoint.get("image_width", 128)

    model = KeypointDetectionModel(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, (image_height, image_width)


# loads pose classifier model and weights.
def load_pose_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 256)
    dropout = args.get("dropout", 0.25)
    visibility_threshold = args.get("visibility_threshold", 0.0)

    model = PoseClassificationModel(
        num_keypoints=18,
        num_classes=4,
        hidden_dim=hidden_dim,
        dropout=dropout,
        visibility_threshold=visibility_threshold,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    label_names = checkpoint.get("label_names", None)
    if isinstance(label_names, list):
        label_names = [normalize_label_name(name) for name in label_names]

    return model, label_names


# runs the full 3-stage pipeline on one image and returns the final predicted class index.
@torch.no_grad()
def run_full_pipeline_on_image(
    pil_image: Image.Image,
    bbox_model,
    keypoint_model,
    pose_model,
    device,
    bbox_input_size=(256, 256),
    keypoint_input_size=(128, 128),
):
    # -------- stage 1: bounding box prediction --------

    bbox_transform = transforms.Compose([
        transforms.Resize(bbox_input_size),
        transforms.ToTensor(),
    ])

    bbox_tensor = bbox_transform(pil_image).unsqueeze(0).to(device)
    pred_bbox = bbox_model(bbox_tensor)[0].detach().cpu()

    # bbox model predicts normalized [x, y, w, h], so convert back to original pixel coordinates
    original_w, original_h = pil_image.size
    x = float(pred_bbox[0].item() * original_w)
    y = float(pred_bbox[1].item() * original_h)
    w = float(pred_bbox[2].item() * original_w)
    h = float(pred_bbox[3].item() * original_h)

    # convert [x, y, w, h] into crop corners [left, top, right, bottom]
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(original_w, int(round(x + w)))
    bottom = min(original_h, int(round(y + h)))

    # if model predicts weird bbox, make sure crop is still valid
    if right <= left:
        right = min(original_w, left + 1)
    if bottom <= top:
        bottom = min(original_h, top + 1)

    cropped_img = pil_image.crop((left, top, right, bottom))

    # -------- stage 2: keypoint prediction --------

    # use same letterbox resize idea as training pipeline
    resized_img = letterbox_resize(cropped_img, size=keypoint_input_size)

    keypoint_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    keypoint_tensor = keypoint_transform(resized_img).unsqueeze(0).to(device)

    heatmaps = keypoint_model(keypoint_tensor)

    # convert heatmaps to [x, y, visibility] keypoint matrix
    keypoints = heatmaps_to_keypoints(heatmaps)

    # normalize keypoints to [0,1] range to match pose model training
    heat_h = heatmaps.size(2)
    heat_w = heatmaps.size(3)
    keypoints = normalize_keypoints_xy(keypoints, heat_h, heat_w)

    # -------- stage 3: pose classification --------

    logits = pose_model(keypoints, return_logits=True)
    pred_index = int(torch.argmax(logits, dim=1)[0].item())

    return pred_index


def save_confusion_matrix(cm, class_names, save_path):
    # creates a confusion matrix graph with counts written in each square.
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Full Pipeline Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks, class_names)

    # writes the value inside each cell
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_per_class_f1_bar_chart(class_names, report_dict, save_path):
    # collects f1 score for each class in the correct class order.
    per_class_f1 = [report_dict[name]["f1-score"] for name in class_names]

    plt.figure(figsize=(7, 5))
    plt.bar(class_names, per_class_f1)

    plt.title("Full Pipeline Per-Class F1 Score")
    plt.xlabel("Pose Class")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25)

    # writes actual f1 score value above each bar
    for i, score in enumerate(per_class_f1):
        plt.text(i, score + 0.02, f"{score:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # dataset / split settings
    root_dir = None
    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    seed = 42

    # checkpoint paths
    bbox_checkpoint_path = "checkpoints/bbox_best.pt"
    keypoint_checkpoint_path = "exports/keypoint_best_state_dict.pt"
    pose_checkpoint_path = "checkpoints/pose_best.pt"

    # output folder
    output_dir = "evaluation_outputs"

    device = get_device()
    print("Device:", device)

    # load trained models
    bbox_model = load_bbox_model(bbox_checkpoint_path, device)
    keypoint_model, keypoint_input_size = load_keypoint_model(keypoint_checkpoint_path, device)
    pose_model, pose_label_names = load_pose_model(pose_checkpoint_path, device)

    # load all labeled image samples
    root_dir = resolve_dataset_root(root_dir)
    samples, dataset_label_names = load_pipeline_samples(
        root_dir=root_dir,
        annotation_files=annotation_files,
        class_order=pose_label_names,
    )

    # use pose checkpoint label order if available, otherwise dataset label order
    class_names = pose_label_names if pose_label_names is not None else dataset_label_names

    print("Dataset root:", root_dir)
    print("Total samples:", len(samples))
    print("Class names:", class_names)

    # stratified split so class balance stays consistent
    X = [sample["img_path"] for sample in samples]
    y = [sample["label_index"] for sample in samples]

    train_X, temp_X, train_y, temp_y = train_test_split(
        X,
        y,
        test_size=(val_split + test_split),
        stratify=y,
        random_state=seed,
    )

    val_ratio = val_split / (val_split + test_split)
    val_X, test_X, val_y, test_y = train_test_split(
        temp_X,
        temp_y,
        test_size=(1 - val_ratio),
        stratify=temp_y,
        random_state=seed,
    )

    print("Train size:", len(train_X))
    print("Val size:", len(val_X))
    print("Test size:", len(test_X))

    # build lookup from image path -> true label index for test set
    path_to_label = {sample["img_path"]: sample["label_index"] for sample in samples}

    # run end-to-end pipeline on each test image
    y_true = []
    y_pred = []
    prediction_rows = []

    for idx, img_path in enumerate(test_X, start=1):
        pil_image = Image.open(img_path).convert("RGB")

        true_label = path_to_label[img_path]
        pred_label = run_full_pipeline_on_image(
            pil_image=pil_image,
            bbox_model=bbox_model,
            keypoint_model=keypoint_model,
            pose_model=pose_model,
            device=device,
            bbox_input_size=(256, 256),
            keypoint_input_size=keypoint_input_size,
        )

        y_true.append(true_label)
        y_pred.append(pred_label)

        prediction_rows.append(
            {
                "img_path": img_path,
                "true_label_index": true_label,
                "true_label_name": class_names[true_label],
                "pred_label_index": pred_label,
                "pred_label_name": class_names[pred_label],
            }
        )

        if idx % 25 == 0 or idx == len(test_X):
            print(f"Processed {idx}/{len(test_X)} test images")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # compute final end-to-end classification metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== FULL PIPELINE TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report_text)

    # save graphs and files
    os.makedirs(output_dir, exist_ok=True)

    confusion_matrix_path = os.path.join(output_dir, "full_pipeline_confusion_matrix.png")
    per_class_f1_path = os.path.join(output_dir, "full_pipeline_per_class_f1_scores.png")
    predictions_csv_path = os.path.join(output_dir, "full_pipeline_predictions.csv")
    metrics_txt_path = os.path.join(output_dir, "full_pipeline_metrics.txt")

    save_confusion_matrix(cm, class_names, confusion_matrix_path)
    save_per_class_f1_bar_chart(class_names, report_dict, per_class_f1_path)

    # save predictions csv
    with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "img_path",
                "true_label_index",
                "true_label_name",
                "pred_label_index",
                "pred_label_name",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    # save metrics summary txt
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("FULL PIPELINE TEST RESULTS\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)

    print("\nSaved:")
    print("-", confusion_matrix_path)
    print("-", per_class_f1_path)
    print("-", predictions_csv_path)
    print("-", metrics_txt_path)


if __name__ == "__main__":
    main()