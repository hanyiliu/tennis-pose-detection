# pose_infer.py

import os
import json
import torch
import kagglehub
import random

from models.pose_classification import PoseClassificationModel
from preprocessing.tensor_preprocessing import normalize_keypoints_xy


# -----------------------------------
# Device
# -----------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------
# Load keypoints from Kaggle JSON
# -----------------------------------

def load_keypoints(json_path: str, ann_index: int = 0):
    """
    Loads one sample's keypoints from COCO annotation file.
    Returns:
        keypoints tensor (18,3) in pixel coords
        width, height of image
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    images = data.get("images", [])

    if len(anns) == 0:
        raise RuntimeError(f"No annotations found in {json_path}")

    ann_index = max(0, min(ann_index, len(anns) - 1))
    ann = anns[ann_index]

    keypoints = ann.get("keypoints", None)
    if not isinstance(keypoints, list) or len(keypoints) != 54:
        raise RuntimeError("Annotation does not have 54 keypoint values")

    # get width + height from image metadata
    image_id = ann["image_id"]
    img_info = next(img for img in images if img["id"] == image_id)
    width = img_info["width"]
    height = img_info["height"]

    keypoints_t = torch.tensor(keypoints, dtype=torch.float32).reshape(18, 3)

    return keypoints_t, height, width


# -----------------------------------
# Pose Inference
# -----------------------------------

def infer_pose(
    keypoints: torch.Tensor,
    H: int,
    W: int,
    checkpoint_path: str = "checkpoints/pose_best.pt",
):

    device = get_device()

    # ---- Normalize to match training ----
    keypoints = normalize_keypoints_xy(keypoints, H, W)

    # add batch dimension if needed
    if keypoints.dim() == 2:
        keypoints = keypoints.unsqueeze(0)  # (1,18,3)

    keypoints = keypoints.to(device)

    # ---- Load checkpoint ----
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

    with torch.no_grad():
        probs = model(keypoints)

    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    pred_index = int(torch.argmax(probs, dim=1)[0].item())

    label_names = checkpoint.get("label_names", None)
    pred_name = (
        label_names[pred_index]
        if isinstance(label_names, list) and pred_index < len(label_names)
        else None
    )

    print("\nCheckpoint:", checkpoint_path)

    if pred_name:
        print("Predicted pose:", pred_index, f"({pred_name})")
    else:
        print("Predicted pose:", pred_index)

    print("\nProbabilities:")
    for i in range(probs.size(1)):
        name = (
            label_names[i]
            if isinstance(label_names, list) and i < len(label_names)
            else f"class_{i}"
        )
        print(f"  {i} ({name}): {probs[0,i].item():.4f}")

    return pred_index, probs.cpu()


# -----------------------------------
# Run Example From Kaggle Dataset
# -----------------------------------

if __name__ == "__main__":
    import random
    import kagglehub

    # download dataset automatically (cached after first run)
    root_dir = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")

    # handle nested folder
    subdirs = os.listdir(root_dir)
    if len(subdirs) == 1:
        possible_root = os.path.join(root_dir, subdirs[0])
        if os.path.isdir(possible_root):
            root_dir = possible_root

    # pick random class file
    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]
    chosen_rel = random.choice(annotation_files)
    json_path = os.path.join(root_dir, chosen_rel)

    # open JSON once to choose random annotation index
    with open(json_path, "r") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if len(anns) == 0:
        raise RuntimeError(f"No annotations found in {json_path}")

    rand_index = random.randint(0, len(anns) - 1)

    print("Random file:", chosen_rel)
    print("Random annotation index:", rand_index)

    # load keypoints + image size
    keypoints, H, W = load_keypoints(json_path, ann_index=rand_index)

    # run pose inference (your infer_pose already normalizes)
    infer_pose(keypoints, H, W)