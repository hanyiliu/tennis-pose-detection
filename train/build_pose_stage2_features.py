# train/build_pose_stage2_features.py

import os
import json
import argparse
from typing import List, Dict
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.keypoint_detection import KeypointDetectionModel
from models.keypoint_attention_detection import KeypointAttentionDetectionModel
from preprocessing.image_preprocessing import convert_image_to_tensor
from preprocessing.pil_preprocessing import crop_pil, letterbox_resize
from preprocessing.tensor_preprocessing import heatmaps_to_keypoints, normalize_keypoints_xy


# checks if we have access to GPU (NVIDIA / Apple Silicon) and returns the best device for PyTorch
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TennisStage2FeatureBuilderDataset(Dataset):
    """
    This dataset is ONLY for building Stage 3 training features from Stage 2 predictions.

    It uses:
        image + GT bbox + GT label

    So that we can:
        crop player correctly with GT bbox,
        run Stage 2 keypoint detection,
        convert heatmaps -> keypoints,
        and save predicted keypoints for Stage 3 training.

    Returns:
        tensor: cropped/resized player image tensor for Stage 2
        label: pose label for Stage 3
    """

    def __init__(
        self,
        root_dir: str,
        annotation_files: List[str],
        class_order=None,
        keypoint_image_size=(256, 256),
    ):
        self.root_dir = root_dir
        self.annotation_files = annotation_files
        self.keypoint_image_size = keypoint_image_size
        self.samples: List[Dict] = []

        # collect all category ids -> names across json files
        cat_id_to_name = {}
        json_datas = []

        for rel_path in annotation_files:
            full_path = os.path.join(root_dir, rel_path)
            with open(full_path, "r") as f:
                data = json.load(f)
            json_datas.append(data)

            for c in data.get("categories", []):
                cat_id_to_name[c["id"]] = c.get("name", str(c["id"]))

        if not cat_id_to_name:
            raise RuntimeError("No categories found in annotation files.")

        # build category_id -> label index mapping
        if class_order is not None:
            # map category names to ids so we can enforce a consistent label order
            name_to_id = {name: cid for cid, name in cat_id_to_name.items()}
            missing = [n for n in class_order if n not in name_to_id]
            if missing:
                raise ValueError(f"class_order name not found in categories: {missing}")

            ordered_ids = [name_to_id[n] for n in class_order]
            self.label_names = class_order[:]
        else:
            # default: sort by category id if user did not provide explicit class order
            ordered_ids = sorted(cat_id_to_name.keys())
            self.label_names = [cat_id_to_name[cid] for cid in ordered_ids]

        self.label_map = {cid: i for i, cid in enumerate(ordered_ids)}

        # build sample list
        for data in json_datas:
            id_to_img = {img["id"]: img for img in data.get("images", [])}

            for ann in data.get("annotations", []):
                # load bbox, category id, and image id from annotation
                bbox = ann.get("bbox", None)
                cat_id = ann.get("category_id", None)
                image_id = ann.get("image_id", None)

                # skip annotation if bbox missing or category id is not used
                if bbox is None or cat_id not in self.label_map:
                    continue

                img_info = id_to_img.get(image_id, {})
                rel_path = img_info.get("path", img_info.get("file_name", None))
                if rel_path is None:
                    continue

                # remove ../ from kaggle json paths so joining with root_dir works correctly
                rel_path = rel_path.replace("../", "")
                img_path = os.path.join(root_dir, rel_path)

                self.samples.append(
                    {
                        "img_path": img_path,
                        "bbox": bbox,  # COCO format [x, y, w, h]
                        "label": self.label_map[cat_id],
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError("Loaded 0 samples while building Stage 2 features.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # load image and force RGB
        img = Image.open(sample["img_path"]).convert("RGB")

        # COCO bbox format = [x_min, y_min, width, height]
        x, y, w, h = sample["bbox"]

        # convert bbox to crop_pil format = [x_min, y_min, x_max, y_max]
        bbox_xyxy = [x, y, x + w, y + h]

        # crop player using GT bbox
        cropped_img = crop_pil(img, bbox_xyxy)

        # resize cropped player to match Stage 2 input size
        resized_img = letterbox_resize(cropped_img, size=self.keypoint_image_size)

        # convert PIL image -> tensor for Stage 2 model
        tensor = convert_image_to_tensor(resized_img)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return tensor, label


@torch.no_grad()
def build_predicted_keypoint_features(
    stage2_model,
    loader,
    device,
    save_path: str,
    label_names=None,
):
    """
    Runs Stage 2 model over the whole dataset and saves predicted keypoints + labels.

    Output file contains:
        keypoints: (N, 18, 3)
        labels: (N,)
        label_names: class names if available
    """
    stage2_model.eval()

    all_keypoints = []
    all_labels = []

    total = len(loader.dataset)
    seen = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Stage 2 predicts heatmaps of shape (B, K, H, W)
        heatmaps = stage2_model(imgs)

        # convert heatmaps -> keypoints (B, K, 3)
        keypoints = heatmaps_to_keypoints(heatmaps)

        # normalize x,y to [0,1] so Stage 3 sees the same scale as during training
        H = heatmaps.size(2)
        W = heatmaps.size(3)
        keypoints = normalize_keypoints_xy(keypoints, H=H, W=W)

        all_keypoints.append(keypoints.cpu())
        all_labels.append(labels.cpu())

        seen += labels.size(0)
        print(f"Processed {seen}/{total} samples")

    all_keypoints = torch.cat(all_keypoints, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "keypoints": all_keypoints,
            "labels": all_labels,
            "label_names": label_names,
            "num_samples": all_keypoints.size(0),
        },
        save_path,
    )

    print("\nSaved predicted Stage 2 keypoint features to:", save_path)
    print("Keypoints shape:", tuple(all_keypoints.shape))
    print("Labels shape:", tuple(all_labels.shape))


def resolve_stage2_model_type(stage2_checkpoint: str, checkpoint_payload, requested_model_type: str) -> str:
    if requested_model_type in {"standard", "attention"}:
        return requested_model_type

    model_type_hint = ""
    if isinstance(checkpoint_payload, dict):
        model_type_hint = str(checkpoint_payload.get("model_type", "")).lower()

    checkpoint_name = os.path.basename(stage2_checkpoint).lower()
    if "attention" in checkpoint_name or "attention" in model_type_hint:
        return "attention"

    return "standard"


def resolve_save_path(save_path: str, stage2_model_type: str) -> str:
    if save_path:
        return save_path
    if stage2_model_type == "attention":
        return "saved_models/stage3_predicted_attention_keypoints.pt"
    return "saved_models/stage3_predicted_keypoints.pt"


def resolve_dataset_root(root_dir_arg: str = None) -> str:
    if root_dir_arg:
        if not os.path.isdir(root_dir_arg):
            raise FileNotFoundError(f"Provided --root_dir does not exist: {root_dir_arg}")
        return root_dir_arg

    dataset_root_candidates = [
        os.path.join(
            PROJECT_ROOT,
            "datasets",
            "orvile",
            "tennis-player-actions-dataset",
            "versions",
            "1",
            "Tennis Player Actions Dataset for Human Pose Estimation",
        ),
        os.path.join(PROJECT_ROOT, "datasets", "walnut"),
    ]

    for candidate in dataset_root_candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find dataset root under repository /datasets directory. "
        "Pass --root_dir explicitly if your dataset is elsewhere."
    )


def main():
    parser = argparse.ArgumentParser()

    # checkpoint path for your trained Stage 2 model
    parser.add_argument("--stage2_checkpoint", type=str, required=True)

    # where to save cached predicted keypoints for Stage 3
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument(
        "--stage2_model_type",
        type=str,
        choices=["auto", "standard", "attention"],
        default="auto",
        help="Stage-2 architecture type. Use 'auto' to infer from checkpoint name/metadata.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Dataset root. Defaults to local repository /datasets candidates.",
    )

    # batch size for feature building
    parser.add_argument("--batch_size", type=int, default=32)

    # optional class order for consistent mapping
    parser.add_argument("--class_order", nargs="*", default=None)

    args = parser.parse_args()

    root_dir = resolve_dataset_root(args.root_dir)

    print("Dataset root:", root_dir)

    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]

    dataset = TennisStage2FeatureBuilderDataset(
        root_dir=root_dir,
        annotation_files=annotation_files,
        class_order=args.class_order,
        keypoint_image_size=(256, 256),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print("Device:", device)

    # load checkpoint file from disk
    checkpoint = torch.load(args.stage2_checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint_state = checkpoint["model_state"]
    else:
        checkpoint_state = checkpoint

    if not isinstance(checkpoint_state, dict):
        raise ValueError("Unexpected checkpoint format. Expected state_dict or dict containing model_state.")

    stage2_model_type = resolve_stage2_model_type(args.stage2_checkpoint, checkpoint, args.stage2_model_type)

    # load trained Stage 2 model checkpoint
    if stage2_model_type == "attention":
        stage2_model = KeypointAttentionDetectionModel().to(device)
    else:
        stage2_model = KeypointDetectionModel().to(device)

    # some files store raw state_dict directly, others wrap it in a dict under "model_state"
    stage2_model.load_state_dict(checkpoint_state)

    stage2_model.eval()

    save_path = resolve_save_path(args.save_path, stage2_model_type)
    print("Stage-2 model type:", stage2_model_type)
    print("Output feature path:", save_path)

    build_predicted_keypoint_features(
        stage2_model=stage2_model,
        loader=loader,
        device=device,
        save_path=save_path,
        label_names=getattr(dataset, "label_names", None),
    )


if __name__ == "__main__":
    main()