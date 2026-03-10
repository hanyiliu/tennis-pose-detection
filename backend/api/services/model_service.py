from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.api.core.config import Settings, get_settings
from backend.api.utils.encoding import pil_image_to_data_url
from backend.api.utils.image_outputs import (
    create_aggregated_heatmap_image,
    create_class_prediction_bar_graph,
    create_final_overlay_image,
    create_keypoint_overlay_image,
)
from inference.inference import InferencePipeline
from models.bbox_detection import BBoxDetectionModel
from models.keypoint_attention_detection import KeypointAttentionDetectionModel
from models.keypoint_detection import KeypointDetectionModel
from models.pose_classification import PoseClassificationModel
from preprocessing.image_preprocessing import convert_image_to_tensor
from preprocessing.pil_preprocessing import crop_pil, letterbox_resize, norm_bbox_to_xyxy_pixels
from preprocessing.tensor_preprocessing import heatmaps_to_keypoints, normalize_keypoints_xy

POSE_KEYS = ["backhand", "forehand", "ready_position", "serve"]


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self.bbox_image_size = (256, 256)
        self.keypoint_image_size = (128, 128)

        self.bbox_model = BBoxDetectionModel().to(self.device)
        self.keypoint_model = KeypointDetectionModel(num_keypoints=18).to(self.device)
        self.pose_model = None
        self.label_names = POSE_KEYS

        e2e_candidates = self._e2e_candidate_paths(settings.pose_model_path)
        loaded_from_e2e = self._try_load_e2e_bundle(e2e_candidates)
        pose_candidates = self._pose_candidate_paths(settings.pose_model_path)
        if not loaded_from_e2e:
            self.pose_model = self._load_pose_model(pose_candidates)
            self.label_names = self._load_pose_label_names(pose_candidates)
            self._load_bbox_weights(self._candidate_paths(settings.bbox_model_path, "bbox_best.pt"))
            self._load_keypoint_weights(self._candidate_paths(settings.keypoint_model_path, "keypoint_best_state_dict.pt"))

        self.bbox_model.eval()
        self.keypoint_model.eval()
        self.pose_model.eval()

        self.pipeline = InferencePipeline(
            bbox_detection=self.bbox_model,
            keypoint_detection=self.keypoint_model,
            pose_detection=self.pose_model,
            bbox_image_size=self.bbox_image_size,
            keypoint_image_size=self.keypoint_image_size,
        )

    def _candidate_paths(self, preferred_path: str, fallback_name: str) -> list[Path]:
        candidates: list[Path] = []
        seen = set()
        for raw_path in [preferred_path, f"exports/{fallback_name}", f"checkpoints/{fallback_name}"]:
            resolved = self.settings.resolve_path(raw_path)
            if str(resolved) not in seen:
                seen.add(str(resolved))
                candidates.append(resolved)
        return candidates

    def _pose_candidate_paths(self, preferred_path: str) -> list[Path]:
        candidates: list[Path] = []
        seen = set()
        for raw_path in [
            preferred_path,
            "exports/colab_e2e_best_2.pt",
            "checkpoints/colab_e2e_best_2.pt",
            "exports/e2e_best.pt",
            "checkpoints/e2e_best.pt",
            "exports/pose_best_predicted_attention.pt",
            "checkpoints/pose_best_predicted_attention.pt",
            "exports/pose_best_predicted.pt",
            "checkpoints/pose_best_predicted.pt",
            "exports/pose_best.pt",
            "checkpoints/pose_best.pt",
        ]:
            resolved = self.settings.resolve_path(raw_path)
            if str(resolved) not in seen:
                seen.add(str(resolved))
                candidates.append(resolved)
        return candidates

    def _e2e_candidate_paths(self, preferred_path: str) -> list[Path]:
        candidates: list[Path] = []
        seen = set()
        for raw_path in [
            "exports/colab_e2e_best_2.pt",
            "checkpoints/colab_e2e_best_2.pt",
            preferred_path,
            "exports/e2e_best.pt",
            "checkpoints/e2e_best.pt",
        ]:
            resolved = self.settings.resolve_path(raw_path)
            if str(resolved) not in seen:
                seen.add(str(resolved))
                candidates.append(resolved)
        return candidates

    def _try_load_e2e_bundle(self, candidate_paths: list[Path]) -> bool:
        for checkpoint_path in candidate_paths:
            if not checkpoint_path.exists():
                continue

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                continue

            if not all(key in checkpoint for key in ["bbox_model_state", "keypoint_model_state", "pose_model_state"]):
                continue

            self.bbox_model.load_state_dict(checkpoint["bbox_model_state"])

            keypoint_state = checkpoint["keypoint_model_state"]
            if not isinstance(keypoint_state, dict):
                raise ValueError(f"Invalid keypoint_model_state in e2e checkpoint: {checkpoint_path}")

            num_keypoints = 18
            has_attention_keys = any(key.startswith("encoder.stem.") for key in keypoint_state.keys())
            if has_attention_keys:
                self.keypoint_model = KeypointAttentionDetectionModel(num_keypoints=num_keypoints).to(self.device)
            else:
                self.keypoint_model = KeypointDetectionModel(num_keypoints=num_keypoints).to(self.device)
            self.keypoint_model.load_state_dict(keypoint_state)

            self.keypoint_image_size = (
                int(checkpoint.get("keypoint_image_height", self.keypoint_image_size[0])),
                int(checkpoint.get("keypoint_image_width", self.keypoint_image_size[1])),
            )

            args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args", {}), dict) else {}
            label_names = checkpoint.get("label_names")
            if isinstance(label_names, list) and len(label_names) == 4:
                normalized_labels = [_normalize_label(label) for label in label_names]
            else:
                normalized_labels = POSE_KEYS

            out_weight = checkpoint["pose_model_state"].get("out.weight")
            inferred_num_classes = int(out_weight.shape[0]) if isinstance(out_weight, torch.Tensor) else len(normalized_labels)
            self.pose_model = PoseClassificationModel(
                num_keypoints=18,
                num_classes=inferred_num_classes,
                hidden_dim=int(args.get("hidden_dim", 384)),
                dropout=float(args.get("dropout", 0.4)),
                visibility_threshold=float(args.get("visibility_threshold", 0.0)),
            ).to(self.device)
            self.pose_model.load_state_dict(checkpoint["pose_model_state"])
            self.label_names = normalized_labels
            return True

        return False

    def _load_bbox_weights(self, candidate_paths: list[Path]):
        errors: list[str] = []
        for model_path in candidate_paths:
            if not model_path.exists():
                continue
            try:
                state = torch.load(model_path, map_location=self.device)
                self.bbox_model.load_state_dict(state)
                return
            except RuntimeError as error:
                errors.append(f"{model_path}: {error}")
        if errors:
            raise RuntimeError("Unable to load a compatible bbox checkpoint. " + " | ".join(errors))
        raise FileNotFoundError(f"BBox checkpoint not found in any candidate path: {candidate_paths}")

    def _load_keypoint_weights(self, candidate_paths: list[Path]):
        errors: list[str] = []
        for model_path in candidate_paths:
            if not model_path.exists():
                continue
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    self.keypoint_model.load_state_dict(checkpoint["model_state"])
                    image_height = checkpoint.get("image_height")
                    image_width = checkpoint.get("image_width")
                    if isinstance(image_height, int) and isinstance(image_width, int):
                        self.keypoint_image_size = (image_height, image_width)
                else:
                    self.keypoint_model.load_state_dict(checkpoint)
                return
            except RuntimeError as error:
                errors.append(f"{model_path}: {error}")
        if errors:
            raise RuntimeError("Unable to load a compatible keypoint checkpoint. " + " | ".join(errors))
        raise FileNotFoundError(f"Keypoint checkpoint not found in any candidate path: {candidate_paths}")

    def _load_pose_model(self, candidate_paths: list[Path]):
        load_errors: list[str] = []
        for checkpoint_path in candidate_paths:
            if not checkpoint_path.exists():
                continue
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if not isinstance(checkpoint, dict):
                    raise ValueError("Unsupported pose checkpoint format. Expected dict payload.")

                if "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                elif "pose_model_state" in checkpoint:
                    state_dict = checkpoint["pose_model_state"]
                else:
                    state_dict = checkpoint

                if not isinstance(state_dict, dict):
                    raise ValueError("Unsupported pose checkpoint format. Missing valid model state dictionary.")

                args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args", {}), dict) else {}
                out_weight = state_dict.get("out.weight")
                inferred_num_classes = int(out_weight.shape[0]) if isinstance(out_weight, torch.Tensor) else 4

                model = PoseClassificationModel(
                    num_keypoints=18,
                    num_classes=inferred_num_classes,
                    hidden_dim=int(args.get("hidden_dim", 384)),
                    dropout=float(args.get("dropout", 0.4)),
                    visibility_threshold=float(args.get("visibility_threshold", 0.0)),
                ).to(self.device)
                model.load_state_dict(state_dict)
                return model
            except RuntimeError as error:
                load_errors.append(f"{checkpoint_path}: {error}")
            except ValueError as error:
                load_errors.append(f"{checkpoint_path}: {error}")

        if load_errors:
            raise RuntimeError("Unable to load a compatible pose checkpoint. " + " | ".join(load_errors))
        raise FileNotFoundError(f"Pose checkpoint not found in any candidate path: {candidate_paths}")

    def _load_pose_label_names(self, candidate_paths: list[Path]):
        for checkpoint_path in candidate_paths:
            if not checkpoint_path.exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            label_names = checkpoint.get("label_names")
            if isinstance(label_names, list) and len(label_names) == 4:
                return [_normalize_label(label) for label in label_names]
        return POSE_KEYS

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image_width, image_height = image.size

        bbox_h, bbox_w = self.bbox_image_size
        bbox_img = image.copy().resize((bbox_w, bbox_h), Image.Resampling.BILINEAR)
        bbox_tensor = convert_image_to_tensor(bbox_img).to(self.device)
        bbox_pred = self.bbox_model(bbox_tensor.unsqueeze(0)).squeeze(0)
        bbox_xyxy = norm_bbox_to_xyxy_pixels(bbox_pred, image_width, image_height)
        x1, y1, x2, y2 = bbox_xyxy

        cropped_bbox = crop_pil(image, bbox_xyxy)
        keypoint_input = letterbox_resize(cropped_bbox.copy(), size=self.keypoint_image_size)
        keypoint_tensor = convert_image_to_tensor(keypoint_input).to(self.device)
        heatmaps = self.keypoint_model(keypoint_tensor.unsqueeze(0))
        keypoints = heatmaps_to_keypoints(heatmaps).squeeze(0)

        heat_h = heatmaps.size(2)
        heat_w = heatmaps.size(3)
        keypoints_norm = normalize_keypoints_xy(keypoints, heat_h, heat_w).squeeze(0)

        probs = self.pipeline.pose_detection(keypoints_norm.to(self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        conf_map = self._map_pose_probabilities(probs)
        predicted_label = max(conf_map, key=lambda label: conf_map[label])

        keypoint_positions = np.asarray(keypoints_norm.cpu(), dtype=float).tolist()
        keypoint_overlay = create_keypoint_overlay_image(
            keypoint_input,
            keypoints,
            (heat_h, heat_w),
        )
        heatmap_image = create_aggregated_heatmap_image(heatmaps.squeeze(0))
        class_bar_graph = create_class_prediction_bar_graph(conf_map)
        final_overlay = create_final_overlay_image(
            image=image,
            bbox_xyxy=bbox_xyxy,
            keypoints_xyv=keypoints,
            keypoint_image_size=self.keypoint_image_size,
            prediction_label=predicted_label,
            confidence=conf_map[predicted_label],
        )

        return {
            "bbox_x": int(x1),
            "bbox_y": int(y1),
            "bbox_width": int(max(x2 - x1, 1)),
            "bbox_height": int(max(y2 - y1, 1)),
            "cropped_bbox_image": pil_image_to_data_url(cropped_bbox),
            "keypoint_positions": keypoint_positions,
            "keypoint_aggregated_heatmap_url": pil_image_to_data_url(heatmap_image),
            "keypoint_cropped_image": pil_image_to_data_url(keypoint_overlay),
            "backhand_conf": float(conf_map["backhand"]),
            "forehand_conf": float(conf_map["forehand"]),
            "ready_position_conf": float(conf_map["ready_position"]),
            "serve_conf": float(conf_map["serve"]),
            "class_prediction_bar_graph": pil_image_to_data_url(class_bar_graph),
            "overlayed_image": pil_image_to_data_url(final_overlay),
        }

    def _map_pose_probabilities(self, probs: np.ndarray) -> dict[str, float]:
        conf_map = {key: 0.0 for key in POSE_KEYS}
        for idx, label in enumerate(self.label_names):
            if idx < len(probs) and label in conf_map:
                conf_map[label] = float(probs[idx])
        return conf_map


@lru_cache
def get_model_service() -> ModelService:
    settings = get_settings()
    return ModelService(settings)
