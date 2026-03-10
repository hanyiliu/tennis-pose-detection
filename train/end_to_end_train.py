import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from models.bbox_detection import BBoxDetectionModel
from models.keypoint_detection import KeypointDetectionModel
from models.pose_classification import PoseClassificationModel


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _norm_label(name: str) -> str:
	return str(name).strip().lower().replace(" ", "_")


def resolve_dataset_root(root_dir_arg: Optional[str]) -> str:
	if root_dir_arg:
		if not os.path.isdir(root_dir_arg):
			raise FileNotFoundError(f"Provided --root_dir does not exist: {root_dir_arg}")
		return root_dir_arg

	candidates = [
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
	for candidate in candidates:
		if os.path.isdir(candidate):
			return candidate

	raise FileNotFoundError(
		"Could not find dataset root in repository /datasets directory. Pass --root_dir explicitly."
	)


class TennisE2EDataset(Dataset):
	def __init__(
		self,
		root_dir: str,
		annotation_files: List[str],
		image_size: Tuple[int, int],
		class_order: Optional[List[str]] = None,
	):
		self.root_dir = root_dir
		self.samples: List[Dict] = []
		self.image_transform = transforms.Compose([
			transforms.Resize(image_size),
			transforms.ToTensor(),
		])

		cat_id_to_name: Dict[int, str] = {}
		json_datas = []

		for ann_rel_path in annotation_files:
			ann_path = os.path.join(root_dir, ann_rel_path)
			with Path(ann_path).open("r", encoding="utf-8") as f:
				data = json.load(f)
			json_datas.append(data)
			for category in data.get("categories", []):
				cat_id_to_name[category["id"]] = category.get("name", str(category["id"]))

		if not cat_id_to_name:
			raise RuntimeError("No categories found in annotation files.")

		if class_order is not None:
			name_to_id = {name: cat_id for cat_id, name in cat_id_to_name.items()}
			missing = [name for name in class_order if name not in name_to_id]
			if missing:
				raise ValueError(f"class_order names not found in categories: {missing}")
			ordered_ids = [name_to_id[name] for name in class_order]
			self.label_names = class_order[:]
		else:
			ordered_ids = sorted(cat_id_to_name.keys())
			self.label_names = [cat_id_to_name[cat_id] for cat_id in ordered_ids]

		label_map = {cat_id: i for i, cat_id in enumerate(ordered_ids)}

		for data in json_datas:
			id_to_img = {img["id"]: img for img in data.get("images", [])}
			for ann in data.get("annotations", []):
				image_id = ann.get("image_id")
				category_id = ann.get("category_id")
				bbox = ann.get("bbox")
				keypoints = ann.get("keypoints")
				if category_id not in label_map:
					continue
				if not isinstance(bbox, list) or len(bbox) != 4:
					continue
				if not isinstance(keypoints, list) or len(keypoints) != 54:
					continue

				img_info = id_to_img.get(image_id)
				if img_info is None:
					continue

				rel_path = img_info.get("path", img_info.get("file_name"))
				if rel_path is None:
					continue
				rel_path = rel_path.replace("../", "")
				img_path = os.path.join(root_dir, rel_path)
				if not os.path.isfile(img_path):
					continue

				width = float(img_info.get("width", 1.0))
				height = float(img_info.get("height", 1.0))
				if width <= 0 or height <= 0:
					continue

				x, y, w, h = [float(v) for v in bbox]
				if w <= 0 or h <= 0:
					continue

				bbox_norm = [x / width, y / height, w / width, h / height]

				keypoints_np = np.array(keypoints, dtype=np.float32).reshape(18, 3)
				keypoints_norm = np.zeros((18, 3), dtype=np.float32)
				keypoints_norm[:, 0] = np.clip(keypoints_np[:, 0] / width, 0.0, 1.0)
				keypoints_norm[:, 1] = np.clip(keypoints_np[:, 1] / height, 0.0, 1.0)
				keypoints_norm[:, 2] = (keypoints_np[:, 2] > 0).astype(np.float32)

				self.samples.append(
					{
						"img_path": img_path,
						"bbox_norm": bbox_norm,
						"keypoints_norm": keypoints_norm,
						"label": label_map[category_id],
					}
				)

		if not self.samples:
			raise RuntimeError("Loaded 0 samples for E2E training.")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		sample = self.samples[index]
		image = Image.open(sample["img_path"]).convert("RGB")
		image_tensor = self.image_transform(image)

		bbox = torch.tensor(sample["bbox_norm"], dtype=torch.float32)
		keypoints = torch.tensor(sample["keypoints_norm"], dtype=torch.float32)
		label = torch.tensor(sample["label"], dtype=torch.long)
		return image_tensor, bbox, keypoints, label


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 0] + boxes[:, 2]
	y2 = boxes[:, 1] + boxes[:, 3]
	return torch.stack([x1, y1, x2, y2], dim=1)


def sanitize_xyxy(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	boxes = boxes.clamp(0.0, 1.0)
	x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
	y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
	x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
	y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
	x2 = torch.maximum(x2, x1 + eps)
	y2 = torch.maximum(y2, y1 + eps)
	return torch.stack([x1, y1, x2, y2], dim=1)


def bbox_iou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
	px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
	tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

	ix1 = torch.maximum(px1, tx1)
	iy1 = torch.maximum(py1, ty1)
	ix2 = torch.minimum(px2, tx2)
	iy2 = torch.minimum(py2, ty2)

	iw = (ix2 - ix1).clamp(min=0.0)
	ih = (iy2 - iy1).clamp(min=0.0)
	inter = iw * ih

	p_area = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
	t_area = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
	union = p_area + t_area - inter
	return inter / (union + eps)


def bbox_loss(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, iou_lambda: float) -> torch.Tensor:
	reg = F.smooth_l1_loss(pred_xywh, target_xywh)
	pred_xyxy = sanitize_xyxy(xywh_to_xyxy(pred_xywh))
	target_xyxy = sanitize_xyxy(xywh_to_xyxy(target_xywh))
	iou = bbox_iou_xyxy(pred_xyxy, target_xyxy)
	iou_term = (1.0 - iou).mean()
	return reg + iou_lambda * iou_term


def differentiable_crop_resize(
	images: torch.Tensor,
	boxes_xywh: torch.Tensor,
	out_size: Tuple[int, int],
) -> torch.Tensor:
	bsz, channels, _, _ = images.shape
	out_h, out_w = out_size

	x1 = boxes_xywh[:, 0].clamp(0.0, 1.0)
	y1 = boxes_xywh[:, 1].clamp(0.0, 1.0)
	w = boxes_xywh[:, 2].clamp(min=1e-4, max=1.0)
	h = boxes_xywh[:, 3].clamp(min=1e-4, max=1.0)
	x2 = (x1 + w).clamp(0.0, 1.0)
	y2 = (y1 + h).clamp(0.0, 1.0)

	x2 = torch.maximum(x2, x1 + 1e-4)
	y2 = torch.maximum(y2, y1 + 1e-4)

	a = (x2 - x1)
	c = (y2 - y1)
	b = (x1 + x2 - 1.0)
	d = (y1 + y2 - 1.0)

	theta = torch.zeros((bsz, 2, 3), device=images.device, dtype=images.dtype)
	theta[:, 0, 0] = a
	theta[:, 0, 2] = b
	theta[:, 1, 1] = c
	theta[:, 1, 2] = d

	grid = F.affine_grid(theta, size=(bsz, channels, out_h, out_w), align_corners=False)
	cropped = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
	return cropped


def soft_argmax_keypoints(heatmaps: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
	if heatmaps.dim() != 4:
		raise ValueError(f"Expected heatmaps shape (B,K,H,W), got {tuple(heatmaps.shape)}")

	bsz, num_keypoints, h, w = heatmaps.shape
	logits = heatmaps.view(bsz, num_keypoints, -1) / max(temperature, 1e-6)
	probs = torch.softmax(logits, dim=-1).view(bsz, num_keypoints, h, w)

	ys = torch.linspace(0.0, 1.0, steps=h, device=heatmaps.device, dtype=heatmaps.dtype)
	xs = torch.linspace(0.0, 1.0, steps=w, device=heatmaps.device, dtype=heatmaps.dtype)
	yy, xx = torch.meshgrid(ys, xs, indexing="ij")

	exp_x = (probs * xx).sum(dim=(2, 3))
	exp_y = (probs * yy).sum(dim=(2, 3))
	vis = torch.sigmoid(heatmaps.amax(dim=(2, 3)))
	return torch.stack([exp_x, exp_y, vis], dim=-1)


def map_local_to_global_keypoints(local_keypoints: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
	x = boxes_xywh[:, 0].unsqueeze(1) + local_keypoints[:, :, 0] * boxes_xywh[:, 2].unsqueeze(1)
	y = boxes_xywh[:, 1].unsqueeze(1) + local_keypoints[:, :, 1] * boxes_xywh[:, 3].unsqueeze(1)
	vis = local_keypoints[:, :, 2]
	x = x.clamp(0.0, 1.0)
	y = y.clamp(0.0, 1.0)
	return torch.stack([x, y, vis], dim=-1)


def keypoint_loss(pred_global_keypoints: torch.Tensor, target_global_keypoints: torch.Tensor) -> torch.Tensor:
	target_xy = target_global_keypoints[:, :, 0:2]
	target_vis = target_global_keypoints[:, :, 2]

	pred_xy = pred_global_keypoints[:, :, 0:2]
	pred_vis = pred_global_keypoints[:, :, 2]

	vis_mask = target_vis > 0.5
	if vis_mask.any():
		xy_loss = F.smooth_l1_loss(pred_xy[vis_mask], target_xy[vis_mask])
	else:
		xy_loss = pred_xy.sum() * 0.0

	vis_loss = F.binary_cross_entropy(pred_vis, target_vis)
	return xy_loss + 0.2 * vis_loss


def set_stage_trainable(
	bbox_model: nn.Module,
	keypoint_model: nn.Module,
	pose_model: nn.Module,
	epoch: int,
	unfreeze_stage2_epoch: int,
	unfreeze_stage1_epoch: int,
) -> str:
	train_stage2 = epoch >= unfreeze_stage2_epoch
	train_stage1 = epoch >= unfreeze_stage1_epoch

	for param in pose_model.parameters():
		param.requires_grad = True
	for param in keypoint_model.parameters():
		param.requires_grad = train_stage2
	for param in bbox_model.parameters():
		param.requires_grad = train_stage1

	if train_stage1:
		return "stage1+stage2+stage3"
	if train_stage2:
		return "stage2+stage3"
	return "stage3"


def load_checkpoint_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
	payload = torch.load(path, map_location=device)
	if isinstance(payload, dict) and "model_state" in payload:
		state = payload["model_state"]
	else:
		state = payload
	if not isinstance(state, dict):
		raise ValueError(f"Unexpected checkpoint format at {path}")
	return state


@torch.no_grad()
def evaluate(
	bbox_model: nn.Module,
	keypoint_model: nn.Module,
	pose_model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	keypoint_image_size: Tuple[int, int],
	keypoint_temperature: float,
	bbox_loss_weight: float,
	keypoint_loss_weight: float,
	pose_loss_weight: float,
	iou_lambda: float,
) -> Tuple[float, float, float, float, float]:
	bbox_model.eval()
	keypoint_model.eval()
	pose_model.eval()

	total_loss = 0.0
	total_bbox_loss = 0.0
	total_keypoint_loss = 0.0
	total_pose_loss = 0.0
	total_count = 0
	correct = 0

	for images, target_bbox, target_keypoints, labels in loader:
		images = images.to(device)
		target_bbox = target_bbox.to(device)
		target_keypoints = target_keypoints.to(device)
		labels = labels.to(device)

		pred_bbox = bbox_model(images)
		crop = differentiable_crop_resize(images, pred_bbox, keypoint_image_size)
		heatmaps = keypoint_model(crop)
		pred_local_keypoints = soft_argmax_keypoints(heatmaps, temperature=keypoint_temperature)
		pred_global_keypoints = map_local_to_global_keypoints(pred_local_keypoints, pred_bbox)

		logits = pose_model(pred_global_keypoints, return_logits=True)

		loss_bbox = bbox_loss(pred_bbox, target_bbox, iou_lambda=iou_lambda)
		loss_kp = keypoint_loss(pred_global_keypoints, target_keypoints)
		loss_pose = F.cross_entropy(logits, labels)
		loss = bbox_loss_weight * loss_bbox + keypoint_loss_weight * loss_kp + pose_loss_weight * loss_pose

		bs = images.size(0)
		total_loss += loss.item() * bs
		total_bbox_loss += loss_bbox.item() * bs
		total_keypoint_loss += loss_kp.item() * bs
		total_pose_loss += loss_pose.item() * bs
		total_count += bs
		correct += (logits.argmax(dim=1) == labels).sum().item()

	mean_loss = total_loss / max(total_count, 1)
	mean_bbox_loss = total_bbox_loss / max(total_count, 1)
	mean_keypoint_loss = total_keypoint_loss / max(total_count, 1)
	mean_pose_loss = total_pose_loss / max(total_count, 1)
	accuracy = correct / max(total_count, 1)
	return mean_loss, mean_bbox_loss, mean_keypoint_loss, mean_pose_loss, accuracy


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--bbox_checkpoint", type=str, default="exports/bbox_best.pt")
	parser.add_argument("--keypoint_checkpoint", type=str, default="exports/keypoint_best_state_dict.pt")
	parser.add_argument("--pose_checkpoint", type=str, default="exports/pose_best_predicted.pt")

	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)

	parser.add_argument("--train_split", type=float, default=0.7)
	parser.add_argument("--val_split", type=float, default=0.15)
	parser.add_argument("--test_split", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--bbox_image_height", type=int, default=256)
	parser.add_argument("--bbox_image_width", type=int, default=256)
	parser.add_argument("--keypoint_image_height", type=int, default=128)
	parser.add_argument("--keypoint_image_width", type=int, default=128)
	parser.add_argument("--keypoint_temperature", type=float, default=0.05)

	parser.add_argument("--bbox_loss_weight", type=float, default=1.0)
	parser.add_argument("--keypoint_loss_weight", type=float, default=1.0)
	parser.add_argument("--pose_loss_weight", type=float, default=1.0)
	parser.add_argument("--iou_lambda", type=float, default=0.5)

	parser.add_argument("--unfreeze_stage2_epoch", type=int, default=4)
	parser.add_argument("--unfreeze_stage1_epoch", type=int, default=8)

	parser.add_argument("--early_stop_patience", type=int, default=10)
	parser.add_argument("--output_checkpoint", type=str, default="checkpoints/e2e_best.pt")
	parser.add_argument("--export_path", type=str, default="exports/e2e_best.pt")

	args = parser.parse_args()

	total_split = args.train_split + args.val_split + args.test_split
	if abs(total_split - 1.0) > 1e-6:
		raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total_split}")

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	root_dir = resolve_dataset_root(args.root_dir)
	annotation_files = [
		"annotations/backhand.json",
		"annotations/forehand.json",
		"annotations/ready_position.json",
		"annotations/serve.json",
	]

	device = get_device()

	pose_payload = torch.load(args.pose_checkpoint, map_location="cpu")
	pose_args = pose_payload.get("args", {}) if isinstance(pose_payload, dict) else {}
	pose_label_names = pose_payload.get("label_names", ["backhand", "forehand", "ready_position", "serve"]) if isinstance(pose_payload, dict) else ["backhand", "forehand", "ready_position", "serve"]

	category_names = []
	seen_names = set()
	for rel_path in annotation_files:
		ann_path = os.path.join(root_dir, rel_path)
		with open(ann_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		for category in data.get("categories", []):
			name = category.get("name")
			if isinstance(name, str) and name not in seen_names:
				seen_names.add(name)
				category_names.append(name)

	canonical_by_norm = {_norm_label(name): name for name in category_names}
	class_order = []
	missing = []
	for label in pose_label_names:
		canonical = canonical_by_norm.get(_norm_label(label))
		if canonical is None:
			missing.append(label)
		else:
			class_order.append(canonical)
	if missing:
		raise ValueError(f"Could not map pose labels to annotation categories: {missing}")

	bbox_image_size = (args.bbox_image_height, args.bbox_image_width)
	keypoint_image_size = (args.keypoint_image_height, args.keypoint_image_width)

	dataset = TennisE2EDataset(
		root_dir=root_dir,
		annotation_files=annotation_files,
		image_size=bbox_image_size,
		class_order=class_order,
	)

	ds_size = len(dataset)
	train_size = int(ds_size * args.train_split)
	val_size = int(ds_size * args.val_split)
	test_size = ds_size - train_size - val_size
	train_ds, val_ds, test_ds = random_split(
		dataset,
		[train_size, val_size, test_size],
		generator=torch.Generator().manual_seed(args.seed),
	)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

	bbox_model = BBoxDetectionModel().to(device)
	keypoint_model = KeypointDetectionModel(num_keypoints=18).to(device)
	pose_model = PoseClassificationModel(
		num_keypoints=18,
		num_classes=len(class_order),
		hidden_dim=int(pose_args.get("hidden_dim", 512)),
		dropout=float(pose_args.get("dropout", 0.25)),
		visibility_threshold=float(pose_args.get("visibility_threshold", 0.0)),
	).to(device)

	bbox_model.load_state_dict(load_checkpoint_state(args.bbox_checkpoint, device))

	keypoint_payload = torch.load(args.keypoint_checkpoint, map_location=device)
	if isinstance(keypoint_payload, dict) and "model_state" in keypoint_payload:
		keypoint_model.load_state_dict(keypoint_payload["model_state"])
	else:
		keypoint_model.load_state_dict(keypoint_payload)

	pose_state = load_checkpoint_state(args.pose_checkpoint, device)
	pose_model.load_state_dict(pose_state)

	print(f"Dataset root: {root_dir}")
	print(f"Device: {device}")
	print(f"Train/Val/Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
	print(f"Loaded bbox checkpoint: {args.bbox_checkpoint}")
	print(f"Loaded keypoint checkpoint: {args.keypoint_checkpoint}")
	print(f"Loaded pose checkpoint: {args.pose_checkpoint}")

	best_val_acc = -1.0
	best_val_loss = float("inf")
	epochs_without_improve = 0
	current_phase = None
	optimizer = None
	scheduler = None

	output_checkpoint_dir = os.path.dirname(args.output_checkpoint)
	export_dir = os.path.dirname(args.export_path)
	if output_checkpoint_dir:
		os.makedirs(output_checkpoint_dir, exist_ok=True)
	if export_dir:
		os.makedirs(export_dir, exist_ok=True)

	for epoch in range(1, args.epochs + 1):
		phase = set_stage_trainable(
			bbox_model,
			keypoint_model,
			pose_model,
			epoch,
			args.unfreeze_stage2_epoch,
			args.unfreeze_stage1_epoch,
		)

		if phase != current_phase:
			trainable_params = [
				param for param in list(bbox_model.parameters()) + list(keypoint_model.parameters()) + list(pose_model.parameters())
				if param.requires_grad
			]
			optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode="max",
				factor=0.5,
				patience=4,
			)
			current_phase = phase
			print(f"[Epoch {epoch}] Training phase: {phase}")

		bbox_model.train()
		keypoint_model.train()
		pose_model.train()

		total_loss = 0.0
		total_bbox = 0.0
		total_kp = 0.0
		total_pose = 0.0
		total_count = 0
		correct = 0

		for images, target_bbox, target_keypoints, labels in train_loader:
			images = images.to(device)
			target_bbox = target_bbox.to(device)
			target_keypoints = target_keypoints.to(device)
			labels = labels.to(device)

			optimizer.zero_grad(set_to_none=True)

			pred_bbox = bbox_model(images)
			crop = differentiable_crop_resize(images, pred_bbox, keypoint_image_size)
			heatmaps = keypoint_model(crop)
			pred_local_keypoints = soft_argmax_keypoints(heatmaps, temperature=args.keypoint_temperature)
			pred_global_keypoints = map_local_to_global_keypoints(pred_local_keypoints, pred_bbox)

			logits = pose_model(pred_global_keypoints, return_logits=True)

			loss_bbox = bbox_loss(pred_bbox, target_bbox, iou_lambda=args.iou_lambda)
			loss_kp = keypoint_loss(pred_global_keypoints, target_keypoints)
			loss_pose = F.cross_entropy(logits, labels)
			loss = (
				args.bbox_loss_weight * loss_bbox
				+ args.keypoint_loss_weight * loss_kp
				+ args.pose_loss_weight * loss_pose
			)

			loss.backward()
			torch.nn.utils.clip_grad_norm_(
				[param for param in list(bbox_model.parameters()) + list(keypoint_model.parameters()) + list(pose_model.parameters()) if param.requires_grad],
				max_norm=args.grad_clip_norm,
			)
			optimizer.step()

			bs = images.size(0)
			total_loss += loss.item() * bs
			total_bbox += loss_bbox.item() * bs
			total_kp += loss_kp.item() * bs
			total_pose += loss_pose.item() * bs
			total_count += bs
			correct += (logits.argmax(dim=1) == labels).sum().item()

		train_loss = total_loss / max(total_count, 1)
		train_bbox_loss = total_bbox / max(total_count, 1)
		train_keypoint_loss = total_kp / max(total_count, 1)
		train_pose_loss = total_pose / max(total_count, 1)
		train_acc = correct / max(total_count, 1)

		val_loss, val_bbox_loss, val_keypoint_loss, val_pose_loss, val_acc = evaluate(
			bbox_model=bbox_model,
			keypoint_model=keypoint_model,
			pose_model=pose_model,
			loader=val_loader,
			device=device,
			keypoint_image_size=keypoint_image_size,
			keypoint_temperature=args.keypoint_temperature,
			bbox_loss_weight=args.bbox_loss_weight,
			keypoint_loss_weight=args.keypoint_loss_weight,
			pose_loss_weight=args.pose_loss_weight,
			iou_lambda=args.iou_lambda,
		)

		scheduler.step(val_acc)

		print(
			f"Epoch {epoch:03d}/{args.epochs} | "
			f"train loss: {train_loss:.4f} (bbox {train_bbox_loss:.4f}, kp {train_keypoint_loss:.4f}, pose {train_pose_loss:.4f}) acc {train_acc:.4f} | "
			f"val loss: {val_loss:.4f} (bbox {val_bbox_loss:.4f}, kp {val_keypoint_loss:.4f}, pose {val_pose_loss:.4f}) acc {val_acc:.4f}"
		)

		improved = False
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			improved = True
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			improved = True

		if improved:
			epochs_without_improve = 0
			checkpoint = {
				"bbox_model_state": bbox_model.state_dict(),
				"keypoint_model_state": keypoint_model.state_dict(),
				"pose_model_state": pose_model.state_dict(),
				"label_names": dataset.label_names,
				"keypoint_image_height": args.keypoint_image_height,
				"keypoint_image_width": args.keypoint_image_width,
				"bbox_image_height": args.bbox_image_height,
				"bbox_image_width": args.bbox_image_width,
				"epoch": epoch,
				"best_val_acc": best_val_acc,
				"best_val_loss": best_val_loss,
				"args": vars(args),
			}
			torch.save(checkpoint, args.output_checkpoint)
			torch.save(checkpoint, args.export_path)
		else:
			epochs_without_improve += 1

		if epochs_without_improve >= args.early_stop_patience:
			print(f"Early stopping triggered at epoch {epoch}")
			break

	print(f"Best val acc: {best_val_acc:.4f}")
	print(f"Best val loss: {best_val_loss:.4f}")
	print(f"Saved checkpoint: {args.output_checkpoint}")
	print(f"Saved export: {args.export_path}")

	best_checkpoint = torch.load(args.output_checkpoint, map_location=device)
	bbox_model.load_state_dict(best_checkpoint["bbox_model_state"])
	keypoint_model.load_state_dict(best_checkpoint["keypoint_model_state"])
	pose_model.load_state_dict(best_checkpoint["pose_model_state"])

	test_loss, test_bbox_loss, test_keypoint_loss, test_pose_loss, test_acc = evaluate(
		bbox_model=bbox_model,
		keypoint_model=keypoint_model,
		pose_model=pose_model,
		loader=test_loader,
		device=device,
		keypoint_image_size=keypoint_image_size,
		keypoint_temperature=args.keypoint_temperature,
		bbox_loss_weight=args.bbox_loss_weight,
		keypoint_loss_weight=args.keypoint_loss_weight,
		pose_loss_weight=args.pose_loss_weight,
		iou_lambda=args.iou_lambda,
	)

	print(
		f"Test loss: {test_loss:.4f} "
		f"(bbox {test_bbox_loss:.4f}, kp {test_keypoint_loss:.4f}, pose {test_pose_loss:.4f}) "
		f"acc {test_acc:.4f}"
	)


if __name__ == "__main__":
	main()
