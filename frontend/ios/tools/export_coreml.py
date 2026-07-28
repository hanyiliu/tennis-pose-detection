#!/usr/bin/env python3
"""Export the end-to-end tennis pose checkpoint to Core ML packages for the iOS app.

Emits two ``.mlpackage`` bundles plus the two JSON sidecars that Swift reads from the
app bundle, so nothing about the model is hardcoded on the client:

    TPDBBox.mlpackage       image (1,3,256,256) -> bbox (1,4)  sigmoid xywh in [0,1]
    TPDPoseNet.mlpackage    image (1,3,128,128) -> probs (1,4) + keypoints (1,18,3)
    TPDLabels.json          normalized class names, in checkpoint order
    TPDModelSpec.json       input sizes / hyperparameters / checkpoint provenance

Stages 2 and 3 are fused into one graph: heatmap argmax decoding and normalization are
baked in, matching ``preprocessing/tensor_preprocessing.py`` exactly (raw max logit as
visibility, no sigmoid). Stage 1 stays separate because the crop between stages depends
on its output and cannot be expressed with static shapes.

Run ``python frontend/ios/tools/export_coreml.py --help`` for options.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn


def find_repo_root(start: Path) -> Path:
    """Walk up from *start* until the directory holding the ``models`` package is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "models" / "pose_classification.py").is_file():
            return candidate
    raise SystemExit(
        f"could not locate the repository root above {start} "
        "(looked for models/pose_classification.py)"
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.bbox_detection import BBoxDetectionModel  # noqa: E402
from models.keypoint_detection import KeypointDetectionModel  # noqa: E402
from models.pose_classification import PoseClassificationModel  # noqa: E402

# Absent from the checkpoint's ``args``; both are numerically inert at export time
# (dropout is disabled by eval(), and a 0.0 threshold makes the visibility mask a
# pass-through). Kept identical to backend/api/services/model_service.py.
DROPOUT = 0.4
VISIBILITY_THRESHOLD = 0.0


class PoseNet(nn.Module):
    """Fused stage 2 + stage 3: cropped player image in, class probabilities out.

    The decode between the two stages is the parity contract with
    ``preprocessing/tensor_preprocessing.py::extract_keypoints_from_heatmaps``.
    """

    def __init__(self, keypoint_model: nn.Module, pose_model: nn.Module):
        super().__init__()
        self.kp = keypoint_model
        self.pose = pose_model

    def forward(self, image: torch.Tensor):
        hm = self.kp(image)  # (B, K, H, W) raw logits
        b, k, h, w = hm.shape
        flat = hm.reshape(b, k, h * w)
        vis, idx = flat.max(dim=-1)  # visibility is the RAW max logit, never sigmoided
        idxf = idx.float()
        y = torch.floor(idxf / w)
        x = idxf - y * w
        kps = torch.stack([x / (w - 1), y / (h - 1), vis], dim=-1)  # (B, K, 3)
        probs = self.pose(kps)  # forward() already applies softmax
        return probs, kps


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def square_size(checkpoint: dict, prefix: str) -> int:
    height = int(checkpoint[f"{prefix}_image_height"])
    width = int(checkpoint[f"{prefix}_image_width"])
    if height != width:
        raise SystemExit(
            f"{prefix} input is {width}x{height}; the exporter and the Swift "
            "preprocessing both assume a square input."
        )
    return height


def pick_deployment_target(ct):
    """Highest iOS target the installed coremltools actually knows about."""
    for name in ("iOS18", "iOS17", "iOS16", "iOS15"):
        target = getattr(ct.target, name, None)
        if target is not None:
            return name, target
    return None, None


def convert(ct, module: nn.Module, example: torch.Tensor, output_names, precision, target):
    traced = torch.jit.trace(module, example, strict=False)
    traced.eval()
    kwargs = dict(
        inputs=[
            ct.ImageType(
                name="image",
                shape=tuple(example.shape),
                scale=1 / 255.0,  # no bias: the pipeline applies no mean/std normalization
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        compute_precision=precision,
    )
    if target is not None:
        kwargs["minimum_deployment_target"] = target
    return ct.convert(traced, **kwargs)


def save(model, path: Path) -> float:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return dir_size_mb(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "exports" / "colab_e2e_best_2.pt",
        help="end-to-end checkpoint to export (default: exports/colab_e2e_best_2.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "frontend" / "ios" / "TPD" / "Models",
        help="destination for the .mlpackage bundles and JSON sidecars",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Core ML weight/activation precision (default: fp16)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import coremltools as ct

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    out_dir = args.out_dir.expanduser().resolve()

    print(f"repo root  {REPO_ROOT}")
    print(f"checkpoint {checkpoint_path}")
    print(f"out dir    {out_dir}")
    print(f"coremltools {ct.__version__} / torch {torch.__version__}\n")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    bbox_size = square_size(ckpt, "bbox")
    kp_size = square_size(ckpt, "keypoint")

    # Hyperparameters live in the tensor shapes, not in ckpt["args"] -- that dict has no
    # hidden_dim, no dropout and no visibility_threshold.
    pose_state = ckpt["pose_model_state"]
    hidden_dim = int(pose_state["fc1.weight"].shape[0])
    num_classes = int(pose_state["out.weight"].shape[0])
    kp_state = ckpt["keypoint_model_state"]
    num_keypoints = int(kp_state["extraction_conv1.weight"].shape[0])

    labels = [str(name).strip().lower().replace(" ", "_") for name in ckpt["label_names"]]
    if len(labels) != num_classes:
        raise SystemExit(
            f"checkpoint has {len(labels)} label names but the classifier emits {num_classes}"
        )

    print("inferred hyperparameters")
    print(f"  bbox input       {bbox_size}x{bbox_size}")
    print(f"  keypoint input   {kp_size}x{kp_size}")
    print(f"  num_keypoints    {num_keypoints}   (extraction_conv1.weight)")
    print(f"  hidden_dim       {hidden_dim}   (pose fc1.weight)")
    print(f"  num_classes      {num_classes}   (pose out.weight)")
    print(f"  labels           {labels}")
    print(f"  epoch {ckpt.get('epoch')}  best_val_acc {ckpt.get('best_val_acc')}\n")

    bbox_model = BBoxDetectionModel()
    bbox_model.load_state_dict(ckpt["bbox_model_state"])
    bbox_model.eval()

    keypoint_model = KeypointDetectionModel(num_keypoints=num_keypoints)
    keypoint_model.load_state_dict(kp_state)
    keypoint_model.eval()

    pose_model = PoseClassificationModel(
        num_keypoints=num_keypoints,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=DROPOUT,
        visibility_threshold=VISIBILITY_THRESHOLD,
    )
    pose_model.load_state_dict(pose_state)
    pose_model.eval()

    posenet = PoseNet(keypoint_model, pose_model).eval()

    bbox_example = torch.rand(1, 3, bbox_size, bbox_size)
    pose_example = torch.rand(1, 3, kp_size, kp_size)

    with torch.no_grad():
        bbox_out = bbox_model(bbox_example)
        heatmaps = keypoint_model(pose_example)
        probs, kps = posenet(pose_example)

    hm_h, hm_w = int(heatmaps.shape[2]), int(heatmaps.shape[3])
    assert heatmaps.shape[1] == num_keypoints, f"heatmap channels {heatmaps.shape[1]}"
    assert (hm_h, hm_w) == (kp_size, kp_size), f"heatmap is {hm_w}x{hm_h}, expected {kp_size}"
    print("traced shapes")
    print(f"  bbox      {tuple(bbox_example.shape)} -> {tuple(bbox_out.shape)}")
    print(f"  heatmaps  {tuple(pose_example.shape)} -> {tuple(heatmaps.shape)}")
    print(f"  posenet   {tuple(pose_example.shape)} -> probs {tuple(probs.shape)}, "
          f"keypoints {tuple(kps.shape)}\n")

    precision = ct.precision.FLOAT32 if args.precision == "fp32" else ct.precision.FLOAT16
    target_name, target = pick_deployment_target(ct)
    print(f"precision {args.precision}, minimum_deployment_target "
          f"{target_name or 'unset (coremltools default)'}\n")

    bbox_path = out_dir / "TPDBBox.mlpackage"
    posenet_path = out_dir / "TPDPoseNet.mlpackage"

    print("converting TPDBBox ...")
    bbox_ml = convert(ct, bbox_model, bbox_example, ["bbox"], precision, target)
    bbox_mb = save(bbox_ml, bbox_path)

    print("converting TPDPoseNet (stages 2+3 fused) ...")
    try:
        posenet_ml = convert(
            ct, posenet, pose_example, ["probs", "keypoints"], precision, target
        )
    except Exception as exc:  # noqa: BLE001 - the fallback decision needs every failure
        print("\n" + "!" * 78, file=sys.stderr)
        print("FUSED STAGE 2+3 CONVERSION FAILED -- no TPDPoseNet.mlpackage was written.",
              file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThe heatmap argmax decode could not be lowered into a Core ML program.\n"
            "Take the documented escape hatch in frontend/IOS_IMPLEMENTATION_PLAN.md "
            "section 3:\nexport stage 2 raw (18 heatmaps out) and stage 3 on its own, then "
            "decode\nargmax/normalize in Swift. Do NOT ship a partially converted graph.",
            file=sys.stderr,
        )
        print("!" * 78, file=sys.stderr)
        return 1
    posenet_mb = save(posenet_ml, posenet_path)

    labels_path = out_dir / "TPDLabels.json"
    spec_path = out_dir / "TPDModelSpec.json"
    labels_path.write_text(json.dumps({"labels": labels}, indent=2) + "\n")
    spec = {
        "bboxInputSize": bbox_size,
        "keypointInputSize": kp_size,
        "numKeypoints": num_keypoints,
        "hiddenDim": hidden_dim,
        "numClasses": num_classes,
        "bboxModel": bbox_path.name,
        "poseModel": posenet_path.name,
        "precision": args.precision,
        "sourceCheckpoint": checkpoint_path.name,
        "sourceCheckpointSha256": sha256_of(checkpoint_path),
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    print("\nwrote")
    print(f"  {bbox_path}  ({bbox_mb:.1f} MB)")
    print(f"  {posenet_path}  ({posenet_mb:.1f} MB)")
    print(f"  {labels_path}  ({labels_path.stat().st_size} B)")
    print(f"  {spec_path}  ({spec_path.stat().st_size} B)")
    print(f"\ntotal {bbox_mb + posenet_mb:.1f} MB of Core ML weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
