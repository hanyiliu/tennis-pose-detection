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

import numpy as np
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
        # Split the flat index with INTEGER ops. Never convert idx itself: it reaches
        # h*w-1 = 16383 and fp16 is only exact on integers up to 2048, so a float divide
        # decodes most channels to the wrong pixel. Row/column are 0..127, exact in fp16.
        y_i = torch.div(idx, w, rounding_mode="floor")
        x_i = idx - y_i * w
        x = x_i.float() / (w - 1)
        y = y_i.float() / (h - 1)
        kps = torch.stack([x, y, vis], dim=-1)  # (B, K, 3)
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


def deployment_target(ct):
    """Pinned to the app's IPHONEOS_DEPLOYMENT_TARGET; never auto-selected."""
    target = getattr(ct.target, "iOS17", None)
    if target is None:
        raise SystemExit(
            f"coremltools {ct.__version__} does not expose ct.target.iOS17, which the "
            "iOS app's deployment target requires. Install a coremltools that does; do "
            "not substitute another target."
        )
    return target


def _input_vars(op):
    for value in op.inputs.values():
        for var in (value if isinstance(value, (list, tuple)) else [value]):
            yield var


def downstream_of_argmax(op, memo):
    """True if *op* transitively consumes the heatmap argmax, i.e. is decode or stage 3."""
    cached = memo.get(op.name)
    if cached is not None:
        return cached
    memo[op.name] = False  # cycle guard
    result = op.op_type == "reduce_argmax" or any(
        var.op is not None and downstream_of_argmax(var.op, memo) for var in _input_vars(op)
    )
    memo[op.name] = result
    return result


def fp16_except_decode_and_stage3(ct):
    """fp16 everywhere upstream of the argmax; fp32 from the argmax onwards.

    Stage 3 divides by ``vis.sum().clamp_min(1e-6)``, and visibility is the raw max logit,
    routinely negative (measured -40.6 on a black crop). The clamp then fires and ``center``
    reaches ~1e6 -- past the fp16 max of 65504, so ``centered_xy`` goes inf/nan into the
    classifier. The U-Net convolutions upstream carry every megabyte and stay fp16.
    """
    memo = {}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: not downstream_of_argmax(op, memo)
    )


def convert(ct, module: nn.Module, example: torch.Tensor, output_names, precision, target):
    traced = torch.jit.trace(module, example, strict=False)
    traced.eval()
    return ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=tuple(example.shape),
                scale=1 / 255.0,  # no bias: the pipeline applies no mean/std normalization
                color_layout=ct.colorlayout.RGB,
            )
        ],
        # fp32 outputs regardless of internal precision: the client reads plain Float32
        # MLMultiArrays, and keypoints/probs are too small to care about the byte count.
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=target,
    )


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
        help="fp16 keeps the convolutions in fp16 and the decode plus stage 3 in fp32; "
             "fp32 forces the whole graph to fp32 (default: fp16)",
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

    fp32 = args.precision == "fp32"
    bbox_precision = ct.precision.FLOAT32 if fp32 else ct.precision.FLOAT16
    # Stage 1 is pure convolution, so plain fp16 is safe there; the fused graph needs
    # the decode and stage 3 kept in fp32.
    posenet_precision = bbox_precision if fp32 else fp16_except_decode_and_stage3(ct)
    precision_label = "fp32" if fp32 else "fp16-unet-fp32-decode-and-stage3"
    target = deployment_target(ct)
    print(f"precision {precision_label}, minimum_deployment_target iOS17\n")

    # Both conversions run to completion before anything touches out_dir: a half-written
    # directory would pair a fresh stage 1 with a stale stage 2+3.
    print("converting TPDBBox ...")
    bbox_ml = convert(ct, bbox_model, bbox_example, ["bbox"], bbox_precision, target)

    print("converting TPDPoseNet (stages 2+3 fused) ...")
    try:
        posenet_ml = convert(
            ct, posenet, pose_example, ["probs", "keypoints"], posenet_precision, target
        )
    except Exception as exc:  # noqa: BLE001 - the fallback decision needs every failure
        print("\n" + "!" * 78, file=sys.stderr)
        print("FUSED STAGE 2+3 CONVERSION FAILED -- out dir left untouched.", file=sys.stderr)
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

    bbox_path = out_dir / "TPDBBox.mlpackage"
    posenet_path = out_dir / "TPDPoseNet.mlpackage"
    bbox_mb = save(bbox_ml, bbox_path)
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
        "precision": precision_label,
        "minimumDeploymentTarget": "iOS17",
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
