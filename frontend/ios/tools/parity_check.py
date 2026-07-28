#!/usr/bin/env python3
"""Assert the exported Core ML packages still agree with the torch pipeline.

Run it from anywhere; like ``export_coreml.py`` it locates the repo root by walking up
to ``models/pose_classification.py``. Inputs are synthesised from ``--seed``, so two runs
on the same machine compare the same pixels and no fixture files are needed.

Two comparison modes, reported separately -- do not quote one's numbers as the other's:

``same-crop``
    Identical 128x128 pixels into torch stage 2+3 and into ``TPDPoseNet``. Nothing but
    the converted graphs can differ, so this is the real model comparison and carries
    the tight gates.

``end-to-end``
    Full frame -> ``TPDBBox`` -> ``norm_bbox_to_xyxy_pixels`` -> crop -> letterbox ->
    stage 2+3, each stack using its OWN stage-1 bbox. Stage 1's fp16 error is sub-pixel
    (~3e-4 normalized) but the bbox is *rounded to integers*, so the crop rectangle
    flips by a pixel on a good fraction of frames and the two stacks then see different
    pixels. Gated here: the bbox, and the class decision **on the frames where both
    stacks rounded to the same rectangle** -- there the pixels really are identical, so a
    disagreement would be a real defect. The aggregate keypoint/probability rows include
    the rounding-affected frames and are reported as information, because they measure
    the rounding rather than the conversion.

Gates follow what this fp16 export can actually deliver, measured, not hoped for:

* Keypoints gate on **flat-argmax index equality**, never on bit-exact floats. The U-Net
  runs in fp16, so heatmap values a few 1e-4 apart collapse onto one fp16 value and a
  near-tied peak can land on the neighbouring pixel. Coordinate deltas are therefore
  only checked across channels whose index already agrees, where 1 fp32 ULP is the
  entire budget.
* Visibility gets its own, looser tolerance: it is the raw unnormalized max logit, not a
  probability, and drifts ~3x further than the class probabilities do.
* Class argmax agreement is the hard gate in both modes -- over every same-crop input,
  and over every end-to-end frame whose crop rectangle came out identical.

The first same-crop input is a black crop on purpose. It drives ``sum(vis)`` to about
-40, which makes stage 3's ``clamp_min(1e-6)`` fire and ``center`` reach ~3.5e7 -- past
the fp16 ceiling of 65504. That is what broke the all-fp16 export; the fp32 decode head
has to keep it finite and on the same class.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Reusing the exporter's PoseNet is deliberate: a re-implementation here could drift from
# the graph that was actually converted, and then parity would be measured against the
# wrong reference. Importing it also puts REPO_ROOT on sys.path.
from export_coreml import (  # noqa: E402
    DROPOUT,
    VISIBILITY_THRESHOLD,
    PoseNet,
    find_repo_root,
    square_size,
)

REPO_ROOT = find_repo_root(Path(__file__).resolve())

from models.bbox_detection import BBoxDetectionModel  # noqa: E402
from models.keypoint_detection import KeypointDetectionModel  # noqa: E402
from models.pose_classification import PoseClassificationModel  # noqa: E402
from preprocessing.image_preprocessing import convert_image_to_tensor  # noqa: E402
from preprocessing.pil_preprocessing import (  # noqa: E402
    crop_pil,
    letterbox_resize,
    norm_bbox_to_xyxy_pixels,
)

# --- gates -----------------------------------------------------------------------
# Per input, at least this many of the 18 channels must decode to the same flat heatmap
# index. Measured on the shipped export: 359/360 channels over 20 inputs, i.e. exactly
# one input at 17/18.
MIN_INDEX_MATCH_PER_INPUT = 17
# Only over index-matching channels, where the two stacks picked the same pixel: the
# coordinate is idx/127 in fp32 and a 1 ULP disagreement near 1.0 is 1.2e-7.
COORD_TOL = 1e-6
# Raw max logit, not a probability. Measured 1.0e-2.
VIS_TOL = 5e-2
# Measured 3.9e-4 at the default seed. The worst seen across the seeds tried so far is
# 1.9e-2 (seed 7, 20 inputs, same on CPU and ALL) -- close to this gate, and NOT caused by
# an argmax flip: that input matched 18/18 indices. It is visibility drift (1.3e-2 of raw
# logit) amplified by stage 3's visibility-weighted centre and spread features. Left at
# 2e-2 deliberately; widening it to buy headroom would stop the gate meaning anything.
PROB_TOL = 2e-2
# Normalized xywh out of stage 1. Measured 3.3e-4.
BBOX_TOL = 1e-2
# Frame sizes cycled through in end-to-end mode; non-square and both orientations so the
# letterbox padding is exercised on each axis.
FRAME_SIZES = ((640, 360), (720, 1280), (512, 512), (960, 540))


def structured_image(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """A low-frequency blob field plus grain -- closer to a video frame than white noise.

    White noise makes stage 1 emit near-degenerate boxes and stage 2 near-flat heatmaps,
    which flatters the comparison instead of testing it.
    """
    seed_grid = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    base = Image.fromarray(seed_grid).resize((width, height), Image.Resampling.BILINEAR)
    grain = rng.integers(-24, 25, size=(height, width, 3))
    return np.clip(np.asarray(base, dtype=np.int16) + grain, 0, 255).astype(np.uint8)


def make_crops(rng, size: int, count: int) -> list:
    """Deterministic 128x128 crops. Index 0 is the black stage-3 overflow probe."""
    images = [
        np.zeros((size, size, 3), dtype=np.uint8),
        np.full((size, size, 3), 255, dtype=np.uint8),
    ]
    while len(images) < count:
        images.append(structured_image(rng, size, size))
    return [Image.fromarray(a, "RGB") for a in images[:count]]


def make_frames(rng, count: int) -> list:
    return [
        Image.fromarray(structured_image(rng, *FRAME_SIZES[i % len(FRAME_SIZES)]), "RGB")
        for i in range(count)
    ]


class Reference:
    """The torch side, built from the checkpoint exactly the way export_coreml.py does."""

    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        self.bbox_size = square_size(ckpt, "bbox")
        self.kp_size = square_size(ckpt, "keypoint")
        pose_state = ckpt["pose_model_state"]
        kp_state = ckpt["keypoint_model_state"]
        self.hidden_dim = int(pose_state["fc1.weight"].shape[0])
        self.num_classes = int(pose_state["out.weight"].shape[0])
        self.num_keypoints = int(kp_state["extraction_conv1.weight"].shape[0])
        self.labels = [str(n).strip().lower().replace(" ", "_") for n in ckpt["label_names"]]

        self.bbox_model = BBoxDetectionModel()
        self.bbox_model.load_state_dict(ckpt["bbox_model_state"])
        self.bbox_model.eval()

        keypoint_model = KeypointDetectionModel(num_keypoints=self.num_keypoints)
        keypoint_model.load_state_dict(kp_state)
        keypoint_model.eval()

        pose_model = PoseClassificationModel(
            num_keypoints=self.num_keypoints,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=DROPOUT,
            visibility_threshold=VISIBILITY_THRESHOLD,
        )
        pose_model.load_state_dict(pose_state)
        pose_model.eval()
        self.posenet = PoseNet(keypoint_model, pose_model).eval()

    @torch.no_grad()
    def bbox(self, resized_frame: Image.Image) -> np.ndarray:
        tensor = convert_image_to_tensor(resized_frame).unsqueeze(0)
        return self.bbox_model(tensor).squeeze(0).numpy().astype(np.float64)

    @torch.no_grad()
    def pose(self, crop: Image.Image):
        probs, kps = self.posenet(convert_image_to_tensor(crop).unsqueeze(0))
        return probs.squeeze(0).numpy().astype(np.float64), kps.squeeze(0).numpy().astype(np.float64)


class CoreML:
    """The Core ML side. Images go in as PIL: the packages declare ct.ImageType."""

    def __init__(self, models_dir: Path, compute_units: str):
        import coremltools as ct

        units = ct.ComputeUnit.CPU_ONLY if compute_units == "cpu" else ct.ComputeUnit.ALL
        self.units = units.name
        self.bbox_model = ct.models.MLModel(str(models_dir / "TPDBBox.mlpackage"), compute_units=units)
        self.pose_model = ct.models.MLModel(str(models_dir / "TPDPoseNet.mlpackage"), compute_units=units)

    def bbox(self, resized_frame: Image.Image) -> np.ndarray:
        out = self.bbox_model.predict({"image": resized_frame})["bbox"]
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def pose(self, crop: Image.Image):
        out = self.pose_model.predict({"image": crop})
        probs = np.asarray(out["probs"], dtype=np.float64).reshape(-1)
        kps = np.asarray(out["keypoints"], dtype=np.float64).reshape(-1, 3)
        return probs, kps


def flat_index(kps: np.ndarray, size: int) -> np.ndarray:
    """Rebuild the flat heatmap argmax index the decode produced from its x,y output."""
    xy = np.rint(kps[:, :2] * (size - 1)).astype(np.int64)
    return xy[:, 1] * size + xy[:, 0]


class Stats:
    """Running worst-case deltas over every input in one mode."""

    def __init__(self):
        self.inputs = 0
        self.index_match = 0
        self.index_total = 0
        self.worst_index_match = None
        self.worst_index_input = -1
        self.coord_matched = 0.0
        self.coord_all = 0.0
        self.vis = 0.0
        self.prob = 0.0
        self.class_agree = 0
        # Same, restricted to inputs where both stacks genuinely saw the same pixels.
        self.strict_inputs = 0
        self.strict_agree = 0
        self.bbox = 0.0
        self.rect_equal = 0
        self.rect_total = 0
        self.nonfinite = []
        self.probe = None

    def note_nonfinite(self, tag: str, **arrays):
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                self.nonfinite.append(f"{tag}[{name}]")

    def add_bbox(self, index: int, ref: np.ndarray, ml: np.ndarray):
        self.note_nonfinite(f"input {index} bbox", torch=ref, coreml=ml)
        self.bbox = max(self.bbox, float(np.max(np.abs(ref - ml))))

    def add_pose(self, index: int, ref, ml, size: int, verbose: bool, same_pixels: bool = True):
        (ref_probs, ref_kps), (ml_probs, ml_kps) = ref, ml
        self.note_nonfinite(f"input {index} pose", probs=ref_probs, keypoints=ref_kps)
        self.note_nonfinite(f"input {index} pose", probs=ml_probs, keypoints=ml_kps)

        same = flat_index(ref_kps, size) == flat_index(ml_kps, size)
        matched = int(same.sum())
        self.inputs += 1
        self.index_match += matched
        self.index_total += same.size
        if self.worst_index_match is None or matched < self.worst_index_match:
            self.worst_index_match, self.worst_index_input = matched, index

        coord_delta = np.abs(ref_kps[:, :2] - ml_kps[:, :2])
        self.coord_all = max(self.coord_all, float(coord_delta.max()))
        if matched:
            self.coord_matched = max(self.coord_matched, float(coord_delta[same].max()))
        self.vis = max(self.vis, float(np.max(np.abs(ref_kps[:, 2] - ml_kps[:, 2]))))
        prob_delta = float(np.max(np.abs(ref_probs - ml_probs)))
        self.prob = max(self.prob, prob_delta)
        ref_class, ml_class = int(np.argmax(ref_probs)), int(np.argmax(ml_probs))
        self.class_agree += int(ref_class == ml_class)
        if same_pixels:
            self.strict_inputs += 1
            self.strict_agree += int(ref_class == ml_class)
        if verbose:
            print(
                f"    input {index:>3}  idx {matched:>2}/{same.size}  "
                f"|dcoord| {coord_delta.max():.2e}  "
                f"|dvis| {np.max(np.abs(ref_kps[:, 2] - ml_kps[:, 2])):.2e}  "
                f"|dprob| {prob_delta:.2e}  class {ref_class}/{ml_class}  "
                f"sum(vis) {ref_kps[:, 2].sum():+.1f}"
            )
        return ref_class, ml_class


class Row:
    def __init__(self, name, measured, limit, ok=None):
        self.name, self.measured, self.limit, self.ok = name, measured, limit, ok

    def status(self) -> str:
        return "info" if self.ok is None else ("PASS" if self.ok else "FAIL")


def pose_rows(stats: Stats, num_keypoints: int, gated: bool) -> list:
    """Rows shared by both modes. *gated* mode asserts; otherwise it only reports."""
    def gate(value):
        return value if gated else None

    worst = stats.worst_index_match if stats.worst_index_match is not None else 0
    percent = 100.0 * stats.index_match / max(stats.index_total, 1)
    return [
        Row("keypoint argmax index, worst input",
            f"{worst}/{num_keypoints} (input {stats.worst_index_input})",
            f">= {MIN_INDEX_MATCH_PER_INPUT}/{num_keypoints}",
            gate(worst >= MIN_INDEX_MATCH_PER_INPUT)),
        Row("keypoint argmax index, all channels",
            f"{stats.index_match}/{stats.index_total} ({percent:.1f}%)", "-"),
        Row("keypoint coord max|d|, matched idx", f"{stats.coord_matched:.3e}",
            f"< {COORD_TOL:.0e}", gate(stats.coord_matched < COORD_TOL)),
        Row("keypoint coord max|d|, all channels", f"{stats.coord_all:.3e}", "-"),
        Row("visibility max|d| (raw logit)", f"{stats.vis:.3e}",
            f"< {VIS_TOL:.0e}", gate(stats.vis < VIS_TOL)),
        Row("class probability max|d|", f"{stats.prob:.3e}",
            f"< {PROB_TOL:.0e}", gate(stats.prob < PROB_TOL)),
        Row("class argmax agreement", f"{stats.class_agree}/{stats.inputs}",
            f"{stats.inputs}/{stats.inputs}" if gated else "-",
            (stats.class_agree == stats.inputs) if gated else None),
        Row("finite outputs (no NaN/Inf)", ", ".join(stats.nonfinite[:3]) or "clean",
            "clean", not stats.nonfinite),
    ]


def run_same_crop(reference: Reference, coreml: CoreML, crops, verbose: bool) -> Stats:
    stats = Stats()
    for index, crop in enumerate(crops):
        ref, ml = reference.pose(crop), coreml.pose(crop)
        classes = stats.add_pose(index, ref, ml, reference.kp_size, verbose)
        if index == 0:  # the black crop: stage-3 fp16 overflow probe
            stats.probe = (float(ref[1][:, 2].sum()), float(ml[1][:, 2].sum()), classes)
    return stats


def run_end_to_end(reference: Reference, coreml: CoreML, frames, verbose: bool) -> Stats:
    stats = Stats()
    size = reference.kp_size
    for index, frame in enumerate(frames):
        width, height = frame.size
        resized = frame.resize((reference.bbox_size, reference.bbox_size), Image.Resampling.BILINEAR)
        ref_bbox, ml_bbox = reference.bbox(resized), coreml.bbox(resized)
        stats.add_bbox(index, ref_bbox, ml_bbox)

        rects = []
        crops = []
        for bbox in (ref_bbox, ml_bbox):
            rect = norm_bbox_to_xyxy_pixels(torch.from_numpy(bbox), width, height)
            rects.append(rect)
            crops.append(letterbox_resize(crop_pil(frame, rect), size=(size, size)))
        same_rect = rects[0] == rects[1]
        stats.rect_total += 1
        stats.rect_equal += int(same_rect)
        if verbose and not same_rect:
            print(f"    input {index:>3}  crop rect torch {rects[0]} vs coreml {rects[1]}")
        stats.add_pose(
            index, reference.pose(crops[0]), coreml.pose(crops[1]), size, verbose, same_rect
        )
    return stats


def print_table(title: str, note: str, rows) -> bool:
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'check':<38}{'measured':>26}{'limit':>18}  status")
    print(f"  {'-' * 88}")
    for row in rows:
        print(f"  {row.name:<38}{row.measured:>26}{row.limit:>18}  {row.status()}")
    return all(row.ok is not False for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "exports" / "colab_e2e_best_2.pt",
        help="checkpoint the packages were exported from (default: exports/colab_e2e_best_2.pt)",
    )
    parser.add_argument(
        "--models-dir",
        "--models",  # the spelling `make parity` passes
        dest="models_dir",
        type=Path,
        default=REPO_ROOT / "frontend" / "ios" / "TPD" / "Models",
        help="directory holding TPDBBox.mlpackage and TPDPoseNet.mlpackage",
    )
    parser.add_argument("--num-inputs", type=int, default=12, help="inputs per mode (default: 12)")
    parser.add_argument("--seed", type=int, default=0, help="input generator seed (default: 0)")
    parser.add_argument(
        "--compute-units",
        choices=("cpu", "all"),
        default="cpu",
        help="Core ML compute units: cpu is reproducible, all lets the ANE/GPU in (default: cpu)",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-input deltas")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_inputs < 2:
        raise SystemExit("--num-inputs must be at least 2 (input 0 is the black overflow probe)")

    checkpoint = args.checkpoint.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    for name in ("TPDBBox.mlpackage", "TPDPoseNet.mlpackage"):
        if not (models_dir / name).exists():
            raise SystemExit(f"missing {models_dir / name} -- run `make export` from frontend/ios")

    print(f"repo root   {REPO_ROOT}")
    print(f"checkpoint  {checkpoint}")
    print(f"models dir  {models_dir}")

    reference = Reference(checkpoint)
    coreml = CoreML(models_dir, args.compute_units)
    print(
        f"models      {reference.bbox_size}px bbox, {reference.kp_size}px keypoint, "
        f"{reference.num_keypoints} keypoints, {reference.num_classes} classes "
        f"{reference.labels}"
    )
    print(f"inputs      {args.num_inputs} per mode, seed {args.seed}, compute units {coreml.units}")

    rng = np.random.default_rng(args.seed)
    crops = make_crops(rng, reference.kp_size, args.num_inputs)
    frames = make_frames(rng, args.num_inputs)

    if args.verbose:
        print("\nsame-crop per-input:")
    same = run_same_crop(reference, coreml, crops, args.verbose)
    if args.verbose:
        print("\nend-to-end per-input:")
    e2e = run_end_to_end(reference, coreml, frames, args.verbose)

    torch_sum, coreml_sum, (ref_class, ml_class) = same.probe
    rows = pose_rows(same, reference.num_keypoints, gated=True)
    rows.append(
        Row("black-crop stage-3 overflow probe",
            f"sum(vis) {torch_sum:+.1f}/{coreml_sum:+.1f} class {ref_class}/{ml_class}",
            "same class", ref_class == ml_class))
    ok = print_table(
        f"SAME-CROP  stages 2+3, identical pixels into both stacks  ({same.inputs} inputs)",
        "the real model comparison: only the converted graphs can differ here",
        rows,
    )

    e2e_rows = [
        Row("stage-1 bbox max|d| (normalized)", f"{e2e.bbox:.3e}",
            f"< {BBOX_TOL:.0e}", e2e.bbox < BBOX_TOL),
        Row("identical integer crop rects", f"{e2e.rect_equal}/{e2e.rect_total}", "-"),
        # The only end-to-end frames on which a class disagreement would mean the
        # conversion is wrong rather than the rounding. Zero such frames fails: it would
        # mean nothing was actually compared.
        Row("class argmax, identical-rect frames", f"{e2e.strict_agree}/{e2e.strict_inputs}",
            f"{e2e.strict_inputs}/{e2e.strict_inputs}, >0",
            e2e.strict_inputs > 0 and e2e.strict_agree == e2e.strict_inputs),
    ] + pose_rows(e2e, reference.num_keypoints, gated=False)
    ok = print_table(
        f"END-TO-END  frame -> bbox -> crop -> stages 2+3  ({e2e.inputs} inputs)",
        "each stack rounds its own crop rect, so the ungated rows measure that rounding",
        e2e_rows,
    ) and ok

    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
