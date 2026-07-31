#!/usr/bin/env python3
"""Assert the exported Core ML packages still agree with the torch pipeline.

Run it from anywhere; like ``export_coreml.py`` it locates the repo root by walking up to
``models/pose_classification.py``. Inputs are synthesised, so no fixture files are needed,
and several seeds run per invocation so one unlucky seed cannot define the verdict. Two
comparison modes, reported separately -- do not quote one's numbers as the other's:

``same-crop``
    Identical 128x128 pixels into torch stage 2+3 and into ``TPDPoseNet``. Nothing but the
    converted graphs can differ, so this carries the tight keypoint gates.

``end-to-end``
    Full frame -> ``TPDBBox`` -> ``norm_bbox_to_xyxy_pixels`` -> crop -> letterbox -> stage
    2+3, each stack using its OWN stage-1 bbox. Stage 1's fp16 error is sub-pixel (~3e-4
    normalized) but the bbox is *rounded to integers*, so the crop rectangle flips by a
    pixel on a good fraction of frames and the two stacks then see different pixels. The
    bbox and the class decision are gated, and so are the keypoints -- but only over the
    frames whose two rounded rects came out **identical**, roughly 70% of them. There the
    pixels really are the same, the rounding cannot be the culprit, and a relocated
    resolvable peak is as much a defect as it is same-crop. The frames the rounding split
    are accumulated separately and printed as ``[info] split-rect`` rows, gated by nothing:
    over 40 seeds they carry 197 relocated peaks reaching 24x the resolvability floor,
    which is the size of the noise that would have to be tolerated to gate them together.

**Why the keypoint gate is prominence-aware.** A heatmap channel only *has* a location if
its peak stands out. When the best value more than ``NEIGHBOR_RADIUS`` pixels away sits
within fp16 noise of the peak, which pixel wins is decided by rounding and the argmax can
move the full width of the map off a 1e-3 difference -- a property of the input, a joint the
crop carries no signal for, not a conversion defect. Gating on those makes red mean nothing,
so every channel is classified against a noise floor built from the actual fp16 ULP at its
own peak (``numpy.spacing`` on float16): **resolvable** channels are gated, **ambiguous**
ones never are but are always counted, because a jump in that count is itself a signal.

The class gate is conditional for a mechanical reason: ``_build_pose_features`` mixes a
visibility-weighted centre and a masked min/spread over all 18 keypoints, so ONE relocated
channel perturbs 76 of the 130 stage-3 features. An input whose keypoints did not fully
agree cannot grade the classifier and is excluded -- but a run where too few inputs are
gradeable FAILS as vacuous rather than reporting a green pass.

The two comparison ResNet-18s get an ordinary tolerance table each, plus a shared **contract traps**
table -- the interesting one, since what differs across the three models is not numerics but
*contracts*. It asserts measurements against registry fields, for EVERY model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Reusing the exporter's PoseNet is deliberate: a re-implementation here could drift from
# the graph that was actually converted, and then parity would be measured against the
# wrong reference. Importing it also puts REPO_ROOT on sys.path.
from export_coreml import (  # noqa: E402
    DROPOUT, VISIBILITY_THRESHOLD, PoseNet, find_repo_root, square_size)
from export_comparison import SPECS, Normalized, build  # noqa: E402

REPO_ROOT = find_repo_root(Path(__file__).resolve())

from models.bbox_detection import BBoxDetectionModel  # noqa: E402
from models.keypoint_detection import KeypointDetectionModel  # noqa: E402
from models.pose_classification import PoseClassificationModel  # noqa: E402
from preprocessing.image_preprocessing import convert_image_to_tensor  # noqa: E402
from preprocessing.pil_preprocessing import (  # noqa: E402
    crop_pil, letterbox_resize, norm_bbox_to_xyxy_pixels)

# --- gates -----------------------------------------------------------------------
# Every constant is a measured worst case times a safety factor, and every gate prints its
# measurement beside its limit so drift shows before it fails. Calibration sweep: 40 seeds
# x 8 inputs x {cpu, all} against the shipped export, 1280 inputs per mode.
# A peak is a location only if it beats everything outside its own 3x3 by more than fp16
# noise; inside that 3x3 a one-pixel slide is routine even for a sharp peak. That lead must
# clear the larger of RESOLVE_ULPS fp16 ULPs at the channel's own peak and HEATMAP_NOISE,
# the absolute drift the converted U-Net puts on a heatmap value (the ULP term vanishes near
# zero, where that drift is all there is). Of 13824 swept same-crop channels, 79 moved
# further than 1px and the most prominent reached 0.12 of this floor -- an 8x margin -- with
# 76.6% of channels still resolvable and gated.
NEIGHBOR_RADIUS = 1
RESOLVE_ULPS = 16.0
HEATMAP_NOISE = 0.05
# Resolvable channels whose peak moved further than NEIGHBOR_RADIUS. Measured 0.
MAX_RELOCATED = 0
# Same-pixel channels: the coordinate is idx/(size-1) in fp32 on both sides and 1 ULP near
# 1.0 is 1.2e-7. Measured 6.0e-08.
COORD_TOL = 1e-6
# Raw max logit, not a probability, so it drifts further than the class probabilities do.
# Measured 4.1e-02 same-crop, 5.2e-02 over end-to-end's identical-rect frames.
VIS_TOL = 1.5e-1
# An input grades the classifier only when torch's OWN top-2 gap clears this, so drift
# cannot flip a gradeable decision and any flip is a real defect. Kept a separate constant
# from PROB_TOL: which inputs are *admissible* is a property of the softmax's steepness,
# while how far the converted graph may move a probability is a property of the conversion,
# and collapsing the two makes tightening one silently re-pick the population the other is
# measured over.
CLASS_TIE_GAP = 3e-1
# Max probability drift over gradeable inputs -- where both stacks fed stage 3 keypoints
# agreeing pixel for pixel, so the residue is entirely visibility drift amplified by the
# classifier. Two independent bounds pin this, and they meet:
#   from below -- 947 gradeable inputs over 40 seeds x {cpu, all} x {same-crop, end-to-end}
#     put the worst at 9.559e-02 (median 3e-11, p99 1.3e-02: nearly every input is exact and
#     a handful sit on a steep part of the softmax, turning 1e-02 of raw logit into 1e-01 of
#     probability). 1.5x that worst is 1.434e-01, so 1.5e-01 is the next clean value up.
#   from above -- an argmax flip needs the top two to swap, i.e. the drifts on them to sum
#     past CLASS_TIE_GAP. Both are bounded by this constant, so PROB_TOL <= CLASS_TIE_GAP/2
#     = 1.5e-01 is exactly what makes "gradeable inputs never flip" hold arithmetically.
# It was 3e-01, whose real detection floor measured +0.2919, so an injected +0.20 passed the
# whole gate. At 1.5e-01 the floor is PROB_TOL minus whatever drift already sits on the class
# the regression lands on, which makes it TARGET-DEPENDENT rather than a single number:
# independently measured at +0.142 onto the runner-up and +0.1438 onto the argmax, i.e. the
# range ~0.142-0.144 on the default config (a 40-seed sweep catches +0.140).
#
# TWO residual blind spots, both real, stated plainly so nobody reads this gate as total:
#   1. On GRADEABLE inputs a stage-3-only regression under ~0.14 that never moves an argmax is
#      invisible. Shrinking that means lowering CLASS_TIE_GAP too, which costs gradeable
#      inputs, so it is documented rather than pushed.
#   2. On inputs the gate EXCLUDES from grading, the probability blind spot is UNBOUNDED in
#      magnitude, not ~0.14 -- nothing about their probabilities is checked at all. The
#      defence is the gradeable-fraction floor below, which fails the run as vacuous when too
#      few inputs are admissible; it is not a bound on what an excluded input may do.
PROB_TOL = 1.5e-1
# Normalized xywh out of stage 1. Measured 3.1e-04.
BBOX_TOL = 5e-3
# Too few gradeable inputs means nothing was compared: fail vacuous rather than green. Worst
# single run measured 8/12 same-crop, 4/12 end-to-end; the all-fp16 mis-export scores 0/12.
MIN_GRADEABLE_FRACTION = 0.15
# Cycled through in end-to-end mode; non-square and both orientations so the letterbox
# padding is exercised on each axis.
FRAME_SIZES = ((640, 360), (720, 1280), (512, 512), (960, 540))

# --- comparison ResNet-18 gates --------------------------------------------------
# Calibrated like the gates above, over 20 seeds x 8 inputs x {cpu, all} x 2 models = 640 runs:
# max|dlogit| 6.32e-03, max|dprob| 2.04e-03, gradeable 50-54%, ZERO argmax flips on any of the 640.
RESNET_LOGIT_TOL = 3e-2      # 4.7x the measured worst; fat on purpose, unseen seeds draw wider
RESNET_PROB_TOL = 1e-2       # 4.9x, and 1/30 of the tie gap below
RESNET_TIE_GAP = 3e-1   # as CLASS_TIE_GAP: too wide for RESNET_PROB_TOL on each to swap the top 2
RESNET_MIN_GRADEABLE_FRACTION = 0.15   # measured 50-54%; the floor only catches a vacuous run
# --- contract traps --------------------------------------------------------------
# Each trap asserts a MEASUREMENT against the REGISTRY FIELD the client branches on, per model:
# measuring a convention and never checking `output.type` leaves the registry free to lie. A logit
# vector has no reason to sum to 1, so a declared "logits" gates the MINIMUM |sum-1| over every
# input and both stacks -- one landing near 1 cannot carry it. Saw 1.132.
LOGIT_SUM_MARGIN = 5e-1
PROB_SUM_TOL = 1e-4      # a declared "probabilities" gates the MAXIMUM: one bad vector breaks it
NORM_TOL = 1e-4          # declared (mean, std) vs real preprocessing; the exporter's own limit
MIN_ORDER_SENSITIVE = 1  # predictions the wrong label list renames; 0 proves nothing. Saw 126/320


# --- synthetic inputs ------------------------------------------------------------
def draw_figure(draw: ImageDraw.ImageDraw, rng, box) -> None:
    """A crude articulated player inside *box* -- torso, head, two arms, two legs.

    Blob fields and white noise drive the U-Net into near-flat heatmaps, i.e. exactly the
    ambiguous channels the gate must discard. A limbed figure is what a player crop looks
    like: over 4320 channels it resolved 91.9% vs a blob field's 90.4% at a fixed cut.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    skin = tuple(int(v) for v in rng.integers(150, 225, size=3))
    shirt = tuple(int(v) for v in rng.integers(20, 235, size=3))
    cx = x0 + width * float(rng.uniform(0.42, 0.58))
    shoulder = y0 + height * float(rng.uniform(0.22, 0.30))
    hip = y0 + height * float(rng.uniform(0.50, 0.58))
    half = width * 0.18
    thick = max(2, int(round(width * 0.09)))

    def limb(sx, sy, angle, length, weight, fill):
        ex, ey = sx + length * np.cos(angle), sy + length * np.sin(angle)
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=weight)
        return ex, ey

    draw.polygon(
        [(cx - half, shoulder), (cx + half, shoulder),
         (cx + half * 0.8, hip), (cx - half * 0.8, hip)], fill=shirt)
    head = height * 0.09
    draw.ellipse([cx - head, shoulder - 2.3 * head, cx + head, shoulder - 0.3 * head], fill=skin)
    for side in (-1, 1):
        flip = np.pi if side < 0 else 0.0
        angle = float(rng.uniform(-1.0, 2.3)) + flip
        ex, ey = limb(cx + side * half, shoulder, angle, height * 0.20, thick, shirt)
        limb(ex, ey, angle + float(rng.uniform(-0.7, 0.7)), height * 0.18, max(2, thick - 1), skin)
    for side in (-1, 1):
        angle = np.pi / 2 + side * float(rng.uniform(0.05, 0.5))
        kx, ky = limb(cx + side * half * 0.6, hip, angle, height * 0.20, thick + 1, shirt)
        limb(kx, ky, np.pi / 2 + side * float(rng.uniform(-0.2, 0.35)), height * 0.19, thick, skin)


def scene(rng, width: int, height: int, box) -> np.ndarray:
    """Court-ish field -- one dominant hue, a couple of court lines, grain -- plus a figure."""
    field = np.zeros((height, width, 3), dtype=np.int16) + rng.integers(60, 150, size=3)
    field += np.linspace(-30, 30, height, dtype=np.int16)[:, None, None]
    image = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(2):
        y = int(rng.integers(0, height))
        draw.line([(0, y), (width, y)], fill=(235, 235, 235), width=2)
    draw_figure(draw, rng, box)
    grain = rng.integers(-10, 11, size=(height, width, 3))
    return np.clip(np.asarray(image, dtype=np.int16) + grain, 0, 255).astype(np.uint8)


def make_crops(rng, size: int, count: int, probes: bool) -> list:
    """``count`` 128x128 player crops. With *probes*, index 0/1 are black/white instead."""
    images = [np.zeros((size, size, 3), np.uint8),
              np.full((size, size, 3), 255, np.uint8)] if probes else []
    while len(images) < count:
        images.append(scene(rng, size, size, (size * 0.12, 0.0, size * 0.88, size * 0.99)))
    return [Image.fromarray(a) for a in images[:count]]


def make_frames(rng, count: int) -> list:
    """Full frames with the figure somewhere inside, so stage 1 has something to find."""
    frames = []
    for index in range(count):
        width, height = FRAME_SIZES[index % len(FRAME_SIZES)]
        left, top = width * float(rng.uniform(0.05, 0.55)), height * float(rng.uniform(0.05, 0.35))
        span = min(width - left, height - top) * float(rng.uniform(0.5, 0.9))
        frames.append(Image.fromarray(
            scene(rng, width, height, (left, top, left + span * 0.55, top + span))))
    return frames


# --- the two stacks --------------------------------------------------------------
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
        pose_model = PoseClassificationModel(
            num_keypoints=self.num_keypoints, num_classes=self.num_classes,
            hidden_dim=self.hidden_dim, dropout=DROPOUT,
            visibility_threshold=VISIBILITY_THRESHOLD)
        pose_model.load_state_dict(pose_state)
        self.posenet = PoseNet(keypoint_model, pose_model).eval()
        # A hook, not a second keypoint-model run: prominence must be measured on the exact
        # tensor the decode saw.
        self._heatmap = {}
        self.posenet.kp.register_forward_hook(
            lambda module, inputs, output: self._heatmap.__setitem__("hm", output))

    @torch.no_grad()
    def bbox(self, resized_frame: Image.Image) -> np.ndarray:
        tensor = convert_image_to_tensor(resized_frame).unsqueeze(0)
        return self.bbox_model(tensor).squeeze(0).numpy().astype(np.float64)

    @torch.no_grad()
    def pose(self, crop: Image.Image):
        probs, kps = self.posenet(convert_image_to_tensor(crop).unsqueeze(0))
        heatmap = self._heatmap["hm"].squeeze(0).numpy().astype(np.float64)
        return (probs.squeeze(0).numpy().astype(np.float64),
                kps.squeeze(0).numpy().astype(np.float64), heatmap)


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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


class Convention:
    """What an output vector IS, per model, so ``output.type`` is CHECKED rather than believed."""

    def __init__(self):
        self.sum_min, self.sum_max, self.value_min = np.inf, 0.0, np.inf

    def observe(self, *vectors) -> None:
        for vector in vectors:
            deviation = abs(float(vector.sum()) - 1.0)
            self.sum_min, self.sum_max = min(self.sum_min, deviation), max(self.sum_max, deviation)
            self.value_min = min(self.value_min, float(vector.min()))

    def row(self, model_id: str, declared: str):
        """Wrong either way fails: "logits" for a softmaxed head collapses ``sum_min`` to ~0,
        "probabilities" for a logit head blows ``sum_max``; an unrecognised type gates nothing."""
        probs = declared == "probabilities"
        seen, limit = ((self.sum_max, f"< {PROB_SUM_TOL:g}, >= 0") if probs
                       else (self.sum_min, f">= {LOGIT_SUM_MARGIN:g}, < 0"))
        held = ((seen < PROB_SUM_TOL and self.value_min >= 0.0) if probs
                else (seen >= LOGIT_SUM_MARGIN and self.value_min < 0.0))
        return Row(f"output.type '{declared}': {model_id}",
                   f"sum-1 {seen:.3g}, min {self.value_min:+.2f}", limit,
                   held and declared in ("logits", "probabilities"))


class ResNet:
    """One comparison ResNet-18: both stacks and the running worst cases, driven by its registry
    entry. The torch side comes from ``export_comparison.build``, the code that produced the
    package, so a registry drifting from that exporter fails here rather than in the app."""

    def __init__(self, entry: dict, models_dir: Path, compute_units: str):
        import coremltools as ct

        spec = next((s for s in SPECS if s["id"] == entry["id"]), None)
        if (spec is None or list(spec["labels"]) != list(entry["labels"])
                or spec["label_order"] != entry["labelOrder"]):
            raise SystemExit(f"registry entry '{entry['id']}' does not match any model "
                             "export_comparison.py builds -- one of the two is stale")
        self.id, self.size, self.labels = entry["id"], entry["input"]["size"], entry["labels"]
        self.output, self.declared = entry["output"]["name"], entry["output"]["type"]
        self.status, self.display = entry["labelOrder"]["status"], entry["displayName"]
        self.torch = Normalized(build(spec, REPO_ROOT / spec["checkpoint"])).eval()
        units = ct.ComputeUnit.CPU_ONLY if compute_units == "cpu" else ct.ComputeUnit.ALL
        self.coreml = ct.models.MLModel(str(models_dir / entry["packages"][0]), compute_units=units)
        self.inputs = self.gradeable = self.agree = self.order_sensitive = 0
        self.logit, self.prob, self.convention, self.predicted = 0.0, 0.0, Convention(), {}
        # The affine this graph really bakes in vs the one the REGISTRY's (mean, std) implies, so
        # perturbing `input.mean` fails here, not on device. Off the buffers: that IS the whole op.
        mean, std = entry["input"]["mean"], entry["input"]["std"]
        want = [1.0 / (255.0 * s) for s in std] + [-m / s for m, s in zip(mean, std)]
        baked = self.torch.scale.flatten().tolist() + self.torch.bias.flatten().tolist()
        self.norm = max(abs(a - b) for a, b in zip(want, baked))

    @torch.no_grad()
    def add(self, label: str, crop: Image.Image, shared_labels, verbose: bool):
        # Same pixels both sides: normalization is inside both graphs, not in the caller.
        raw = torch.from_numpy(np.array(crop)).permute(2, 0, 1).unsqueeze(0).float()
        ref = self.torch(raw).squeeze(0).numpy().astype(np.float64)
        ml = np.asarray(self.coreml.predict({"image": crop})[self.output], np.float64).reshape(-1)
        self.inputs += 1
        # No NaN row here: a non-finite logit makes this max NaN and `NaN < tol` is False anyway.
        self.logit = max(self.logit, float(np.max(np.abs(ref - ml))))
        self.convention.observe(ref, ml)   # both stacks: a Core ML-side lie counts too

        ref_probs, ml_probs = softmax(ref), softmax(ml)
        delta = float(np.max(np.abs(ref_probs - ml_probs)))
        ref_class, ml_class = int(np.argmax(ref)), int(np.argmax(ml))
        gradeable = float(np.diff(np.sort(ref_probs)[-2:])[0]) > RESNET_TIE_GAP
        if gradeable:
            self.gradeable += 1
            self.prob = max(self.prob, delta)
            self.agree += int(ref_class == ml_class)
        # The label trap on real predictions: this model's order names the winning index one
        # thing, the shared alphabetical order another.
        name = self.labels[ref_class]
        self.predicted[name] = self.predicted.get(name, 0) + 1
        self.order_sensitive += int(name != shared_labels[ref_class])
        if verbose:
            print(f"    {self.id} {label}  |dprob| {delta:.2e}  class {ref_class}/{ml_class}"
                  f" = {name}  {'gradeable' if gradeable else 'ungraded'}")

    def rows(self) -> list:
        floor = int(np.ceil(RESNET_MIN_GRADEABLE_FRACTION * self.inputs))
        return [
            Row("logits max|d| (fp16 conversion)", f"{self.logit:.3e}",
                f"< {RESNET_LOGIT_TOL:g}", self.logit < RESNET_LOGIT_TOL),
            Row("class probability max|d|, gradeable", f"{self.prob:.3e}",
                f"< {RESNET_PROB_TOL:g}", self.gradeable > 0 and self.prob < RESNET_PROB_TOL),
            Row("class argmax, gradeable inputs", f"{self.agree}/{self.gradeable}",
                f"{self.gradeable}/{self.gradeable}",
                self.gradeable > 0 and self.agree == self.gradeable),
            Row("gradeable inputs (torch top-2 gap)", f"{self.gradeable}/{self.inputs}",
                f">= {floor}", self.gradeable >= max(floor, 1)),
            Row("torch predictions, OWN label order",
                ", ".join(f"{n} x{c}" for n, c in sorted(self.predicted.items())) or "-", "-"),
        ]


def declared_drift(entry: dict) -> float:
    """The 3-stage pipeline preprocesses in ``ct.ImageType`` (scale 1/255) and its torch reference
    in ``ToTensor()``, so it has no baked affine to read -- but the registry's `(x/255 - mean)/std`
    describes it exactly when mean is 0 and std is 1, and that is a claim like any other."""
    return max([abs(m) for m in entry["input"]["mean"]]
               + [abs(s - 1.0) for s in entry["input"]["std"]])


def trap_rows(measured, resnets, shared) -> list:
    """Every model's DECLARED contract against what its tensors did: one row per model per contract,
    never one representative. *measured* is (id, output.type, Convention, norm) per model, pipeline
    included. Both halves used to be weaker than they read -- the order gate asserted only the FIRST
    deviating classifier, so a second would have retired the first one's gate, and no measurement
    met ``output.type``/``input.mean``, so a misdeclared registry stayed green."""
    rows = [convention.row(name, declared) for name, declared, convention, _ in measured]
    rows += [Row(f"input.mean/std: {name}", f"max|d| {norm:.1e}", f"< {NORM_TOL:g}",
                 norm < NORM_TOL) for name, _, _, norm in measured]
    for r in resnets:   # EVERY classifier, not just the first one whose order deviates
        spec = next(s for s in SPECS if s["id"] == r.id)
        # Where this model's indices land in the pipeline's order: 0,1,2,3 same, 1,0,2,3 the swap.
        mapping = ",".join(str(shared.index(n)) if n in shared else "?" for n in r.labels)
        rows.append(Row(f"labels [{r.status}]: {r.id}", f"pipeline idx {mapping}",
                        "perm, = exporter", list(r.labels) == list(spec["labels"])
                        and sorted(r.labels) == sorted(shared)))
    # The vacuity guard, kept: nothing deviating and nothing renamed makes the rows above trivially
    # true. `>= 1` and not a bare count -- print_table gates `is not False`, which a 0 passes.
    deviating = sum(list(r.labels) != list(shared) for r in resnets)
    sensitive, total = sum(r.order_sensitive for r in resnets), sum(r.inputs for r in resnets)
    return rows + [
        Row("classifier orders that deviate", f"{deviating}/{len(resnets)}", ">= 1", deviating >= 1),
        Row("predictions the wrong list would rename", f"{sensitive}/{total} inputs",
            f">= {MIN_ORDER_SENSITIVE}", sensitive >= MIN_ORDER_SENSITIVE)]


# --- prominence ------------------------------------------------------------------
def flat_index(kps: np.ndarray, size: int) -> np.ndarray:
    """Rebuild the flat heatmap argmax index the decode produced from its x,y output."""
    xy = np.rint(kps[:, :2] * (size - 1)).astype(np.int64)
    return xy[:, 1] * size + xy[:, 0]


def peak_prominence(heatmap: np.ndarray, radius: int) -> np.ndarray:
    """Per channel: the peak's lead over everything further than *radius* away, measured in
    units of the noise floor that lead must clear. Above 1.0 the location is resolvable.
    """
    channels, height, width = heatmap.shape
    flat = heatmap.reshape(channels, -1)
    index = flat.argmax(axis=1)
    peak = flat[np.arange(channels), index]
    rows, cols = index // width, index % width
    near_row = np.abs(np.arange(height)[None, :] - rows[:, None]) <= radius
    near_col = np.abs(np.arange(width)[None, :] - cols[:, None]) <= radius
    near = near_row[:, :, None] & near_col[:, None, :]
    runner_up = np.where(near, -np.inf, heatmap).reshape(channels, -1).max(axis=1)
    ulp = np.abs(np.spacing(peak.astype(np.float16))).astype(np.float64)
    return (peak - runner_up) / np.maximum(RESOLVE_ULPS * ulp, HEATMAP_NOISE)


class Peaks:
    """Keypoint aggregates over one subset of the inputs in a mode.

    Two are kept per mode and never mixed. ``identical`` holds the inputs both stacks saw
    the *same pixels* for -- every same-crop input, and in end-to-end the frames whose
    integer crop rects matched -- and is the only one anything is asserted against.
    ``split`` holds the end-to-end frames the rounding pulled apart, where a moved peak is
    a different crop rather than a different graph.
    """

    def __init__(self):
        self.inputs = self.channels = self.resolvable = self.relocated = 0
        self.worst_relocated = 0
        self.worst_prominence = 0.0      # prominence of the most prominent relocated channel
        self.worst_input = "-"
        self.coord = self.vis = 0.0

    def add(self, label, prominence, resolvable, relocated, moved, ref_kps, ml_kps):
        self.inputs += 1
        self.channels += prominence.size
        self.resolvable += int(resolvable.sum())
        gated_bad = int((resolvable & relocated).sum())
        if gated_bad > self.worst_relocated:
            self.worst_relocated, self.worst_input = gated_bad, label
        self.relocated += gated_bad
        if relocated.any():
            self.worst_prominence = max(self.worst_prominence, float(prominence[relocated].max()))
        exact = moved == 0
        if exact.any():
            delta = np.abs(ref_kps[:, :2] - ml_kps[:, :2])
            self.coord = max(self.coord, float(delta[exact].max()))
        self.vis = max(self.vis, float(np.max(np.abs(ref_kps[:, 2] - ml_kps[:, 2]))))
        return gated_bad


class Stats:
    """Running worst cases over every input in one mode."""

    def __init__(self):
        self.inputs = 0
        self.identical = Peaks()
        self.split = Peaks()
        self.prob = self.bbox = 0.0
        self.convention = Convention()
        self.gradeable = self.class_agree = 0
        self.rect_equal = self.rect_total = 0
        self.nonfinite = []
        self.probe = None

    def note_nonfinite(self, tag: str, **arrays):
        self.nonfinite += [f"{tag}[{n}]" for n, a in arrays.items() if not np.isfinite(a).all()]

    def add_bbox(self, label: str, ref: np.ndarray, ml: np.ndarray):
        self.note_nonfinite(f"{label} bbox", torch=ref, coreml=ml)
        self.bbox = max(self.bbox, float(np.max(np.abs(ref - ml))))

    def add_pose(self, label, ref, ml, size, verbose, same_pixels=True):
        (ref_probs, ref_kps, heatmap), (ml_probs, ml_kps) = ref, ml
        self.note_nonfinite(f"{label} pose", probs=ref_probs, keypoints=ref_kps)
        self.note_nonfinite(f"{label} pose", probs=ml_probs, keypoints=ml_kps)

        prominence = peak_prominence(heatmap, NEIGHBOR_RADIUS)
        resolvable = prominence > 1.0
        ref_index, ml_index = flat_index(ref_kps, size), flat_index(ml_kps, size)
        moved = np.maximum(np.abs(ref_index // size - ml_index // size),
                           np.abs(ref_index % size - ml_index % size))
        relocated = moved > NEIGHBOR_RADIUS

        self.inputs += 1
        # The one branch that decides what is asserted and what is only counted.
        bucket = self.identical if same_pixels else self.split
        gated_bad = bucket.add(label, prominence, resolvable, relocated, moved, ref_kps, ml_kps)

        # Stage 3 mixes every keypoint into 76 of its 130 features, so only an input whose
        # keypoints fully agreed can grade the classifier -- and only when torch's own
        # decision is not itself a near tie.
        ordered = np.sort(ref_probs)
        gradeable = (same_pixels and not relocated.any()
                     and (ordered[-1] - ordered[-2]) > CLASS_TIE_GAP)
        prob_delta = float(np.max(np.abs(ref_probs - ml_probs)))
        self.convention.observe(ref_probs, ml_probs)   # stage 3 softmaxed, so these sum to 1
        ref_class, ml_class = int(np.argmax(ref_probs)), int(np.argmax(ml_probs))
        if gradeable:
            self.gradeable += 1
            self.prob = max(self.prob, prob_delta)
            self.class_agree += int(ref_class == ml_class)
        if verbose:
            print(f"    {label:>12}  resolvable {int(resolvable.sum()):>2}/{prominence.size}"
                  f"  relocated {gated_bad}  |dprob| {prob_delta:.2e}"
                  f"  class {ref_class}/{ml_class}  {'gradeable' if gradeable else 'ungraded'}")
        return ref_class, ml_class


class Row:
    def __init__(self, name, measured, limit, ok=None):
        self.name, self.measured, self.limit, self.ok = name, measured, limit, ok


def pose_rows(stats: Stats, split_rows: bool) -> list:
    """Rows shared by both modes.

    Everything here is asserted against ``stats.identical`` -- the inputs both stacks saw
    the same pixels for. Same-crop mode is entirely that by construction; end-to-end mode
    is the frames whose two integer crop rects came out equal, and gating those is the
    point: only the frames the rounding *split* have an innocent explanation for a moved
    peak, and they are the only ones left ungated. *split_rows* appends them as clearly
    labelled info, so the rounding's size stays visible instead of being folded into a
    number something depends on.
    """
    gated = stats.identical
    resolvable_pct = 100.0 * gated.resolvable / max(gated.channels, 1)
    floor = int(np.ceil(MIN_GRADEABLE_FRACTION * stats.inputs))
    rows = [
        Row("resolvable peaks relocated > 1px",
            f"{gated.relocated}/{gated.resolvable} (worst {gated.worst_input})",
            f"<= {MAX_RELOCATED}",
            # Zero resolvable channels would satisfy "<= 0" while comparing nothing.
            gated.resolvable > 0 and gated.relocated <= MAX_RELOCATED),
        Row("closest call: prominence of a moved peak",
            f"{gated.worst_prominence:.2f}x noise floor", "resolvable > 1.00"),
        Row("resolvable channels (input quality)",
            f"{gated.resolvable}/{gated.channels} ({resolvable_pct:.1f}%)", "-"),
        Row("ambiguous channels (ungated, watch it)",
            f"{gated.channels - gated.resolvable}/{gated.channels}", "-"),
        Row("keypoint coord max|d|, same-pixel channels", f"{gated.coord:.3e}",
            f"< {COORD_TOL:g}", gated.coord < COORD_TOL),
        Row("visibility max|d| (raw logit)", f"{gated.vis:.3e}",
            f"< {VIS_TOL:g}", gated.vis < VIS_TOL),
        Row("gradeable inputs (keypoints agreed)",
            f"{stats.gradeable}/{stats.inputs}", f">= {floor}",
            stats.gradeable >= max(floor, 1)),
        # Both would read a vacuous PASS on zero gradeable inputs, so they carry the same
        # >0 requirement the gradeable row does rather than pass on no evidence.
        Row("class probability max|d|, gradeable", f"{stats.prob:.3e}",
            f"< {PROB_TOL:g}", stats.gradeable > 0 and stats.prob < PROB_TOL),
        Row("class argmax, gradeable inputs", f"{stats.class_agree}/{stats.gradeable}",
            f"{stats.gradeable}/{stats.gradeable}",
            stats.gradeable > 0 and stats.class_agree == stats.gradeable),
        Row("finite outputs (no NaN/Inf)", ", ".join(stats.nonfinite[:3]) or "clean",
            "clean", not stats.nonfinite),
    ]
    if split_rows:
        split = stats.split
        rows += [
            Row("[info] split-rect frames (own crop each)",
                f"{split.inputs}/{stats.inputs} frames, {split.channels} channels", "-"),
            Row("[info] split-rect peaks relocated", f"{split.relocated}/{split.resolvable}"
                f" (worst {split.worst_input})", "-"),
            Row("[info] split-rect moved-peak prominence",
                f"{split.worst_prominence:.2f}x noise floor", "-"),
            Row("[info] split-rect visibility max|d|", f"{split.vis:.3e}", "-"),
        ]
    return rows


def run_same_crop(reference: Reference, coreml: CoreML, crops, label: str, stats, verbose):
    for index, crop in enumerate(crops):
        ref, ml = reference.pose(crop), coreml.pose(crop)
        classes = stats.add_pose(f"{label}:{index}", ref, ml, reference.kp_size, verbose)
        if stats.probe is None:  # the black crop: stage-3 fp16 overflow probe
            stats.probe = (float(ref[1][:, 2].sum()), float(ml[1][:, 2].sum()), classes)


def run_end_to_end(reference: Reference, coreml: CoreML, frames, label: str, stats, verbose):
    size = reference.kp_size
    for index, frame in enumerate(frames):
        width, height = frame.size
        resized = frame.resize((reference.bbox_size, reference.bbox_size), Image.Resampling.BILINEAR)
        ref_bbox, ml_bbox = reference.bbox(resized), coreml.bbox(resized)
        stats.add_bbox(f"{label}:{index}", ref_bbox, ml_bbox)

        rects = [norm_bbox_to_xyxy_pixels(torch.from_numpy(b), width, height)
                 for b in (ref_bbox, ml_bbox)]
        crops = [letterbox_resize(crop_pil(frame, r), size=(size, size)) for r in rects]
        same_rect = rects[0] == rects[1]
        stats.rect_total += 1
        stats.rect_equal += int(same_rect)
        stats.add_pose(f"{label}:{index}", reference.pose(crops[0]), coreml.pose(crops[1]),
                       size, verbose, same_rect)


def print_table(title: str, note: str, rows) -> bool:
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'check':<42}{'measured':>26}{'limit':>18}  status")
    print(f"  {'-' * 92}")
    for row in rows:
        status = "info" if row.ok is None else ("PASS" if row.ok else "FAIL")
        print(f"  {row.name:<42}{row.measured:>26}{row.limit:>18}  {status}")
    return all(row.ok is not False for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "exports" / "colab_e2e_best_2.pt",
        help="checkpoint the packages were exported from (default: exports/colab_e2e_best_2.pt)")
    parser.add_argument(
        "--models-dir", "--models",  # --models is the spelling `make parity` passes
        dest="models_dir", type=Path,
        default=REPO_ROOT / "frontend" / "ios" / "TPD" / "Models",
        help="directory holding TPDBBox.mlpackage and TPDPoseNet.mlpackage")
    parser.add_argument("--num-inputs", type=int, default=8, help="inputs per seed per mode")
    parser.add_argument("--seed", type=int, default=0, help="first input generator seed")
    parser.add_argument(
        "--num-seeds", type=int, default=4,
        help="consecutive seeds from --seed, so one unlucky seed cannot decide the verdict")
    parser.add_argument(
        "--compute-units", choices=("cpu", "all"), default="cpu",
        help="Core ML compute units: cpu is reproducible, all lets the ANE/GPU in (default: cpu)")
    parser.add_argument("--verbose", action="store_true", help="print per-input deltas")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_inputs < 2 or args.num_seeds < 1:
        raise SystemExit("--num-inputs must be >= 2 (0/1 are the black/white probes), "
                         "--num-seeds >= 1")

    checkpoint = args.checkpoint.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    registry_path = models_dir / "TPDModelRegistry.json"
    if not registry_path.is_file():
        raise SystemExit(f"missing {registry_path} -- run `make export` from frontend/ios")
    registry = json.loads(registry_path.read_text())
    missing = [n for e in registry["models"] for n in e["packages"] if not (models_dir / n).exists()]
    if missing:
        raise SystemExit(f"missing {missing} -- run `make export` from frontend/ios")

    print(f"repo root   {REPO_ROOT}\ncheckpoint  {checkpoint}\nmodels dir  {models_dir}")
    reference = Reference(checkpoint)
    coreml = CoreML(models_dir, args.compute_units)
    pipeline = next(e for e in registry["models"] if e["kind"] == "pipeline")
    resnets = [ResNet(e, models_dir, args.compute_units)
               for e in registry["models"] if e["kind"] == "classifier"]
    if not resnets:
        raise SystemExit("registry lists no classifier models -- nothing to compare")
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    print(f"models      {reference.bbox_size}px bbox, {reference.kp_size}px keypoint, "
          f"{reference.num_keypoints} keypoints, {reference.num_classes} classes "
          f"{reference.labels}")
    print("comparison  " + ", ".join(f"{r.id} @{r.size}px" for r in resnets))
    print(f"inputs      {args.num_inputs} per seed per mode, seeds {seeds}, "
          f"compute units {coreml.units}")

    same, e2e = Stats(), Stats()
    # One 3-stage convention over both runs, and the resnets' scenes are drawn at the larger size.
    e2e.convention, biggest = same.convention, max(r.size for r in resnets)
    for position, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        crops = make_crops(rng, reference.kp_size, args.num_inputs, probes=position == 0)
        frames = make_frames(rng, args.num_inputs)
        run_same_crop(reference, coreml, crops, f"same s{seed}", same, args.verbose)
        run_end_to_end(reference, coreml, frames, f"e2e  s{seed}", e2e, args.verbose)
        # One scene per input, resampled per model, so the label trap compares like with like.
        for index, big in enumerate(make_crops(rng, biggest, args.num_inputs, position == 0)):
            for r in resnets:
                crop = big if r.size == biggest else big.resize((r.size,) * 2,
                                                                Image.Resampling.BILINEAR)
                r.add(f"s{seed}:{index}", crop, pipeline["labels"], args.verbose)

    torch_sum, coreml_sum, (ref_class, ml_class) = same.probe
    rows = pose_rows(same, split_rows=False)
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
            f"< {BBOX_TOL:g}", e2e.bbox < BBOX_TOL),
        Row("identical integer crop rects", f"{e2e.rect_equal}/{e2e.rect_total}", "-"),
    ] + pose_rows(e2e, split_rows=True)
    ok = print_table(
        f"END-TO-END  frame -> bbox -> crop -> stages 2+3  ({e2e.inputs} inputs)",
        "gated over the identical-rect frames only; [info] rows are the rounding's own noise",
        e2e_rows,
    ) and ok

    for r in resnets:
        ok = print_table(f"{r.id.upper()}  {r.size}px RGB -> {r.declared} ({r.inputs} inputs)",
                         f"{r.display}; class order {r.labels} [{r.status}]", r.rows()) and ok

    # The 3-stage model joins on equal terms -- same tuple, same rows -- so no model is exempt.
    measured = [(pipeline["id"], pipeline["output"]["type"], same.convention,
                 declared_drift(pipeline))] + [(r.id, r.declared, r.convention, r.norm)
                                               for r in resnets]
    ok = print_table("CONTRACT TRAPS  measured behaviour vs the registry fields the client uses",
                     "output.type, input.mean/std and labels, for every model the app can select",
                     trap_rows(measured, resnets, pipeline["labels"])) and ok

    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
