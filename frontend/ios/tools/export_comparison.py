#!/usr/bin/env python3
"""Export the two comparison ResNet-18 classifiers to Core ML, plus the model registry.

    TPDResNetKaggle160.mlpackage   image (1,3,160,160) 0-255 RGB -> logits (1,4)
    TPDResNetPose256.mlpackage     image (1,3,256,256) 0-255 RGB -> logits (1,4)
    TPDModelRegistry.json          all three shipped models, each self-describing

Both checkpoints are torchvision-shaped ResNet-18s (122 state-dict keys), so one builder hosts both
and ``timm`` is never needed. Three things differ per model, none may be guessed, each is silent
when wrong: output convention (logits here, already-softmaxed there), class order, and how well that
order is KNOWN -- one of the three is a guess. All are registry fields; README.md has why. The two
checkpoints' own training/inference sources describe an 8-layer CNN, not these weights, and are
stale: the state dicts are authoritative."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms as T

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_coreml import REPO_ROOT, deployment_target, save, sha256_of  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# ``(x/255 - mean)/std`` collapsed to one per-channel affine, ``x * SCALE + BIAS``, x in 0..255.
SCALE = tuple(1.0 / (255.0 * s) for s in IMAGENET_STD)
BIAS = tuple(-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))

# All that differs between the two nets. `labels` is each model's OWN order; `label_order.status` is
# how well it is known -- "derived" = read off whatever assigns the indices, "assumed" = plausible,
# unsupported by the training code, unconfirmed here. Shipping them as equally known is a lie the
# client cannot see through, so it surfaces "assumed" instead; promoting one is a word here.
SPECS = (
    dict(id="resnet18_kaggle_160", package="TPDResNetKaggle160.mlpackage",
         display="ResNet-18 (Kaggle, 160px)", size=160, prefix="", head="linear",
         checkpoint="saved_models/tennis_timm_resnet18_160_best.pth",
         labels=("backhand", "forehand", "ready_position", "serve"),
         # ASSUMED: the training branch assigns ids in FIRST-ENCOUNTER order over glob(), the
         # sorted() nearby labels nothing, and there is no dataset here to check against.
         label_order=dict(status="assumed", source="Kaggle_Model/dataset.py: first-encounter order "
                          "over glob(), not alphabetical; unverifiable, no dataset checked in")),
    dict(id="resnet18_pose_256", package="TPDResNetPose256.mlpackage",
         display="ResNet-18 (pose split, 256px)", size=256, prefix="model.", head="sequential",
         checkpoint="models/Comparison_Models/Resnet_Model/best_model.pt",
         labels=("forehand", "backhand", "ready_position", "serve"),
         label_order=dict(status="derived", source="Resnet_Model/data/pose_dataset.py: hardcoded "
                          "class_to_idx (forehand=0!)")),
)


class Normalized(nn.Module):
    """Raw 0-255 RGB in, logits out. The transform is the graph's first op because ``ct.ImageType``
    cannot carry it: ``bias`` is per channel but ``scale`` is a SCALAR, so a length-3 array
    broadcasts on the LAST axis -- coremltools 8.3.0: "Incompatible dim 3 ... vs. (1,1,1,3)"."""

    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body
        self.register_buffer("scale", torch.tensor(SCALE).view(1, 3, 1, 1))
        self.register_buffer("bias", torch.tensor(BIAS).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.body(image * self.scale + self.bias)


def build(spec: dict, checkpoint_path: Path) -> nn.Module:
    """Reconstruct one ResNet-18 and load its checkpoint with ``strict=True``."""
    classes = len(spec["labels"])
    net = torchvision.models.resnet18(num_classes=classes)
    if spec["head"] == "sequential":
        # best_model.pt's head is `fc.1.*`: index 0 has no parameters, so Dropout is all it can be.
        net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, classes))
    state, prefix = torch.load(str(checkpoint_path), map_location="cpu"), spec["prefix"]
    if prefix:
        stray = [k for k in state if not k.startswith(prefix)]
        if stray:
            raise SystemExit(f"{checkpoint_path.name}: {len(stray)} keys lack the '{prefix}' "
                             f"prefix, e.g. {stray[:3]}")
        state = {k[len(prefix):]: v for k, v in state.items()}
    # strict=True is the point: a renamed layer fails loudly instead of shipping random weights.
    net.load_state_dict(state, strict=True)
    return net.eval()


@torch.no_grad()
def check_normalization(module: nn.Module, net: nn.Module, size: int):
    """Baked-in vs the real torchvision eval transform, same pixels: measured, never trusted."""
    array = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                           T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    reference = net(transform(Image.fromarray(array)).unsqueeze(0))
    raw = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()  # as Core ML feeds it
    return float((reference - module(raw)).abs().max()), float(reference.abs().max())


def convert(ct, module: nn.Module, size: int, precision, target):
    traced = torch.jit.trace(module, torch.rand(1, 3, size, size) * 255.0).eval()
    return ct.convert(
        traced,
        # scale=1.0/no bias: normalization is inside the graph, so a scale here would double it.
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), scale=1.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram", compute_precision=precision, minimum_deployment_target=target)


def compact_json(obj) -> str:
    """indent=2, but flat arrays and leaf objects stay on one line -- this is read in diffs."""
    text = json.dumps(obj, indent=2)
    for p in (r"\[\s*([^][{}]*?)\s*\]", r"\{\s*([^{}]*?)\s*\}"):  # flat arrays, then leaf objects
        text = re.sub(p, lambda m: m.group(0)[0] + " ".join(m.group(1).split()) + m.group(0)[-1], text)
    return text + "\n"


def classifier_entry(spec: dict, checkpoint: Path, precision_label: str) -> dict:
    return {
        "id": spec["id"], "displayName": spec["display"], "kind": "classifier",
        "packages": [spec["package"]],
        "input": {"name": "image", "size": spec["size"], "mean": list(IMAGENET_MEAN),
                  "std": list(IMAGENET_STD), "normalizationBakedIn": True},
        "output": {"name": "logits", "type": "logits"},
        "labels": list(spec["labels"]), "labelOrder": dict(spec["label_order"]),
        "build": {"precision": precision_label, "minimumDeploymentTarget": "iOS17"},
        # Basename like TPDModelSpec.json's, one convention across all three; sha256 is the identity.
        "source": {"checkpoint": checkpoint.name, "sha256": sha256_of(checkpoint)},
    }


def pipeline_entry(models_dir: Path) -> dict:
    """The 3-stage model in the same shape, read from its own committed sidecars, restating none."""
    spec_path, labels_path = models_dir / "TPDModelSpec.json", models_dir / "TPDLabels.json"
    for path in (spec_path, labels_path):
        if not path.is_file():
            raise SystemExit(f"missing {path} -- run `make export-3stage` first")
    spec = json.loads(spec_path.read_text())
    return {
        "id": "tpd_3stage", "displayName": "TPD 3-stage (bbox + keypoints + pose)",
        "kind": "pipeline", "packages": [spec["bboxModel"], spec["poseModel"]],
        # Not a placeholder: under `(x/255 - mean)/std`, the one formula these fields carry, the
        # identity pair IS this pipeline's ToTensor()-only preprocessing, and parity asserts it.
        "input": {"name": "image", "size": spec["bboxInputSize"], "mean": [0.0] * 3,
                  "std": [1.0] * 3, "normalizationBakedIn": True},
        # forward() already applies softmax -- iOS must NOT softmax this one.
        "output": {"name": "probs", "type": "probabilities"},
        "labels": json.loads(labels_path.read_text())["labels"],
        "labelOrder": {"status": "derived", "source": "checkpoint label_names (TPDLabels.json)"},
        "build": {"precision": spec["precision"], "pipelineSpec": spec_path.name,
                  "minimumDeploymentTarget": spec["minimumDeploymentTarget"]},
        "source": {"checkpoint": spec["sourceCheckpoint"],
                   "sha256": spec["sourceCheckpointSha256"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, help="where the packages and the registry land",
                        default=REPO_ROOT / "frontend" / "ios" / "TPD" / "Models")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                        help="fp16 halves the 45 MB of fp32 weights; parity gates the drift")
    args = parser.parse_args()
    import coremltools as ct

    out_dir, label, target = args.out_dir.expanduser().resolve(), args.precision, deployment_target(ct)
    precision = ct.precision.FLOAT32 if label == "fp32" else ct.precision.FLOAT16
    print(f"out dir     {out_dir}\ncoremltools {ct.__version__} / torch {torch.__version__}"
          f" / torchvision {torchvision.__version__}\nprecision   {label}, target iOS17"
          f"\nbaked-in    (x/255 - {list(IMAGENET_MEAN)}) / {list(IMAGENET_STD)}  =  x * "
          + ",".join(f"{v:.10f}" for v in SCALE) + " + " + ",".join(f"{v:+.8f}" for v in BIAS))

    built = []
    for spec in SPECS:
        checkpoint = (REPO_ROOT / spec["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        net = build(spec, checkpoint)
        module = Normalized(net).eval()
        drift, magnitude = check_normalization(module, net, spec["size"])
        print(f"\n{spec['id']}  {spec['size']}px RGB 0-255 -> logits (1, 4)"
              f"\n  checkpoint     {spec['checkpoint']}"
              f"\n  class order    {list(spec['labels'])}  [{spec['label_order']['status'].upper()}]"
              f"\n  strict load    OK ({len(net.state_dict())} keys), normalization max|d| "
              f"{drift:.3e} vs torchvision (logits ~{magnitude:.2f})")
        if drift > 1e-4:
            raise SystemExit("baked-in normalization does not reproduce the torch transform")
        built.append((spec, checkpoint, convert(ct, module, spec["size"], precision, target)))

    # out_dir is untouched until both conversions succeeded: half a pair is worse than none.
    entries = [pipeline_entry(out_dir)]
    for spec, checkpoint, model in built:
        path = out_dir / spec["package"]
        print(f"\nwrote {path}  ({save(model, path):.1f} MB)")
        entries.append(classifier_entry(spec, checkpoint, label))

    registry = out_dir / "TPDModelRegistry.json"
    text = compact_json({"schemaVersion": 1, "models": entries})
    registry.write_text(text)
    print(f"wrote {registry}  ({len(text)} B)\n\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
