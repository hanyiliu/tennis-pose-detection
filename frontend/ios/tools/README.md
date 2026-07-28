# iOS model export tooling

Converts the trained end-to-end checkpoint into the Core ML packages the iOS app loads.

## Run it

```bash
python3.9 -m venv .venv-coreml
.venv-coreml/bin/pip install -r frontend/ios/tools/requirements.txt
.venv-coreml/bin/python frontend/ios/tools/export_coreml.py
```

Run it from anywhere — the script locates the repo root by walking up to the directory that
contains `models/pose_classification.py` and puts that on `sys.path` itself.

Options: `--checkpoint` (default `exports/colab_e2e_best_2.pt`), `--out-dir`
(default `frontend/ios/TPD/Models`), `--precision {fp16,fp32}` (default `fp16`).

## What it emits

Into `frontend/ios/TPD/Models/`:

| File | Contents |
|---|---|
| `TPDBBox.mlpackage` | stage 1. `image` 256×256 RGB → `bbox` (1,4), sigmoid `x,y,w,h` in [0,1]. ~4 MB fp16 |
| `TPDPoseNet.mlpackage` | stages 2+3 fused. `image` 128×128 RGB → `probs` (1,4) + `keypoints` (1,18,3). ~30 MB fp16 |
| `TPDLabels.json` | `{"labels": [...]}`, normalized `strip().lower().replace(" ", "_")`, checkpoint order |
| `TPDModelSpec.json` | input sizes, `numKeypoints`, `hiddenDim`, `numClasses`, source checkpoint name + sha256 |

The two JSON files are committed; the `.mlpackage` bundles are git-ignored and regenerated
from the checkpoint. Swift reads both JSONs from the bundle, so class order, class count and
input sizes are never hardcoded on the client.

Nothing is read from `checkpoint["args"]`: that dict has no `hidden_dim`, no `dropout` and no
`visibility_threshold`. `hidden_dim` comes from `pose_model_state["fc1.weight"].shape[0]`,
`num_classes` from `out.weight.shape[0]`, `num_keypoints` from
`extraction_conv1.weight.shape[0]`, and the input sizes from the checkpoint's own
`{bbox,keypoint}_image_{height,width}` keys.

Images enter as `ct.ImageType` with `scale = 1/255` and **no bias** — the Python pipeline
applies `ToTensor()` and no mean/std normalization, so a bias would silently break parity.
`minimum_deployment_target` is the highest `ct.target` the installed coremltools exposes,
detected with `getattr` rather than hardcoded (iOS18 on coremltools 8.3).

### The fused decode

`PoseNet.forward` bakes the heatmap decode from
`preprocessing/tensor_preprocessing.py::extract_keypoints_from_heatmaps` into the graph:
per-channel flat argmax, `y = idx // W`, `x = idx % W`, `x/(W-1)`, `y/(H-1)`, and the **raw max
logit** as visibility (never sigmoided). Stage 3's `forward()` already softmaxes, so the exported
`probs` are probabilities — iOS must not softmax again.

Stage 1 stays a separate package because the crop between stages depends on stage-1 output and
cannot be expressed with static shapes. If the fused stage 2+3 conversion ever fails, the script
prints the reason, writes no `TPDPoseNet.mlpackage`, and exits non-zero pointing at the 3-model
fallback in `frontend/IOS_IMPLEMENTATION_PLAN.md` §3 — it never emits a broken package.

## Why the pins are so low

The development machine is an Intel Mac. PyTorch publishes no macOS x86_64 wheels above 2.2.x,
and torch 2.2.2 is built against the numpy 1.x ABI, which forces `numpy<2` and coremltools 8.x.
See `requirements.txt` for the full chain. Apple Silicon can use newer versions; the pins are
kept so the export stays reproducible on both.

## fp16 vs fp32

Default `fp16` halves the download (34 MB vs 68 MB total). Verified on this Mac against the torch
pipeline (CPU compute units, synthetic input): the fp32 export reproduces the torch keypoints
exactly (18/18 pixel-exact) with `max|Δprob| ≈ 1.8e-6`, confirming the traced graph is faithful.
The fp16 export agrees on the predicted class but can shift an individual keypoint argmax by a few
pixels when a heatmap has a near-flat peak — on the synthetic probe the median top-1/top-2 logit
gap was 0.10, i.e. inside fp16 rounding. Use `--precision fp32` when chasing a numeric
discrepancy. PR3's parity harness pins the real tolerances against actual fixture frames.
