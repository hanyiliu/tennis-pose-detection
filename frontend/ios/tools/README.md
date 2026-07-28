# iOS model export tooling

Converts the trained end-to-end checkpoint into the Core ML packages the iOS app loads.

## Run it

```bash
python3.9 -m venv .venv-coreml
.venv-coreml/bin/pip install -r frontend/ios/tools/requirements.txt
.venv-coreml/bin/python frontend/ios/tools/export_coreml.py
```

Run it from anywhere: the script finds the repo root by walking up to `models/pose_classification.py`.

Options: `--checkpoint` (default `exports/colab_e2e_best_2.pt`), `--out-dir`
(default `frontend/ios/TPD/Models`), `--precision {fp16,fp32}` (default `fp16`).

## What it emits

Into `frontend/ios/TPD/Models/`:

| File | Contents |
|---|---|
| `TPDBBox.mlpackage` | stage 1. `image` 256×256 RGB → `bbox` (1,4), sigmoid `x,y,w,h` in [0,1]. 4.2 MB |
| `TPDPoseNet.mlpackage` | stages 2+3 fused. `image` 128×128 RGB → `probs` (1,4) + `keypoints` (1,18,3). 30.3 MB |
| `TPDLabels.json` | `{"labels": [...]}`, normalized `strip().lower().replace(" ", "_")`, checkpoint order |
| `TPDModelSpec.json` | input sizes, `numKeypoints`, `hiddenDim`, `numClasses`, `precision`, `minimumDeploymentTarget`, checkpoint name + sha256 |

The two JSON files are committed; the `.mlpackage` bundles are git-ignored and regenerated from the
checkpoint. Swift reads both JSONs from the bundle, so class order, class count and input sizes are
never hardcoded on the client.

Nothing is read from `checkpoint["args"]` — it has no `hidden_dim`/`dropout`/`visibility_threshold`.
Shapes carry them: `hidden_dim` from `pose fc1.weight`, `num_classes` from `out.weight`,
`num_keypoints` from `extraction_conv1.weight`, sizes from `{bbox,keypoint}_image_{height,width}`.

Images enter as `ct.ImageType` with `scale = 1/255` and **no bias** — the Python pipeline
applies `ToTensor()` and no mean/std normalization, so a bias would silently break parity.
`minimum_deployment_target` is hardcoded to `ct.target.iOS17` to match the app's
`IPHONEOS_DEPLOYMENT_TARGET`; if a coremltools without that target is installed the export fails
loudly rather than substituting another one. Both models declare `float32` outputs, so Swift
reads plain `Float32` `MLMultiArray`s whatever the internal precision is.

### The fused decode

`PoseNet.forward` bakes the heatmap decode from
`preprocessing/tensor_preprocessing.py::extract_keypoints_from_heatmaps` into the graph:
per-channel flat argmax, `y = idx // W`, `x = idx % W`, `x/(W-1)`, `y/(H-1)`, and the **raw max
logit** as visibility (never sigmoided). Stage 3's `forward()` already softmaxes, so the exported
`probs` are probabilities — iOS must not softmax again.

The index split is done with **integer** ops (`torch.div(..., rounding_mode="floor")`) and only
the resulting row/column are cast to float. See below for why that matters.

Stage 1 stays a separate package because the crop between stages depends on stage-1 output and
cannot be expressed with static shapes. If the fused stage 2+3 conversion ever fails, the script
prints the reason, writes no `TPDPoseNet.mlpackage`, and exits non-zero pointing at the 3-model
fallback in `frontend/IOS_IMPLEMENTATION_PLAN.md` §3 — it never emits a broken package.

## Why the pins are so low

The development machine is an Intel Mac. PyTorch publishes no macOS x86_64 wheels above 2.2.x, and
torch 2.2.2 is built against the numpy 1.x ABI, which forces `numpy<2` and coremltools 8.x — see
`requirements.txt`. Apple Silicon can use newer versions; the pins keep the export reproducible.

## Precision: why it is mixed, not plain fp16

Plain fp16 broke the fused graph in two ways. Both were reproduced at the MIL level and fixed;
`--precision fp16` now means *fp16 convolutions, fp32 decode and stage 3*, recorded honestly in
`TPDModelSpec.json` as `"fp16-unet-fp32-decode-and-stage3"`.

1. **fp16 rounded the argmax index.** coremltools inserted `cast(reduce_argmax → fp16)` before the
   `idx // W` split. The flat index runs to `H*W-1 = 16383` and fp16's 11-bit mantissa stops being
   exact at 2048, so 4097 decoded as 4096 and 16383 as 16384: most channels landed on the wrong
   pixel. Fixed by splitting the index with integer ops before any float cast — the row and column
   that do get cast are 0..127, which fp16 holds exactly.
2. **fp16 overflowed in the stage-3 body center.** Visibility here is the raw max logit, so
   `vis.sum()` is often negative (measured −40.6 on a black crop). `clamp_min(1e-6)` then makes the
   divisor 1e-6 and `center = (xy*vis).sum() / vis_sum` reaches ~1e6, past the fp16 ceiling of
   65504 — `centered_xy` became inf/nan and fed the classifier. Fixed by running everything
   downstream of the argmax in fp32 via `ct.transform.FP16ComputePrecision(op_selector=…)`.

The earlier note in this file blamed "near-flat heatmap peaks" with a median top-1/top-2 logit gap
of 0.10. That diagnosis was wrong: the peaks are fine, the index cast and the overflow were not.
Keeping the U-Net in fp16 costs 0.5 MB against the old all-fp16 build (TPDPoseNet 29.8 → 30.3 MB,
34.0 → 34.5 MB total) rather than the 68 MB an all-fp32 export needs. Measured here on CPU compute
units over 20 deterministic inputs (black / white / two flat grays / 10 random 128×128 crops /
6 random full frames), exported keypoints are **bit-identical** to the torch pipeline on all 360
channels, `max|Δprob| = 6.3e-3`, and the predicted class agrees 20/20.

Stage 1 stays plain fp16. Its bbox differs from torch by at most `3.3e-4` normalized (≈0.2 px at
640 px wide), which is sub-pixel, but `norm_bbox_to_xyxy_pixels` rounds to integers and that
rounding flips by one pixel on 19 of 60 random frames. A parity harness must therefore compare
bboxes with a tolerance and feed **the same** crop into both stacks when checking stages 2+3;
comparing independently-rounded crops measures the rounding, not the models.
