# iOS model export tooling

Converts the trained checkpoints into the Core ML packages the iOS app loads: the 3-stage
end-to-end model (`export_coreml.py`) and the two comparison ResNet-18s (`export_comparison.py`).
`make export` runs both; `make parity` gates all three.

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
`preprocessing/tensor_preprocessing.py::heatmaps_to_keypoints` into the graph:
per-channel flat argmax, `y = idx // W`, `x = idx % W`, `x/(W-1)`, `y/(H-1)`, and the **raw max
logit** as visibility (never sigmoided). Stage 3's `forward()` already softmaxes, so the exported
`probs` are probabilities — iOS must not softmax again.

The index split is done with **integer** ops (`torch.div(..., rounding_mode="floor")`) and only
the resulting row/column are cast to float. See below for why that matters.

Stage 1 stays a separate package because the crop between stages depends on stage-1 output and
cannot be expressed with static shapes. If the fused stage 2+3 conversion ever fails, the script
prints the reason, writes no `TPDPoseNet.mlpackage`, and exits non-zero pointing at the 3-model
fallback in `frontend/IOS_IMPLEMENTATION_PLAN.md` §3 — it never emits a broken package.

## The comparison ResNet-18s

`export_comparison.py` emits `TPDResNetKaggle160.mlpackage` (`image` 160×160 RGB **0-255** →
`logits` (1,4), 21.3 MB), `TPDResNetPose256.mlpackage` (same at 256×256, 21.3 MB) and
`TPDModelRegistry.json`, which describes all three shipped models. Both checkpoints are
torchvision-shaped ResNet-18s — 122 state-dict keys, identical to
`torchvision.models.resnet18(num_classes=4)` — so **timm is not a dependency**. They differ only in
key prefix and head: `tennis_timm_resnet18_160_best.pth` is unprefixed with a plain `fc`;
`Resnet_Model/best_model.pt` prefixes every key `model.` and its head is a `Sequential` whose
`fc.1` is the `Linear`. Both load with **`strict=True`** — a renamed layer must fail the export,
not load randomly initialised and ship confident garbage. `Resnet_Model`'s own
`models/pose_classification.py` (an 8-layer plain CNN), `train/train_classification.py` and
`inference/pose_infer.py` do **not** describe `best_model.pt`; no checkpoint in this repo matches
that architecture, so the weights are authoritative and that source is stale.

**Two contracts the client must not guess.** (1) Both ResNets return **raw logits**
(`CrossEntropyLoss`), while the 3-stage `PoseClassificationModel.forward` already softmaxes and
returns **probabilities** — softmaxing the wrong one is silent, not a crash. (2)
`Resnet_Model/data/pose_dataset.py` hardcodes `forehand=0, backhand=1`; the Kaggle and 3-stage
models use the alphabetical `backhand=0, forehand=1`, so indices 0 and 1 mean **opposite** things
across the two ResNets. Both are recorded per model in the registry (`output.type`, `labels`,
`labelOrderSource`) and gated by the CONTRACT TRAPS table in `make parity`. The orders come from
each model's own training source; **no dataset is checked in, so neither has been confirmed against
real images** — that still needs one on-device check with a known forehand.

`TPDModelSpec.json` and `TPDLabels.json` are untouched and keep working — they remain the truth for
the 3-stage model, and the registry *reads* them rather than restating them. The registry is purely
additive: one array of self-describing entries (`id`, `displayName`, `kind`, `packages`, `input`,
`output`, `labels`, `labelOrderSource`, `precision`, `minimumDeploymentTarget`, `source`), so a
fourth model is an entry, not a Swift change — reshaping a format Swift already parses would have
bought nothing.

**Normalization is baked in.** ImageNet `(x/255 − mean)/std` collapses to a per-channel
`x * scale + bias` → `scale 0.0171247538 0.0175070028 0.0174291939`,
`bias −2.11790393 −2.03571429 −1.80444444`. Those cannot live on `ct.ImageType` — its `scale` is a
scalar, see `Normalized` in `export_comparison.py` for the measured failure — so the packages take
plain 0-255 RGB and the affine is the graph's first op; same guarantee for the client, which hands
over an image and cannot get the normalization wrong. The exporter measures the result against the
real `torchvision.transforms` eval pipeline instead of trusting the algebra (`max|Δ|` 1.5e-07
Kaggle, 6.6e-07 pose) and refuses to write a package above 1e-4.

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
   `vis.sum()` is often negative (measured −40.6 on a black crop, −4.4 on mid-gray).
   `clamp_min(1e-6)` then makes the divisor 1e-6 and `center = (xy*vis).sum() / vis_sum` reaches
   3.5e7, past the fp16 ceiling of 65504. No inf/nan surfaces — the classifier collapses to one
   constant vector, predicting `serve` where torch says `forehand`. Fixed: fp32 downstream of argmax.

An earlier note here blamed "near-flat heatmap peaks" (median top-1/top-2 gap 0.10). That diagnosis
was wrong: the peaks are fine, the index cast and the overflow were not. The fp32 head costs 0.5 MB
over all-fp16 (TPDPoseNet 29.8 → 30.3 MB), not the 68 MB all-fp32 needs. Re-measured independently
on a second 20-input set, same pixels into both stacks: flat-argmax **index equality 359/360**
(360/360 on CPU+GPU), `max|Δprob| = 6.3e-3`, class 20/20. **Gate on index equality, not
bit-exactness** — 34 channels differ by ≤1 fp32 ULP and one lands 2 px out where two pixels 4.0e-4
apart in fp32 collapse to one fp16 value, a tie inherent to the fp16 U-Net. Visibility needs its own
tolerance: it is the raw logit, `max|Δvis| = 1.0e-2`, ~3× the probability delta.

Stage 1 stays plain fp16. Its bbox differs from torch by at most `3.3e-4` normalized (≈0.2 px at
640 px wide), which is sub-pixel, but `norm_bbox_to_xyxy_pixels` rounds to integers and that
rounding flips by one pixel on 19 of 60 random frames. A parity harness must therefore compare
bboxes with a tolerance and feed **the same** crop into both stacks when checking stages 2+3;
comparing independently-rounded crops measures the rounding, not the models.
