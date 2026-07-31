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
`logits` (1,4), 21.3 MB), `TPDResNetPose256.mlpackage` (same at 256×256) and
`TPDModelRegistry.json` — one array of self-describing entries covering all three shipped models, so
a fourth is an entry rather than a Swift change, while `TPDModelSpec.json`/`TPDLabels.json` stay the
3-stage model's truth and are *read* by it. `export_comparison.py`'s docstrings cover the rest.

**Three contracts the client must not guess**, each silent when wrong rather than a crash: both
ResNets emit **raw logits** while the 3-stage model already softmaxed; `pose_dataset.py` hardcodes
`forehand=0, backhand=1` against the other two's alphabetical order, so indices 0 and 1 mean
**opposite** things across the two ResNets; and the three class orders are not equally *known*. All
are per-model registry fields (`output.type`, `labels`, `labelOrder`), and `make parity`'s CONTRACT
TRAPS table asserts each model's measured convention and preprocessing against those very fields,
for **every** model — a registry claiming `probabilities` for a logit head fails rather than ships,
and class order is asserted per classifier because a gate written around "the one model that
deviates" stops covering the first when a second does. `labelOrder.status` is `derived` when the
order was read off whatever assigns the indices and `assumed` when not, so the client can surface a
guess as a guess.

| model | order | status | why |
|---|---|---|---|
| `tpd_3stage` | `backhand, forehand, …` | `derived` | the checkpoint's own `label_names` |
| `resnet18_pose_256` | `forehand, backhand, …` | `derived` | `pose_dataset.py`'s hardcoded `class_to_idx` |
| `resnet18_kaggle_160` | `backhand, forehand, …` | `assumed` | **not** what its training code does |

`Kaggle_Model/dataset.py` does have a `sorted()` over class folder names, but it is in the branch
that rebuilds a split from `image_paths`/`image_labels`: it runs only once labels exist and assigns
none. The branch that *trains* globs each extension in turn and assigns ids in **first-encounter
order** — extension then filesystem order, not the alphabet. (Those branches disagree with each
other too; upstream's bug, not one to inherit.) **No dataset is checked in and no kagglehub cache
exists, so this repo cannot confirm that order at all** — alphabetical is a guess, shipped as one. A
still of a known forehand settles it: one word in `SPECS` and a re-export, no Swift and no harness
change. `resnet18_pose_256`'s is derived but likewise unconfirmed on images.

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
