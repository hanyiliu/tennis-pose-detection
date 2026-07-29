# TPD — iOS app

SwiftUI front end for the tennis pose detection pipeline. Runs the three-stage model on device
through Core ML. See `../IOS_README.md` for product scope and `../IOS_IMPLEMENTATION_PLAN.md` for
the execution plan.

---

## ⚠️ Xcode.app is required and is NOT currently installed

`xcode-select -p` on this machine returns `/Library/Developer/CommandLineTools`. Command Line Tools
ship **no iOS SDK, no simulators and no `simctl`**, so `xcodebuild -destination 'platform=iOS
Simulator,…'` cannot run and `make build` will refuse with a non-zero exit. The scaffold, the Swift
sources and the Core ML export are all still developable; only the iOS build and simulator QA are
blocked.

Remediation, in order:

```bash
# 1. Install Xcode from the Mac App Store (~17 GB), then:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept

# 2. Confirm the iOS SDK is visible:
xcodebuild -showsdks | grep iphonesimulator
xcrun simctl list devices available

# 3. Then the normal loop works:
cd frontend/ios && make generate && make build
```

Steps 1 and 2 need the user's password and an App Store session; they cannot be automated.

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Xcode | 16+ | iOS 17.0 deployment target (hard floor, see below), Swift 6 language mode |
| XcodeGen | 2.46+ | `brew install xcodegen` |
| Python | 3.9 | Export only. This is an Intel Mac; see below. |

`make build` targets the `iPhone 16` simulator (`SIM_DEVICE`), which is in Xcode 16's default device
set; override with `make build SIM_DEVICE='iPhone 16 Pro'` if you run a different one.

## Build

```bash
cd frontend/ios
make generate   # xcodegen generate -> TPD.xcodeproj (git-ignored)
make lint       # swiftc -parse over every Swift file — works without Xcode.app
make build      # xcodebuild for the iOS Simulator — needs Xcode.app
open TPD.xcodeproj
```

`project.yml` is the source of truth. `TPD.xcodeproj`, `TPD/Info.plist` and everything under
`build/` are generated artifacts and are git-ignored — never edit them by hand, and never commit
them. The app target's `sources:` is a **directory glob** (`TPD`), so adding a Swift file requires
no spec edit; just re-run `make generate`.

### `make lint` is the syntax gate

`swiftc -parse` stops after parsing, so it resolves no imports and needs no SDK. It is the only
Swift check that runs on this machine today and it catches every syntax error. It does **not**
type-check, so SwiftUI/AVFoundation API misuse survives it and is only caught by a real iOS build.

Because it is the only gate, **finding zero Swift files is a failure, not a pass**: `make lint`
exits non-zero if `TPD/` or `TPDTests/` is missing or contains no `.swift`, so a renamed directory
cannot masquerade as a clean run.

## Model artifacts

**`frontend/ios/TPD/Models/` is the one and only model directory.** `tools/export_coreml.py`
defaults to it and the `MODELS_DIR` in the Makefile points at it. Never add a second models
directory anywhere under `TPD/`, however nested: the whole tree is inside the app target's
`sources:` glob, so a duplicate silently bundles two divergent copies of the JSON sidecars into the
app and the loser is whichever one the copy phase happens to write last.

```bash
make venv     # creates ../../.venv-coreml (python3.9)
make export   # checkpoint -> TPD/Models/*.mlpackage + JSON sidecars
make parity   # torch vs Core ML numeric parity at fp16 tolerances
```

**`.mlpackage` bundles are git-ignored and never committed** — ~35 MB of binary weights, and the
repo has never used Git LFS. `make export` regenerates them from `exports/colab_e2e_best_2.pt`,
which *is* committed. **Run `make export` before `make build`** — a clone has no `.mlpackage` until
you do.

**`TPDLabels.json` and `TPDModelSpec.json` *are* committed.** They are a few hundred bytes of text,
they document the contract Swift reads for class order and input sizes, and they let the project
build and lint without a 35 MB export. Re-running `make export` is expected to overwrite them; if
that produces a diff, the checkpoint drifted from the committed contract and the diff is the alarm.

The pinned stack is `torch==2.2.2`, `torchvision==0.17.2`, `numpy<2`, `coremltools>=8,<9` on
python3.9. PyTorch publishes no macOS **x86_64** wheels above 2.2.x, so on this Intel host those
pins are the only combination that resolves. Do not bump them here.

### The iOS 17 floor is a two-sided constraint

The app pins `IPHONEOS_DEPLOYMENT_TARGET = 17.0` (`project.yml`), so **every exported `.mlpackage`
must be built with `minimum_deployment_target = ct.target.iOS17`**. `export_coreml.py` hardcodes
exactly that; it must not auto-select the newest target coremltools happens to know about. A package
emitted for iOS 18 still loads on a newer simulator runtime and then fails at `MLModel(contentsOf:)`
on a real iOS 17 device — the worst possible place to discover it. If you ever raise the deployment
target, raise both sides in the same change.

## Decided divergences from the Python backend

These are intentional. Do not "fix" them.

1. **Single softmax.** `PoseClassificationModel.forward()` already applies softmax;
   `backend/api/services/model_service.py` applies a second one, flattening the web demo's
   confidences. iOS softmaxes **once**, matching `inference/inference.py`. Argmax is identical
   either way, so iOS confidence values legitimately read *higher* than the web app.
2. **Clamped letterbox scale — and a correct reverse map.** `letterbox_resize` is
   `PIL.thumbnail`, which never upscales, so the forward scale is `min(1, min(128/w, 128/h))`
   and a crop already smaller than 128 px is pasted at native size on black. Swift does the
   same. This *reverses* an earlier decision that used the unclamped scale "because it matches
   training". That premise was false: `train/keypoint_train.py:220` letterboxes the training
   **input** through `letterbox_resize` too, so the model was fed thumbnailed, never-upscaled
   crops. Only the *target* heatmaps (`preprocessing/tensor_preprocessing.py:120`) use an
   unclamped scale, which makes training self-inconsistent below 128 px — the model is
   unreliable on that branch whatever the client does — but the input convention is
   unambiguous and the deployed backend shares it. Measured on 12 sub-128 crops through the
   real stage-2 + stage-3 checkpoint: the unclamped scale flipped the predicted class on 3 of
   12, every one at a 1.000 softmax margin; clamped agrees 12/12 and its 128x128 buffer is
   byte-identical to Python's on all 12.

   `letterboxToFrame` uses the same clamped scale, so it is a true inverse of the forward
   transform. `_letterbox_xy_to_bbox_xy` in `backend/api/utils/image_outputs.py` still divides
   by the unclamped scale and therefore mis-places keypoints on sub-128 crops. iOS is
   deliberately correct rather than bug-compatible there: that path only draws overlays and
   can never change the predicted class.
3. **Lowercase label names.** The checkpoint ships `['Backhand', 'Forehand', 'Ready_Position',
   'Serve']`. iOS reads the normalized `backhand`, `forehand`, `ready_position`, `serve` from
   `TPDLabels.json`, matching the backend's normalization. Never hardcode the order.
4. **Resampling: a full PIL port, no residual.** `TPDResample` ports Pillow's resampler because
   no `CIFilter` reproduces PIL's downscale-scaled filter support — `CILanczosScaleTransform`
   moved the stage-1 crop rect by a median of 34 px (max 89, 0/24 frames matching). Stage 1 is
   byte-identical to `Image.resize(..., BILINEAR)`, 24/24 rects. Stage 2 now ports
   `PIL.thumbnail` in full — `round_aspect` sizing, the `reducing_gap=2.0` `Image.reduce`
   pre-pass, `_get_safe_box`, and the fact that `_imaging.c` takes the resize box as a C
   **float** — and is byte-identical to `letterbox_resize` on 66/66 curated crop sizes plus 800
   random ones. Never put a `CIFilter` back.

   The boundary this file used to claim — byte identity "only while both crop sides are under
   512 px" — was wrong twice over, and both errors are measured, not argued:

   All figures below come from one scan of every crop size in `1..1399` squared with a side
   over 128 px — 1940817 sizes, all of which `thumbnail` actually resizes.

   * The pre-reduce factor is `int(side / final_side / reducing_gap)` per axis, and the minor
     axis's `final_side` is itself about `128 * minor / major`, so it is the **larger** side
     reaching 512 that arms *both* factors: 0 of the 1940817 have a larger side >= 512 without
     arming it. The smaller side is not the gate at all — 3273 sizes arm the pre-reduce with
     **both** sides under 512, extreme aspect ratios where `round_aspect` collapses the minor
     axis to a pixel or two (4x342 fits to 1x128 and so reduces 2x horizontally).
   * Independently of the pre-reduce, plain `round()` on the scaled minor side disagrees with
     `round_aspect` on 9342 of the 1940817 (0.481%): 635 exact `.5` ties, which `round_aspect`
     breaks toward `floor` where `round()` goes up, and 8707 genuine disagreements because the
     minor-axis branch minimizes `|aspect - x/n|`, not `|x/aspect - n|`. Each is a 1-px change
     in the minor axis, which shifts every pasted pixel — so byte identity broke well below
     512 px too.

## Privacy keys

`Info.plist` declares `NSCameraUsageDescription` (live inference) and
`NSPhotoLibraryAddUsageDescription` (saving overlaid video). It deliberately omits
`NSPhotoLibraryUsageDescription`: `PhotosPicker` runs out of process and grants access to only the
picked item, so the read permission is not needed and asking for it would be over-broad.
