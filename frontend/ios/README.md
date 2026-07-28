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
2. **Unclamped letterbox scale.** Python's `letterbox_resize` uses `PIL.thumbnail`, which never
   upscales, but the training targets in `preprocessing/tensor_preprocessing.py` and the reverse
   map in `backend/api/utils/image_outputs.py::_letterbox_xy_to_bbox_xy` both assume the
   **unclamped** `scale = min(128/w, 128/h)`. Swift uses the unclamped scale — it always fits the
   crop to 128 px. This matches training and the reverse map, and diverges from `inference.py`
   only on sub-128-px crops, which are degenerate.
3. **Lowercase label names.** The checkpoint ships `['Backhand', 'Forehand', 'Ready_Position',
   'Serve']`. iOS reads the normalized `backhand`, `forehand`, `ready_position`, `serve` from
   `TPDLabels.json`, matching the backend's normalization. Never hardcode the order.

## Privacy keys

`Info.plist` declares `NSCameraUsageDescription` (live inference) and
`NSPhotoLibraryAddUsageDescription` (saving overlaid video). It deliberately omits
`NSPhotoLibraryUsageDescription`: `PhotosPicker` runs out of process and grants access to only the
picked item, so the read permission is not needed and asking for it would be over-broad.
