# TPD iOS app — build and model-export entry points.
# Every target is meant to be run from frontend/ios/.

SHELL       := /bin/bash
REPO_ROOT   := $(abspath $(CURDIR)/../..)
VENV        := $(REPO_ROOT)/.venv-coreml
PY          := $(VENV)/bin/python
PYTHON39    ?= python3.9
CHECKPOINT  ?= $(REPO_ROOT)/exports/colab_e2e_best_2.pt
TOOLS       := tools
# Single canonical home for Core ML artifacts. tools/export_coreml.py defaults
# to this same path — do not add a second models directory under TPD/, both
# would land inside the `sources: [TPD]` glob and get bundled twice.
MODELS_DIR  ?= TPD/Models
SRC_DIRS    := TPD TPDTests
# Hardcoding a device name rots: this defaulted to "iPhone 15", was bumped to "iPhone 16",
# and Xcode 26.5 ships neither -- its newest set starts at iPhone 17. Empty means "ask simctl
# in the recipe"; override with `make build SIM_DEVICE="iPhone 17 Pro"` to pin one.
# The lookup deliberately lives in the recipe, not in a $(shell ...): the device list contains
# literal parentheses, and any sed that matches them unbalances make's own paren counting and
# fails the whole file with "unterminated call to function `shell'".
SIM_DEVICE  ?=
CONFIG      ?= Debug

.PHONY: all venv export export-3stage export-comparison parity generate lint build clean
.DEFAULT_GOAL := lint

all: generate lint

# ---------------------------------------------------------------------------
# Python side (Core ML export + parity)
# ---------------------------------------------------------------------------
# This Mac is x86_64. PyTorch publishes no macOS x86_64 wheels above 2.2.x, so
# the only stack that resolves is python3.9 + torch 2.2.2 + numpy<2 +
# coremltools 8.x. Do not "upgrade" these pins on an Intel host.
# Pins live in tools/requirements.txt and nowhere else — duplicating them here
# would let the two drift apart silently.
$(PY):
	@test -f $(TOOLS)/requirements.txt || { \
		echo "$(TOOLS)/requirements.txt not found (it lands with the export tooling)" >&2; \
		exit 1; }
	$(PYTHON39) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r $(TOOLS)/requirements.txt

venv: $(PY)
	@$(PY) -c "import sys, torch, coremltools; \
	print('python', sys.version.split()[0], '| torch', torch.__version__, \
	'| coremltools', coremltools.__version__)"

# Regenerates TPDBBox.mlpackage / TPDPoseNet.mlpackage plus TPDLabels.json and
# TPDModelSpec.json, all into $(MODELS_DIR). The .mlpackage bundles are
# git-ignored on purpose; the two JSON sidecars are committed, so re-running
# this and seeing a diff in them is the intended way to surface model drift.
# `export` is every model the app can select, because `parity` gates every model the app can
# select; the halves stay separate targets so one can be re-exported while iterating.
export: export-3stage export-comparison

export-3stage: venv
	@test -f $(TOOLS)/export_coreml.py || \
		{ echo "missing $(TOOLS)/export_coreml.py (lands with the export PR)"; exit 1; }
	@test -f $(CHECKPOINT) || { echo "missing checkpoint $(CHECKPOINT)"; exit 1; }
	mkdir -p $(MODELS_DIR)
	$(PY) $(TOOLS)/export_coreml.py --checkpoint $(CHECKPOINT) --out-dir $(MODELS_DIR)

# The two comparison ResNet-18s plus TPDModelRegistry.json, which describes all three models. It
# READS TPDModelSpec.json/TPDLabels.json, so the prerequisite below is what keeps it after
# export-3stage even under `make -j`, where it would otherwise describe the PREVIOUS export.
export-comparison: export-3stage
	mkdir -p $(MODELS_DIR)
	$(PY) $(TOOLS)/export_comparison.py --out-dir $(MODELS_DIR)

# Runs the same fixture through torch and through the converted packages and
# asserts they agree at fp16 tolerances.
parity: venv
	@test -f $(TOOLS)/parity_check.py || \
		{ echo "missing $(TOOLS)/parity_check.py (lands with the parity PR)"; exit 1; }
	$(PY) $(TOOLS)/parity_check.py --checkpoint $(CHECKPOINT) --models $(MODELS_DIR)

# ---------------------------------------------------------------------------
# Xcode project
# ---------------------------------------------------------------------------
# TPD.xcodeproj is generated, git-ignored, and safe to delete at any time.
generate:
	xcodegen generate --spec project.yml

# Syntax gate. `swiftc -parse` stops after parsing, so it needs no iOS SDK and
# no Xcode.app — it is the only Swift check that runs on this machine today.
# Because it is the ONLY Swift gate, "found nothing to check" is a failure, not
# a pass: a missing or renamed source directory must not look like success.
# find's stderr is deliberately not discarded, and the file list is NUL-
# delimited into an array so paths with spaces survive.
lint:
	@set -eo pipefail; \
	for d in $(SRC_DIRS); do \
		[ -d "$$d" ] || { echo "lint: source directory '$$d' not found (run from frontend/ios)" >&2; exit 1; }; \
	done; \
	tmp=$$(mktemp); \
	find $(SRC_DIRS) -type f -name '*.swift' -print0 | sort -z > "$$tmp"; \
	st=$${PIPESTATUS[0]}; \
	[ "$$st" -eq 0 ] || { rm -f "$$tmp"; \
		echo "lint: find exited $$st — part of the source tree is unreadable" >&2; exit 1; }; \
	sources=(); \
	while IFS= read -r -d '' f; do sources+=("$$f"); done < "$$tmp"; \
	rm -f "$$tmp"; \
	if [ "$${#sources[@]}" -eq 0 ]; then \
		echo "lint: no Swift sources found under $(SRC_DIRS)" >&2; exit 1; \
	fi; \
	echo "lint: parsing $${#sources[@]} Swift file(s)"; \
	swiftc -parse -swift-version 6 "$${sources[@]}"; \
	echo "lint: OK"

# Requires a full Xcode.app; see README.md.
build: generate
	@if [ "$$(xcode-select -p)" = "/Library/Developer/CommandLineTools" ]; then \
		echo "ERROR: xcode-select points at CommandLineTools, which has no iOS SDK."; \
		echo "       Install Xcode.app, then:"; \
		echo "         sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"; \
		echo "         sudo xcodebuild -license accept"; \
		echo "       Until then use 'make lint' as the Swift gate."; \
		exit 1; \
	fi
	@set -eo pipefail; \
	device='$(SIM_DEVICE)'; \
	if [ -z "$$device" ]; then \
		device=$$(xcrun simctl list devices available 2>/dev/null \
			| sed -n 's/^ *iPhone/iPhone/p' | sed 's/ *(.*//' \
			| sed 's/ *$$//' | grep . | tail -1 || true); \
	fi; \
	if [ -z "$$device" ]; then \
		echo "ERROR: no iOS simulator device found. The runtime installs separately from the"; \
		echo "       SDK, so a working xcode-select is not enough:"; \
		echo "         xcodebuild -downloadPlatform iOS"; \
		echo "       Then re-run, or pin one with SIM_DEVICE=\"iPhone 17 Pro\"."; \
		exit 1; \
	fi; \
	echo "building for simulator: $$device"; \
	xcodebuild -project TPD.xcodeproj -scheme TPD -configuration $(CONFIG) \
		-destination "platform=iOS Simulator,name=$$device" \
		-derivedDataPath build/DerivedData build

clean:
	rm -rf build DerivedData TPD.xcodeproj TPD.xcworkspace TPD/Info.plist
