# TPD iOS app — build and model-export entry points.
# Every target is meant to be run from frontend/ios/.

SHELL       := /bin/bash
REPO_ROOT   := $(abspath $(CURDIR)/../..)
VENV        := $(REPO_ROOT)/.venv-coreml
PY          := $(VENV)/bin/python
PYTHON39    ?= python3.9
CHECKPOINT  ?= $(REPO_ROOT)/exports/colab_e2e_best_2.pt
TOOLS       := tools
MODELS_DIR  ?= TPD/Resources/Models
SIM_DEVICE  ?= iPhone 15
CONFIG      ?= Debug

.PHONY: all venv export parity generate lint build clean
.DEFAULT_GOAL := lint

all: generate lint

# ---------------------------------------------------------------------------
# Python side (Core ML export + parity)
# ---------------------------------------------------------------------------
# This Mac is x86_64. PyTorch publishes no macOS x86_64 wheels above 2.2.x, so
# the only stack that resolves is python3.9 + torch 2.2.2 + numpy<2 +
# coremltools 8.x. Do not "upgrade" these pins on an Intel host.
$(PY):
	$(PYTHON39) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install \
		"torch==2.2.2" "torchvision==0.17.2" "numpy<2" "coremltools>=8,<9"

venv: $(PY)
	@$(PY) -c "import sys, torch, coremltools; \
	print('python', sys.version.split()[0], '| torch', torch.__version__, \
	'| coremltools', coremltools.__version__)"

# Regenerates TPDBBox.mlpackage / TPDPoseNet.mlpackage plus TPDLabels.json and
# TPDModelSpec.json. The .mlpackage bundles are git-ignored on purpose.
export: venv
	@test -f $(TOOLS)/export_coreml.py || \
		{ echo "missing $(TOOLS)/export_coreml.py (lands with the export PR)"; exit 1; }
	@test -f $(CHECKPOINT) || { echo "missing checkpoint $(CHECKPOINT)"; exit 1; }
	mkdir -p $(MODELS_DIR)
	$(PY) $(TOOLS)/export_coreml.py --checkpoint $(CHECKPOINT) --out $(MODELS_DIR)

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
lint:
	@sources=$$(find TPD TPDTests -type f -name '*.swift' 2>/dev/null | sort); \
	if [ -z "$$sources" ]; then echo "lint: no Swift sources found"; exit 0; fi; \
	count=$$(echo "$$sources" | wc -l | tr -d ' '); \
	echo "lint: parsing $$count Swift file(s)"; \
	swiftc -parse -swift-version 6 $$sources && echo "lint: OK"

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
	xcodebuild -project TPD.xcodeproj -scheme TPD -configuration $(CONFIG) \
		-destination 'platform=iOS Simulator,name=$(SIM_DEVICE)' \
		-derivedDataPath build/DerivedData build

clean:
	rm -rf build DerivedData TPD.xcodeproj TPD.xcworkspace TPD/Info.plist
