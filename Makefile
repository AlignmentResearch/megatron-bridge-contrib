# Megatron-Bridge (FAR.AI) Makefile. Three concerns:
#   1. Docs:    build the upstream Sphinx site (docs-html / docs-live / docs-clean).
#   2. Tests:   run the unit-test suites that FAR.AI CI gates on (test-unit family).
#   3. Image:   build docker/Dockerfile.ci and push ghcr.io/alignmentresearch/megatron-bridge,
#               which is what the gpu-tests workflow runs in (image-remote / image-local).
#
# Image build modes:
#   make image-remote  REMOTE (default). Spins up an ephemeral builder pod on the flamingo
#                      cluster, syncs your working tree into it, builds against the cluster's
#                      shared BuildKit (a CPU node with generous RAM — NOT your machine),
#                      pushes, then tears the pod down. Works from a laptop (only needs kubectl
#                      pointed at flamingo). See tools/build_image_on_flamingo.sh and
#                      k8s/build-image-pod.yaml.
#   make image-local   LOCAL. Builds on this machine's Docker daemon. This image is heavy
#                      (DeepEP + a full uv sync, ~90 min cold), so prefer the remote path.
#
# Dockerfile.ci builds from the BUILD CONTEXT (it COPYs the tree rather than cloning a git ref),
# so what gets built is your working tree — no need to commit or push your branch first.
#
# See README.farai.md for the fork overview and .github/workflows/README.md for CI mechanics.

# --- Image ----------------------------------------------------------------------------------
# The CI image is upstream's docker/Dockerfile.ci, built with INSTALL_DIFFUSION_DEPS=true when that
# base declares the arg (see DIFFUSION_ARGS below). It pulls the hash-locked WAN codecs that
# tests/unit_tests/diffusion needs; bases that gate them keep the arg off by default so those
# CVE-carrying codecs stay out of shipped framework images, and turn it on only for test images.
# Bases that predate the gating install the same deps unconditionally via `uv sync --all-extras
# --all-groups`, so no arg is needed there. There is deliberately no Dockerfile.farai — we need no
# fork-specific layers beyond rsync, so adding one would be a merge-conflict surface for nothing.
IMAGE_REPO ?= ghcr.io/alignmentresearch/megatron-bridge
IMAGE_TAG ?= latest
IMAGE_REF ?= $(IMAGE_REPO):$(IMAGE_TAG)
DOCKERFILE ?= docker/Dockerfile.ci
PLATFORM ?= linux/amd64
# Upstream's Dockerfile.ci default base is nvcr.io/nvidia/pytorch:26.06-py3; the deleted
# cicd-main.yml pinned 25.09-py3 for its own CI. Override BASE_IMAGE to pin a different NGC tag.
BASE_IMAGE ?=
BASE_ARGS := $(if $(strip $(BASE_IMAGE)),--build-arg BASE_IMAGE=$(BASE_IMAGE),)

# Pass INSTALL_DIFFUSION_DEPS only when the Dockerfile actually declares it. Newer bases exclude
# the diffusion group from `uv sync` and gate it behind this arg; older bases install it
# unconditionally via `uv sync --all-extras --all-groups` and declare no such ARG, where passing it
# emits an "unconsumed build arg" warning. Probing keeps this tooling working on either base, which
# matters because it gets cherry-picked across branches with very different Dockerfiles.
DIFFUSION_ARGS := $(shell grep -q '^ARG INSTALL_DIFFUSION_DEPS' $(DOCKERFILE) 2>/dev/null && \
	echo --build-arg INSTALL_DIFFUSION_DEPS=true)

# Registry-side layer cache (set CACHE_REF= to disable). Needs GHCR auth to read/write.
# Built into a separate var because make's $(if) would mis-split the commas in these flags.
CACHE_REF ?= $(IMAGE_REPO):cache
ifneq ($(strip $(CACHE_REF)),)
CACHE_ARGS := --cache-from type=registry,ref=$(CACHE_REF) --cache-to type=registry,ref=$(CACHE_REF),mode=max
endif

# Remote build (flamingo) — ephemeral builder Job + orchestration script. Leave BUILD_JOB_NAME
# empty to let the script name it <username>-mbridge-build from the detected user. GHCR push
# credentials are read in-cluster from your api-keys-<username> secret, so no token is needed
# locally for this path.
BUILD_JOB_NAME ?=
GHCR_SECRET_NAME ?=
GHCR_SECRET_KEY ?=
GHCR_USER ?=

# docker-container driver, so --secret and the registry cache work.
LOCAL_BUILDER ?= megatron-bridge-local
# --push publishes; --load keeps the result in the local docker daemon instead.
IMAGE_OUTPUT ?= --push
# Dockerfile.ci consumes GH_TOKEN as a build secret. Exported rather than interpolated into the
# recipe so the value reaches the build via the environment and never appears in any process's
# argv (world-readable via ps / /proc/<pid>/cmdline).
GH_TOKEN ?= $(GITHUB_PAT)
export GH_TOKEN

# --- Tests ----------------------------------------------------------------------------------
# Both launchers pin CUDA_VISIBLE_DEVICES="0,1" internally, so these need 2 visible GPUs. They
# are the same scripts the gpu-tests workflow runs in-cluster, so local and CI stay in parity.
UNIT_CORE := tests/unit_tests/Launch_Unit_Tests_Core.sh
UNIT_DIFFUSION := tests/unit_tests/Launch_Unit_Tests_Diffusion.sh

# Remote tests (flamingo) — ephemeral GPU pod + orchestration script. Leave TEST_JOB_NAME empty
# to let the script name it <username>-mbridge-test-<rand>. These rsync your WORKING TREE (not a
# commit) into the pod, so uncommitted changes are what gets tested.
TEST_JOB_NAME ?=
PRIORITY ?= interactive
# Cluster username. Leave empty to let the script auto-detect it from `kubectl auth whoami`;
# set it only to override.
FLAMINGO_USERNAME ?=
# TEST_GPUS sizes the test pod (default 2 = CI parity, and what the launchers pin internally).
# PYTEST_K is a pytest -k filter to target a single test.
TEST_GPUS ?= 2
PYTEST_K ?=
# READY_TIMEOUT: seconds to wait for the pod (a cold pull of the large image can take many
# minutes; default 30 min). IMAGE_PULL_POLICY: Always by default, matching CI — IMAGE_REF is a
# mutable tag, so IfNotPresent would silently test whatever :latest a node happened to cache.
# Always re-checks the digest and reuses cached layers when they match; set IfNotPresent to keep
# working through a registry outage.
READY_TIMEOUT ?= 1800
IMAGE_PULL_POLICY ?= Always
# TEST_DISK: pod ephemeral storage. Empty = let the script scale it (50Gi per GPU, 100Gi floor).
TEST_DISK ?=

# Shared env for every remote-test invocation; TEST_SUITE is appended per target.
REMOTE_TEST_ENV = JOB_NAME='$(TEST_JOB_NAME)' PRIORITY='$(PRIORITY)' IMAGE_REF='$(IMAGE_REF)' \
	FLAMINGO_USERNAME='$(FLAMINGO_USERNAME)' TEST_GPUS='$(TEST_GPUS)' PYTEST_K='$(PYTEST_K)' \
	READY_TIMEOUT='$(READY_TIMEOUT)' IMAGE_PULL_POLICY='$(IMAGE_PULL_POLICY)' \
	TEST_DISK='$(TEST_DISK)' KEEP='$(KEEP)' FOLLOW='$(FOLLOW)'

# --- Docs -----------------------------------------------------------------------------------
# --only-group docs installs just the Sphinx stack (no torch, no GPU deps); sphinx-autodoc2
# parses megatron.bridge statically without importing it, so builds stay fast. The submodule is
# still required: uv resolves megatron-core from 3rdparty/Megatron-LM even for a docs-only group.
UV_RUN_DOCS = uv run --only-group docs
DOCS_PORT ?= 8001

.PHONY: help check-uv check-submodule \
	docs-html docs-live docs-clean \
	test-unit test-unit-core test-unit-diffusion \
	test-unit-remote test-unit-core-remote test-unit-diffusion-remote \
	test-list test-logs test-teardown \
	lint image-remote image-local buildx-builder-local

help:
	@echo ""
	@echo "Megatron-Bridge (FAR.AI) development commands"
	@echo "============================================="
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-html            Build HTML documentation into docs/_build/html"
	@echo "  make docs-live            Start live-reload server on port $(DOCS_PORT)"
	@echo "  make docs-clean           Remove built docs and generated apidocs/skills trees"
	@echo ""
	@echo "Testing on the flamingo cluster (rsyncs your WORKING TREE into an ephemeral GPU pod):"
	@echo "  make test-unit-remote            Both unit suites (CI: @flamingo run gpu-tests)"
	@echo "  make test-unit-core-remote       Core suite only (CI: @flamingo run gpu-tests core)"
	@echo "  make test-unit-diffusion-remote  Diffusion suite only (CI: @flamingo run gpu-tests diffusion)"
	@echo "  make test-list                   List detached test jobs still running on the cluster"
	@echo "  make test-logs [JOB=..]          Re-attach to a detached run's output (auto-selects if only one)"
	@echo "  make test-teardown [JOB=..]      Stop a detached test job (JOB=all stops all)"
	@echo ""
	@echo "Testing locally (needs 2 GPUs on THIS machine — the launchers pin CUDA_VISIBLE_DEVICES=0,1):"
	@echo "  make test-unit            Both unit suites"
	@echo "  make test-unit-core       Core suite only"
	@echo "  make test-unit-diffusion  Diffusion suite only"
	@echo ""
	@echo "  Functional tests are not wired into FAR.AI CI; run a launcher directly on a GPU node:"
	@echo "    bash tests/functional_tests/launch_scripts/h100/active/L0_Launch_converter.sh"
	@echo ""
	@echo "Lint:"
	@echo "  make lint                 Run all pre-commit hooks over the tree (same as CI)"
	@echo ""
	@echo "Docker image (builds $(DOCKERFILE), pushes $(IMAGE_REF)):"
	@echo "  make image-remote         Build on the flamingo cluster (ephemeral pod -> shared BuildKit) and push"
	@echo "  make image-local          Build on the local Docker daemon (heavy; ~90 min cold)"
	@echo "  make buildx-builder-local Ensure the local '$(LOCAL_BUILDER)' buildx builder exists"
	@echo ""
	@echo "Variables (override on the command line, e.g. 'make image-remote IMAGE_TAG=dev'):"
	@echo "  IMAGE_REF=$(IMAGE_REF)"
	@echo "  DOCKERFILE=$(DOCKERFILE)"
	@echo "  PLATFORM=$(PLATFORM)"
	@echo "  BASE_IMAGE=$(BASE_IMAGE)   (empty => Dockerfile.ci's default NGC PyTorch tag)"
	@echo "  CACHE_REF=$(CACHE_REF)   (set empty to disable the registry layer cache)"
	@echo "  IMAGE_OUTPUT=$(IMAGE_OUTPUT)   (image-local only: --push to publish, --load to keep it local)"
	@echo "  BUILD_JOB_NAME=$(BUILD_JOB_NAME)   (remote build pod name; auto-derived <username>-mbridge-build if empty)"
	@echo "  GHCR_SECRET_NAME=$(GHCR_SECRET_NAME)   (GHCR push token secret; defaults to api-keys-<username>)"
	@echo "  GHCR_SECRET_KEY=$(GHCR_SECRET_KEY)   (key inside that secret holding the push token; defaults to GITHUB_PAT)"
	@echo "  GHCR_USER=$(GHCR_USER)   (ghcr.io login user; defaults to <username>)"
	@echo "  DOCS_PORT=$(DOCS_PORT)"
	@echo "  GH_TOKEN   (or GITHUB_PAT; from your environment — Dockerfile.ci needs it as a build secret)"
	@echo "  TEST_JOB_NAME=$(TEST_JOB_NAME)   (remote job name; auto-derived <username>-mbridge-test-<rand> if empty)"
	@echo "  FLAMINGO_USERNAME=$(FLAMINGO_USERNAME)   (auto-detected from 'kubectl auth whoami' if empty)"
	@echo "  TEST_GPUS=$(TEST_GPUS)   (remote test pod GPU count)"
	@echo "  PYTEST_K=$(PYTEST_K)   (pytest -k filter, e.g. 'test_auto_bridge and not slow')"
	@echo "  READY_TIMEOUT=$(READY_TIMEOUT)   (seconds to wait for the pod; raise for a slow cold image pull)"
	@echo "  IMAGE_PULL_POLICY=$(IMAGE_PULL_POLICY)   (Always=re-check the digest each run; IfNotPresent=reuse node cache)"
	@echo "  TEST_DISK=$(TEST_DISK)   (pod ephemeral storage; empty => 50Gi/GPU, 100Gi floor)"
	@echo "  PRIORITY=$(PRIORITY)   (k8s priorityClassName for the remote test pod)"
	@echo "  KEEP=$(KEEP)   (KEEP=1 leaves the test job up after the run; teardown via make test-teardown)"
	@echo "  FOLLOW=$(FOLLOW)   (FOLLOW=0 launches the test job detached and returns; re-attach via make test-logs)"
	@echo ""

check-uv:
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1 \
	)

# uv resolves megatron-core from the submodule for every group, docs included, and fails with
# "does not appear to be a Python project" when it is uninitialized.
check-submodule:
	@test -f 3rdparty/Megatron-LM/pyproject.toml || ( \
		echo ""; \
		echo "3rdparty/Megatron-LM is not initialized. Run:"; \
		echo "    git submodule update --init --recursive"; \
		exit 1 \
	)

# ==============================
# Documentation
# ==============================
# -d keeps the doctree cache OUT of the published tree. Without it Sphinx writes it to
# <outdir>/.doctrees, which on this repo is ~7 GB sitting inside a 131 MB site.

docs-html: check-uv check-submodule
	@echo "Building HTML documentation..."
	$(UV_RUN_DOCS) sphinx-build -b html -d docs/_build/doctrees docs docs/_build/html
	@echo "Built: docs/_build/html/index.html"

# sphinx-autobuild watches the source dir, and docs/ IS the source dir, so the doctree cache
# must live outside it or each rebuild retriggers the watcher into a loop.
docs-live: check-uv check-submodule
	@echo "Starting live-reload server on http://localhost:$(DOCS_PORT) ..."
	$(UV_RUN_DOCS) sphinx-autobuild -d .doctrees-cache docs docs/_build/html --port $(DOCS_PORT)

# docs/apidocs and docs/skills are generated at build time by conf.py and autodoc2
# (all three are gitignored).
docs-clean:
	@echo "Cleaning built documentation..."
	rm -rf docs/_build docs/apidocs docs/skills .doctrees-cache

# ==============================
# Tests
# ==============================

# --- Remote: run on an ephemeral GPU pod against your local working tree ---------------------
# Each target spins up a test pod (k8s/test-pod.yaml) on the cluster kubectl points at, rsyncs
# the working tree in, runs the same launchers the gpu-tests CI workflow runs, and tears the pod
# down. The tests run DETACHED on the pod, so a dropped connection does not kill them — re-attach
# with `make test-logs`, clean up with `make test-teardown`.

test-unit-remote:
	@$(REMOTE_TEST_ENV) TEST_SUITE='' bash tools/test_on_flamingo.sh

test-unit-core-remote:
	@$(REMOTE_TEST_ENV) TEST_SUITE='core' bash tools/test_on_flamingo.sh

test-unit-diffusion-remote:
	@$(REMOTE_TEST_ENV) TEST_SUITE='diffusion' bash tools/test_on_flamingo.sh

# Manage detached remote runs. JOB=<name> selects a specific job; with one running it is
# auto-selected; with several you get an interactive picker. `make test-teardown JOB=all` removes
# them all (local-dev jobs only — the selector never matches a CI run).
test-list:
	@bash tools/test_on_flamingo.sh list

test-logs:
	@JOB='$(JOB)' bash tools/test_on_flamingo.sh logs

test-teardown:
	@JOB='$(JOB)' bash tools/test_on_flamingo.sh teardown

# --- Local: run on THIS machine (requires 2 GPUs here) ---------------------------------------

test-unit: test-unit-core test-unit-diffusion

test-unit-core: check-uv check-submodule
	@bash $(UNIT_CORE)

test-unit-diffusion: check-uv check-submodule
	@bash $(UNIT_DIFFUSION)

# ==============================
# Lint
# ==============================
# Invoked as bare `pre-commit`, not `uv run --group dev pre-commit`: the dev group pulls
# nvidia-resiliency-ext, which ships Linux-only wheels and cannot install on macOS/arm64.
# No hook in .pre-commit-config.yaml needs the dev environment.

lint:
	@command -v pre-commit >/dev/null 2>&1 || { \
		echo "pre-commit not found on PATH. Install it: uv tool install pre-commit"; exit 1; }
	@pre-commit run --all-files --show-diff-on-failure --color=always

# ==============================
# Docker image
# ==============================

# --- Remote: build on the cluster (preferred) -------------------------------------------------
image-remote:
	@JOB_NAME='$(BUILD_JOB_NAME)' PRIORITY='$(PRIORITY)' IMAGE_REF='$(IMAGE_REF)' \
		PLATFORM='$(PLATFORM)' CACHE_REF='$(CACHE_REF)' DOCKERFILE='$(DOCKERFILE)' \
		BASE_IMAGE='$(BASE_IMAGE)' FLAMINGO_USERNAME='$(FLAMINGO_USERNAME)' \
		GHCR_SECRET_NAME='$(GHCR_SECRET_NAME)' GHCR_SECRET_KEY='$(GHCR_SECRET_KEY)' \
		GHCR_USER='$(GHCR_USER)' \
		bash tools/build_image_on_flamingo.sh

# --- Local: build on this machine -------------------------------------------------------------
image-local: buildx-builder-local
	@if [ -z "$$GH_TOKEN" ]; then \
		echo "ERROR: GH_TOKEN (or GITHUB_PAT) must be set — Dockerfile.ci takes it as a build secret."; \
		exit 1; \
	fi
	@test -f "$(DOCKERFILE)" || { echo "ERROR: $(DOCKERFILE) not found."; exit 1; }
	@echo "Building $(IMAGE_REF) from $(DOCKERFILE) [$(PLATFORM)] (output: $(IMAGE_OUTPUT))"
	@echo "This is a long build (DeepEP + a full uv sync); expect on the order of 90 minutes cold."
	@docker buildx build \
		--builder $(LOCAL_BUILDER) \
		--platform $(PLATFORM) \
		--file $(DOCKERFILE) \
		$(DIFFUSION_ARGS) \
		$(BASE_ARGS) \
		--secret id=GH_TOKEN,env=GH_TOKEN \
		$(CACHE_ARGS) \
		--tag $(IMAGE_REF) \
		$(IMAGE_OUTPUT) \
		.
	@echo "Done: $(IMAGE_REF) ($(IMAGE_OUTPUT))"

buildx-builder-local:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH."; exit 1; }
	@docker buildx inspect $(LOCAL_BUILDER) >/dev/null 2>&1 || { \
		echo "Creating local buildx builder '$(LOCAL_BUILDER)' (docker-container driver)"; \
		docker buildx create --name $(LOCAL_BUILDER) --driver docker-container; }
