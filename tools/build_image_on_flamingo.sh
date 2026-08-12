#!/usr/bin/env bash
#
# Build docker/Dockerfile.ci on the FAR.AI "flamingo" cluster and push it to GHCR.
#
# The cluster's shared BuildKit (buildkitd.buildkitd.svc.cluster.local:1234) is only reachable
# from inside the cluster, so we can't build from a laptop directly. This script:
#   1. creates an ephemeral builder Job (k8s/build-image-pod.yaml),
#   2. syncs the local working tree into the pod,
#   3. execs `docker buildx build --push` against the shared remote BuildKit,
#   4. tears the Job down (always, via an EXIT trap).
#
# The build compute runs on the cluster BuildKit (a CPU node with generous RAM), not locally —
# which matters here: this image compiles DeepEP and runs a full `uv sync`, on the order of 90
# minutes cold.
#
# NOTE: Dockerfile.ci builds from the BUILD CONTEXT (`COPY --chmod=644 . /opt/Megatron-Bridge`)
# rather than cloning a git ref inside the build. So what gets built is exactly the working tree
# synced below — there is no git ref to pass and no need to commit or push your branch first.
# Uncommitted changes are built as-is.
#
# Usage: invoked by `make image-remote`. Override via env: IMAGE_REF, PLATFORM, BASE_IMAGE,
#        CACHE_REF, JOB_NAME, PRIORITY, READY_TIMEOUT.
set -euo pipefail

# ---- cluster preflight (needed before we introspect the cluster for identity) ----------------
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found on PATH."; exit 1; }
# Namespaced reachability check — `kubectl cluster-info` reads cluster-scoped resources that a
# namespaced user often can't, so it false-negatives even when the cluster is reachable.
kubectl get pods >/dev/null 2>&1 || {
  echo "ERROR: kubectl can't reach the cluster or access the current namespace (context: $(kubectl config current-context 2>/dev/null || echo none))."
  exit 1
}

# ---- identity + config (overridable via env) -------------------------------------------------
# Your cluster username, used to locate your per-user secret. Priority: explicit FLAMINGO_USERNAME,
# else parsed from `kubectl auth whoami` (e.g. oidc:you@far.ai -> you).
if [ -z "${FLAMINGO_USERNAME:-}" ]; then
  FLAMINGO_USERNAME="$(kubectl auth whoami -o jsonpath='{.status.userInfo.username}' 2>/dev/null \
    | sed -e 's/^[^:]*://' -e 's/@.*//' || true)"
fi

JOB_NAME="${JOB_NAME:-${FLAMINGO_USERNAME:-${USER:-mbridge}}-mbridge-build}"
PRIORITY="${PRIORITY:-interactive}"
IMAGE_REF="${IMAGE_REF:-ghcr.io/alignmentresearch/megatron-bridge:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
# Note: `-` not `:-` so an explicitly empty CACHE_REF (the documented "disable cache" knob,
# forwarded by `make image-remote CACHE_REF=`) is honored rather than reset to the default.
CACHE_REF="${CACHE_REF-ghcr.io/alignmentresearch/megatron-bridge:cache}"
# Empty => Dockerfile.ci's own default NGC PyTorch tag.
BASE_IMAGE="${BASE_IMAGE:-}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.ci}"
MANIFEST="${MANIFEST:-k8s/build-image-pod.yaml}"
BUILDKIT_ADDRESS="${BUILDKIT_ADDRESS:-buildkitd.buildkitd.svc.cluster.local}"
CERT_ROOT="${CERT_ROOT:-/etc/buildkit-certs}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

# GHCR push credential: a user PAT with write:packages, read from your own per-user k8s secret
# (the shared buildkit-ghcr-auth secret is read-only). The build `docker login`s with it before
# pushing, and the same PAT is handed to the Dockerfile as its GH_TOKEN build secret. Default
# secret name is api-keys-<username>; set GHCR_SECRET_NAME to override the whole name, or
# FLAMINGO_USERNAME to change just the <username> part.
if [ -z "${GHCR_SECRET_NAME:-}" ]; then
  if [ -z "$FLAMINGO_USERNAME" ]; then
    echo "ERROR: couldn't determine your cluster username from 'kubectl auth whoami'."
    echo "       Set FLAMINGO_USERNAME=<you> (secret defaults to api-keys-<you>),"
    echo "       or GHCR_SECRET_NAME=<full-secret-name> to point at your token secret directly."
    exit 1
  fi
  GHCR_SECRET_NAME="api-keys-${FLAMINGO_USERNAME}"
fi
GHCR_SECRET_KEY="${GHCR_SECRET_KEY:-GITHUB_PAT}"
GHCR_USER="${GHCR_USER:-${FLAMINGO_USERNAME:-${USER:-}}}"

REPO_ROOT="$(git rev-parse --show-toplevel)"

# ---- remaining preflight ---------------------------------------------------------------------
test -f "$REPO_ROOT/$DOCKERFILE" || { echo "ERROR: $DOCKERFILE not found."; exit 1; }
test -f "$REPO_ROOT/$MANIFEST" || { echo "ERROR: $MANIFEST not found."; exit 1; }
# Dockerfile.ci resolves megatron-core from the submodule and hard-fails if its compiled dataset
# helper is absent, so an uninitialized submodule would only surface ~an hour into the build.
test -f "$REPO_ROOT/3rdparty/Megatron-LM/pyproject.toml" || {
  echo "ERROR: 3rdparty/Megatron-LM is not initialized — the build needs it in the context."
  echo "       Run: git submodule update --init --recursive"
  exit 1
}
# The GHCR push token must exist before we spin up the pod, else it fails to start opaquely.
kubectl get secret "$GHCR_SECRET_NAME" -o "jsonpath={.data.$GHCR_SECRET_KEY}" 2>/dev/null | grep -q . || {
  echo "ERROR: secret '$GHCR_SECRET_NAME' has no key '$GHCR_SECRET_KEY' in this namespace."
  echo "       It must hold a GitHub PAT with write:packages on $IMAGE_REF."
  echo "       Override the secret with GHCR_SECRET_NAME= / GHCR_SECRET_KEY=, or your username with FLAMINGO_USERNAME=."
  exit 1
}

echo "▶ Building $IMAGE_REF"
echo "    dockerfile : $DOCKERFILE (working tree)"
echo "    source     : the synced working tree IS the build context (no git ref; uncommitted changes included)"
echo "    platform   : $PLATFORM"
echo "    base image : ${BASE_IMAGE:-<Dockerfile.ci default>}"
echo "    push creds : secret ${GHCR_SECRET_NAME}[${GHCR_SECRET_KEY}] as ghcr.io user '$GHCR_USER'"
echo "    job/context: $JOB_NAME @ $(kubectl config current-context)"

# ---- always tear the builder down ------------------------------------------------------------
cleanup() {
  echo "▶ Tearing down builder job $JOB_NAME"
  kubectl delete job "$JOB_NAME" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- create the builder job ------------------------------------------------------------------
kubectl delete job "$JOB_NAME" --ignore-not-found >/dev/null 2>&1 || true
sed -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    -e "s|__PRIORITY__|${PRIORITY}|g" \
    -e "s|__GHCR_SECRET_NAME__|${GHCR_SECRET_NAME}|g" \
    -e "s|__GHCR_SECRET_KEY__|${GHCR_SECRET_KEY}|g" \
    -e "s|__GHCR_USER__|${GHCR_USER}|g" \
    "$REPO_ROOT/$MANIFEST" | kubectl create -f -

# ---- wait for the pod to be running ----------------------------------------------------------
echo "▶ Waiting for builder pod (timeout ${READY_TIMEOUT}s)…"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
POD=""
while :; do
  POD="$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  [ "$(date +%s)" -ge "$deadline" ] && {
    echo "ERROR: builder pod never scheduled (kueue admission / quota?). Check: kubectl describe job $JOB_NAME"
    exit 1
  }
  sleep 3
done
kubectl wait --for=condition=ready "pod/$POD" --timeout="${READY_TIMEOUT}s"

# ---- sync the working tree into the pod ------------------------------------------------------
# TWO passes, and this is load-bearing: `git ls-files --cached` reports the submodule as a single
# gitlink entry, so a one-pass list would ship an EMPTY 3rdparty/Megatron-LM and the build would
# fail deep into `uv sync`. --recurse-submodules expands it (~3k files) but git refuses to combine
# it with --others, so untracked-but-not-ignored files come from a second pass. The two lists are
# disjoint (tracked vs untracked), so no entry is tarred twice.
echo "▶ Syncing working tree into $POD:/build/repo (incl. 3rdparty/Megatron-LM)"
kubectl exec "$POD" -- mkdir -p /build/repo
# COPYFILE_DISABLE=1 is load-bearing on macOS: bsdtar otherwise emits an AppleDouble "._<name>"
# companion for every file carrying extended attributes (com.apple.provenance is set on ~14k files
# in a normal checkout). GNU tar in the pod extracts those as real "._foo.py" files, Dockerfile.ci
# then COPYs them into the image, and code that globs sources and calls read_text() — e.g.
# flat_perf_recipe_names() in scripts/performance/utils/utils.py — reads one and dies with
# "UnicodeDecodeError: invalid start byte", failing ~21 otherwise-passing tests.
#
# Note this cannot be verified with `tar -tzf` on macOS: that transparently re-merges AppleDouble
# entries and reports a clean listing. Use Python's tarfile to inspect. Ignored by GNU tar, so it
# is safe to set unconditionally.
( cd "$REPO_ROOT" && { git ls-files -z --cached --recurse-submodules; \
                       git ls-files -z --others --exclude-standard; } \
    | COPYFILE_DISABLE=1 tar --null -T - -czf - ) \
  | kubectl exec -i "$POD" -- tar -xzf - -C /build/repo

# ---- build + push on the cluster's BuildKit --------------------------------------------------
echo "▶ Building on flamingo BuildKit and pushing $IMAGE_REF (DeepEP + full uv sync; ~90 min cold)"
cache_args=""
[ -n "$CACHE_REF" ] && cache_args="--cache-from type=registry,ref=${CACHE_REF} --cache-to type=registry,ref=${CACHE_REF},mode=max"
base_args=""
[ -n "$BASE_IMAGE" ] && base_args="--build-arg BASE_IMAGE=${BASE_IMAGE}"

# INSTALL_DIFFUSION_DEPS=true pulls the hash-locked WAN codecs that tests/unit_tests/diffusion
# needs. Newer bases exclude the diffusion group from `uv sync` and gate it behind this arg; older
# bases install it unconditionally via `uv sync --all-extras --all-groups` and declare no such ARG,
# where passing it emits an "unconsumed build arg" warning. Probe the Dockerfile so this script
# works on either base — it gets cherry-picked across branches with very different Dockerfiles.
diffusion_args=""
if grep -q '^ARG INSTALL_DIFFUSION_DEPS' "$REPO_ROOT/$DOCKERFILE" 2>/dev/null; then
  diffusion_args="--build-arg INSTALL_DIFFUSION_DEPS=true"
fi

# GH_TOKEN / GHCR_TOKEN are read from the pod's env (k8s secrets) — they never leave the cluster,
# and `\$GHCR_TOKEN` is escaped so the host shell doesn't interpolate it here.
kubectl exec -i "$POD" -- sh -eu -c "
  cd /build/repo
  printf '%s' \"\$GHCR_TOKEN\" | docker login ghcr.io -u \"\$GHCR_USER\" --password-stdin
  if ! docker buildx inspect flamingo >/dev/null 2>&1; then
    docker buildx create --name flamingo \
      --driver remote tcp://${BUILDKIT_ADDRESS}:1234 \
      --driver-opt=cacert=${CERT_ROOT}/ca.crt \
      --driver-opt=cert=${CERT_ROOT}/tls.crt \
      --driver-opt=key=${CERT_ROOT}/tls.key
  fi
  docker buildx build \
    --builder flamingo \
    --platform ${PLATFORM} \
    --file ${DOCKERFILE} \
    ${diffusion_args} \
    ${base_args} \
    --secret id=GH_TOKEN,env=GH_TOKEN \
    ${cache_args} \
    --tag ${IMAGE_REF} \
    --push \
    .
"

echo "▶ Done: pushed $IMAGE_REF"
