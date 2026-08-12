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
# Usage: invoked by `make image-build-remote`. Override via env: IMAGE_REF, PLATFORM, BASE_IMAGE,
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

# ---- job discovery / follow helpers ----------------------------------------------------------
# Discovers THIS tool's builder jobs only.
BUILD_JOB_SELECTOR="app.kubernetes.io/name=megatron-bridge-image-build,app.kubernetes.io/component=local-dev"
# Detached-run bookkeeping files written under /build on the pod.
LOGF_NAME=".mbridge-build.log"
EXITF_NAME=".mbridge-build.exit"
RUNNER_NAME=".mbridge-build.sh"

list_build_jobs() {
  kubectl get jobs -l "$BUILD_JOB_SELECTOR" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\t"}{.metadata.creationTimestamp}{"\n"}{end}' \
    2>/dev/null
}

# Echo the chosen job name (empty if none). JOB= wins; one job auto-selects; several -> interactive
# menu on a TTY, else an error listing the names.
select_build_job() {
  local names=() line n
  while IFS= read -r line; do
    n="${line%%	*}"
    [ -n "$n" ] && names+=("$n")
  done < <(list_build_jobs)
  if [ "${#names[@]}" -eq 0 ]; then return 0; fi
  if [ -n "${JOB:-}" ]; then printf '%s\n' "$JOB"; return 0; fi
  if [ "${#names[@]}" -eq 1 ]; then printf '%s\n' "${names[0]}"; return 0; fi
  if [ -t 0 ] && [ -t 2 ]; then
    >&2 echo "Multiple builder jobs — pick one:"
    local choice
    select choice in "${names[@]}"; do
      [ -n "$choice" ] && { printf '%s\n' "$choice"; return 0; }
    done
  fi
  >&2 echo "Multiple builder jobs found; re-run with JOB=<name>:"
  local m; for m in "${names[@]}"; do >&2 echo "    $m"; done
  return 1
}

pod_of_job() { kubectl get pods -l job-name="$1" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }

# Follow a detached build's log, printing only new lines, until the exit sentinel appears; returns
# the build's exit code. Each step is a short independent `kubectl exec`, so a transient connection
# blip just skips a poll and resumes — the build runs on the pod regardless of this follower. This
# is the whole point of detaching: a ~90 min build previously died with the streaming exec.
follow_logs() {
  local pod="$1" logf="$2" exitf="$3" printed=0 total rc phase gone=0
  while :; do
    total="$(kubectl exec "$pod" -- sh -c "wc -l < '$logf' 2>/dev/null" 2>/dev/null | tr -dc '0-9' || true)"
    total="${total:-0}"
    if [ "$total" -gt "$printed" ] 2>/dev/null; then
      kubectl exec "$pod" -- sed -n "$((printed + 1)),${total}p" "$logf" 2>/dev/null || true
      printed="$total"
    fi
    rc="$(kubectl exec "$pod" -- sh -c "cat '$exitf' 2>/dev/null" 2>/dev/null | tr -dc '0-9' || true)"
    if [ -n "$rc" ]; then
      kubectl exec "$pod" -- sed -n "$((printed + 1)),\$p" "$logf" 2>/dev/null || true
      return "$rc"
    fi
    # Stop polling a pod that died without writing the sentinel (evicted / OOM / lost node).
    # Two consecutive observations, so a transient `kubectl get` blip can't end the follow early.
    phase="$(kubectl get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    case "${phase:-gone}" in
      Running|Pending|Unknown) gone=0 ;;
      *)
        gone=$((gone + 1))
        if [ "$gone" -ge 2 ]; then
          echo "▶ pod $pod is no longer running (phase='${phase:-gone}') and wrote no exit code —"
          echo "    the build was killed mid-flight. Investigate with:"
          echo "    kubectl describe pod $pod | grep -iE 'reason|message|evict|oom'"
          return 137
        fi
        ;;
    esac
    sleep 5
  done
}

# ---- action dispatch ---------------------------------------------------------------------------
ACTION="${1:-${ACTION:-run}}"
case "$ACTION" in run|logs|teardown|list) ;; *) echo "ERROR: unknown action '$ACTION' (run|logs|teardown|list)."; exit 1 ;; esac

if [ "$ACTION" = "list" ]; then
  rows="$(list_build_jobs)"
  if [ -z "$rows" ]; then echo "No builder jobs running."; exit 0; fi
  echo "Builder jobs (follow: make image-build-logs JOB=<name> ; stop: make image-build-teardown JOB=<name>):"
  printf '%s\n' "$rows" | awk -F'\t' 'NF{printf "  %-44s active=%-4s created=%s\n",$1,($2==""?"0":$2),$3}'
  exit 0
fi

if [ "$ACTION" = "teardown" ]; then
  if [ "${JOB:-}" = "all" ]; then
    echo "▶ Tearing down ALL builder jobs"
    kubectl delete jobs -l "$BUILD_JOB_SELECTOR" --ignore-not-found
    exit 0
  fi
  j="$(select_build_job)" || exit 1
  [ -z "$j" ] && { echo "No builder jobs to tear down."; exit 0; }
  echo "▶ Tearing down builder job $j"
  kubectl delete job "$j" --ignore-not-found
  exit 0
fi

if [ "$ACTION" = "logs" ]; then
  j="$(select_build_job)" || exit 1
  [ -z "$j" ] && { echo "No builder jobs to follow."; exit 0; }
  pod="$(pod_of_job "$j")"
  [ -z "$pod" ] && { echo "ERROR: no pod found for job $j (still scheduling, or torn down)."; exit 1; }
  echo "▶ Following $j ($pod) — Ctrl-C stops following; the build keeps running."
  rc=0; follow_logs "$pod" "/build/$LOGF_NAME" "/build/$EXITF_NAME" || rc=$?
  echo "▶ Build on $j finished (rc=$rc)."
  exit "$rc"
fi

# ===================================  ACTION = run  ===========================================

# ---- identity + config (overridable via env) -------------------------------------------------
# Your cluster username, used to locate your per-user secret. Priority: explicit FLAMINGO_USERNAME,
# else parsed from `kubectl auth whoami` (e.g. oidc:you@far.ai -> you).
if [ -z "${FLAMINGO_USERNAME:-}" ]; then
  FLAMINGO_USERNAME="$(kubectl auth whoami -o jsonpath='{.status.userInfo.username}' 2>/dev/null \
    | sed -e 's/^[^:]*://' -e 's/@.*//' || true)"
fi

# Random DNS-1123-safe suffix so concurrent/detached builds don't collide on one job name.
rand_suffix() {
  local chars=abcdefghijklmnopqrstuvwxyz0123456789 s=""
  for _ in 1 2 3 4 5; do s+="${chars:RANDOM%36:1}"; done
  printf '%s' "$s"
}
JOB_NAME="${JOB_NAME:-${FLAMINGO_USERNAME:-${USER:-mbridge}}-mbridge-build-$(rand_suffix)}"
USER_LABEL="${FLAMINGO_USERNAME:-unknown}"
PRIORITY="${PRIORITY:-interactive}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/alignmentresearch/megatron-bridge}"
# Tags to publish. Normally supplied by `make image-build-remote` as a space-separated TAG_LIST
# (branch + sha [+ ci/latest when promoting]); standalone runs fall back to deriving them here so
# the script is usable without make. See the tagging block in the Makefile for the full scheme.
if [ -z "${TAG_LIST:-}" ]; then
  _sha="$(git -C "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" rev-parse --short=9 HEAD 2>/dev/null || true)"
  git diff --quiet HEAD 2>/dev/null || _sha="${_sha}-dirty"
  _branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-' | tr -cd '[:alnum:]._-' || true)"
  TAG_LIST="$(printf '%s\n%s\n' "$_branch" "$_sha" | grep -v '^$' | sort -u | tr '\n' ' ')"
fi
IMAGE_REF="${IMAGE_REF:-${IMAGE_REPO}:$(printf '%s' "$TAG_LIST" | awk '{print $1}')}"
PLATFORM="${PLATFORM:-linux/amd64}"
# Note: `-` not `:-` so an explicitly empty CACHE_REF (the documented "disable cache" knob,
# forwarded by `make image-build-remote CACHE_REF=`) is honored rather than reset to the default.
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

echo "▶ Building$(for t in $TAG_LIST; do printf ' %s:%s' "$IMAGE_REPO" "$t"; done)"
echo "    dockerfile : $DOCKERFILE (working tree)"
echo "    source     : the synced working tree IS the build context (no git ref; uncommitted changes included)"
echo "    platform   : $PLATFORM"
echo "    base image : ${BASE_IMAGE:-<Dockerfile.ci default>}"
echo "    push creds : secret ${GHCR_SECRET_NAME}[${GHCR_SECRET_KEY}] as ghcr.io user '$GHCR_USER'"
echo "    job/context: $JOB_NAME @ $(kubectl config current-context)"

# ---- teardown only on EARLY failure ----------------------------------------------------------
# Once the build is launched DETACHED on the pod it owns its lifecycle: a Ctrl-C or dropped
# connection must NOT kill it (re-attach with `logs`, clean up with `teardown`). So the trap only
# tears the Job down if we exit before that hand-off (e.g. the pod never scheduled).
LAUNCHED=0
cleanup() {
  [ "$LAUNCHED" = 1 ] && return 0
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
    -e "s|__USER_LABEL__|${USER_LABEL}|g" \
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

# Every tag in one push, so they land atomically — no window where the image exists but a moving
# tag still points at the previous build.
tag_args=""
for _t in $TAG_LIST; do
  tag_args="${tag_args} --tag ${IMAGE_REPO}:${_t}"
done

# Write the build to a script on the pod, then start it under setsid with its own log and
# exit-code sentinel. setsid + redirected stdio means the build outlives both this `kubectl exec`
# returning AND any later disconnect — a ~90 min build streamed over one exec dies on any network
# blip, which is exactly what used to happen at stage 10/12.
#
# GH_TOKEN / GHCR_TOKEN are read from the pod's env (k8s secrets) — they never leave the cluster,
# and `\$GHCR_TOKEN` is escaped so the host shell doesn't interpolate it here.
# Write via `tee`, NOT `sh -c "cat > file"`. This pod is alpine (docker:27-cli), whose busybox ash
# consumes the piped stdin while parsing, so cat sees EOF and writes a ZERO-BYTE file. `sh` then
# runs an empty script, exits 0 instantly, and the run looks like an instant success while nothing
# was built. tools/test_on_flamingo.sh gets away with the cat form because its pod is Ubuntu-based
# with a different shell.
echo "▶ Writing build runner to $POD:/build/$RUNNER_NAME"
kubectl exec -i "$POD" -- tee "/build/$RUNNER_NAME" >/dev/null <<RUNNER
set -eu
cd /build/repo
printf '%s' "\$GHCR_TOKEN" | docker login ghcr.io -u "\$GHCR_USER" --password-stdin
if ! docker buildx inspect flamingo >/dev/null 2>&1; then
  docker buildx create --name flamingo \\
    --driver remote tcp://${BUILDKIT_ADDRESS}:1234 \\
    --driver-opt=cacert=${CERT_ROOT}/ca.crt \\
    --driver-opt=cert=${CERT_ROOT}/tls.crt \\
    --driver-opt=key=${CERT_ROOT}/tls.key
fi
docker buildx build \\
  --builder flamingo \\
  --platform ${PLATFORM} \\
  --file ${DOCKERFILE} \\
  ${diffusion_args} \\
  ${base_args} \\
  --secret id=GH_TOKEN,env=GH_TOKEN \\
  ${cache_args} \\
  ${tag_args} \\
  --push \\
  .
RUNNER

# Fail fast if the runner did not land or is not valid shell. An empty runner exits 0 immediately,
# which would otherwise be indistinguishable from a successful build — the exact failure this
# guards against.
_runner_bytes="$(kubectl exec "$POD" -- wc -c "/build/$RUNNER_NAME" 2>/dev/null | tr -dc '0-9' | head -c 12 || true)"
if [ "${_runner_bytes:-0}" -lt 100 ]; then
  echo "ERROR: build runner did not land on the pod (/build/$RUNNER_NAME is ${_runner_bytes:-0} bytes)."
  echo "       Refusing to launch — an empty runner exits 0 and looks like a successful build."
  exit 1
fi
if ! kubectl exec "$POD" -- sh -n "/build/$RUNNER_NAME" 2>/dev/null; then
  echo "ERROR: build runner on the pod is not valid shell — refusing to launch."
  exit 1
fi

LOGF="/build/$LOGF_NAME"
EXITF="/build/$EXITF_NAME"
echo "▶ Launching build (detached on the pod — survives disconnects)"
kubectl exec "$POD" -- sh -c "
  cd /build && rm -f '$EXITF' '$LOGF'
  setsid sh -c 'sh \"$RUNNER_NAME\" > \"$LOGF_NAME\" 2>&1; echo \$? > \"$EXITF_NAME\"' \
    </dev/null >/dev/null 2>&1 &
  sleep 1   # let setsid fork into its own session before this exec session closes
"
LAUNCHED=1   # the build now owns its lifecycle; the EXIT trap no longer tears the Job down

if [ "${FOLLOW:-1}" != "1" ]; then
  echo "▶ Launched in the background as job $JOB_NAME (FOLLOW=0)."
  echo "    follow:   make image-build-logs JOB=$JOB_NAME"
  echo "    teardown: make image-build-teardown JOB=$JOB_NAME"
  exit 0
fi

echo "▶ Following build output — Ctrl-C stops following (build keeps running; re-attach: make image-build-logs JOB=$JOB_NAME)"
rc=0; follow_logs "$POD" "$LOGF" "$EXITF" || rc=$?

# Do not trust the exit code alone: confirm every tag actually resolves in the registry before
# claiming success. A build that silently does nothing still exits 0, and callers (and humans)
# reasonably treat "Done: pushed" as ground truth. Run from inside the pod, which already holds
# the GHCR credentials from the runner's `docker login`.
if [ "$rc" -eq 0 ]; then
  for _t in $TAG_LIST; do
    if ! kubectl exec "$POD" -- docker buildx imagetools inspect "${IMAGE_REPO}:${_t}" >/dev/null 2>&1; then
      echo "ERROR: build exited 0 but ${IMAGE_REPO}:${_t} is not in the registry."
      echo "       Treating this as a failure; inspect the build log with:"
      echo "       kubectl exec $POD -- cat $LOGF"
      rc=1
      break
    fi
  done
fi

if [ "$rc" -eq 0 ]; then
  echo "▶ Done: pushed$(for t in $TAG_LIST; do printf ' %s:%s' "$IMAGE_REPO" "$t"; done)"
else
  echo "▶ Build failed (rc=$rc)"
fi
if [ "${KEEP:-0}" = "1" ]; then
  echo "▶ KEEP=1 — leaving job $JOB_NAME up (teardown: make image-build-teardown JOB=$JOB_NAME)"
else
  kubectl delete job "$JOB_NAME" --ignore-not-found --wait=false >/dev/null 2>&1 || true
fi
exit "$rc"
