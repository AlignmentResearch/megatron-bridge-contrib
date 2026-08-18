#!/usr/bin/env bash
#
# Run Megatron-Bridge unit tests on the FAR.AI cluster against your LOCAL working tree.
#
# This mirrors the gpu-tests CI workflow (.github/workflows/gpu-tests.yml): it uses the same
# container image and the same launcher scripts, on a pod with TEST_GPUS GPUs (default 2 = CI
# parity, and what both launchers pin via CUDA_VISIBLE_DEVICES="0,1"). Instead of cloning the
# repo at a PR commit it rsyncs your local working tree into the pod, so you can test
# uncommitted changes.
#
# Actions (first arg, or ACTION=; default "run"):
#   run       create an ephemeral GPU Job, sync the tree, then run the tests DETACHED on the pod
#             (so a dropped laptop connection can't kill them) and follow the output. On normal
#             completion the Job is torn down (KEEP=1 leaves it up; FOLLOW=0 launches and returns
#             without following). If the follow is interrupted, the tests keep running — re-attach
#             with `logs` or clean up with `teardown`.
#   shell     provision the same 2-GPU test pod (or REUSE one this action created earlier — the
#             job name is stable per user), sync the tree, and attach an interactive shell for
#             iterative test runs. Re-running `shell` re-syncs the current working tree into the
#             live pod, so edit-locally/run-remotely loops need no new pod. Without a TTY it
#             prints the `kubectl exec` invocation instead of attaching. The pod stays up until
#             `teardown` — nothing is auto-torn-down on exit.
#   logs      follow a running test job's output (reconnect-resilient).
#   teardown  delete a test job (JOB=all deletes them all).
#   list      list this user's local-dev test jobs.
#
# Job selection for logs/teardown: JOB=<name> picks it explicitly; with exactly one job it's
# auto-selected; with several, an interactive `select` menu (or pass JOB=).
#
# Suites (TEST_SUITE): empty = both (like a no-arg `@flamingo run gpu-tests`); "core" or
# "diffusion" runs just that launcher.
#
# Used by `make test-unit-remote` / `test-unit-{core,diffusion}-remote` / `test-logs` /
# `test-teardown` / `test-list`. Override via env: TEST_SUITE, IMAGE_REF, JOB_NAME / JOB,
# PRIORITY, READY_TIMEOUT, TEST_GPUS, PYTEST_K, KEEP, FOLLOW.
set -euo pipefail

CONTAINER="${CONTAINER:-test-container}"
# Discovers THIS tool's jobs (component=local-dev) — never the CI jobs (component=ci).
TEST_JOB_SELECTOR="app.kubernetes.io/name=megatron-bridge-gpu-tests,app.kubernetes.io/component=local-dev"
# Detached-run bookkeeping files written under REMOTE_DIR on the pod.
LOGF_NAME=".mbridge-test.log"
EXITF_NAME=".mbridge-test.exit"
RUNNER_NAME=".mbridge-runtests.sh"

# ---- cluster preflight -----------------------------------------------------------------------
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found on PATH."; exit 1; }
# Namespaced reachability check — `kubectl cluster-info` reads cluster-scoped resources that a
# namespaced user often can't, so it false-negatives even when the cluster is reachable.
kubectl get pods >/dev/null 2>&1 || {
  echo "ERROR: kubectl can't reach the cluster or access the current namespace (context: $(kubectl config current-context 2>/dev/null || echo none))."
  exit 1
}

# ---- job discovery / selection / follow helpers ---------------------------------------------
# One line per local-dev test job: "<name>\t<active>\t<created>".
list_test_jobs() {
  kubectl get jobs -l "$TEST_JOB_SELECTOR" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\t"}{.metadata.creationTimestamp}{"\n"}{end}' \
    2>/dev/null
}

# Echo the chosen job name (empty if none). JOB= wins; one job auto-selects; several -> interactive
# `select` menu on a TTY, else an error listing the names so the caller can pass JOB=.
select_test_job() {
  local names=() line n
  while IFS= read -r line; do
    n="${line%%	*}"   # field before the first TAB
    [ -n "$n" ] && names+=("$n")
  done < <(list_test_jobs)
  if [ "${#names[@]}" -eq 0 ]; then return 0; fi
  if [ -n "${JOB:-}" ]; then printf '%s\n' "$JOB"; return 0; fi
  if [ "${#names[@]}" -eq 1 ]; then printf '%s\n' "${names[0]}"; return 0; fi
  if [ -t 0 ] && [ -t 2 ]; then
    >&2 echo "Multiple test jobs — pick one:"
    local choice
    select choice in "${names[@]}"; do
      [ -n "$choice" ] && { printf '%s\n' "$choice"; return 0; }
    done
  fi
  >&2 echo "Multiple test jobs found; re-run with JOB=<name>:"
  local m; for m in "${names[@]}"; do >&2 echo "    $m"; done
  return 1
}

pod_of_job() { kubectl get pods -l job-name="$1" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }

# Follow a detached run's log, printing only new lines, until the exit sentinel appears; returns
# the test exit code. Each step is a short, independent `kubectl exec`, so a transient connection
# blip just skips a poll and resumes — the tests run on the pod regardless of this follower.
follow_logs() {
  local pod="$1" logf="$2" exitf="$3" printed=0 total rc phase gone=0
  while :; do
    total="$(kubectl exec "$pod" -c "$CONTAINER" -- sh -c "wc -l < '$logf' 2>/dev/null" 2>/dev/null | tr -dc '0-9' || true)"
    total="${total:-0}"
    if [ "$total" -gt "$printed" ] 2>/dev/null; then
      kubectl exec "$pod" -c "$CONTAINER" -- sed -n "$((printed + 1)),${total}p" "$logf" 2>/dev/null || true
      printed="$total"
    fi
    rc="$(kubectl exec "$pod" -c "$CONTAINER" -- sh -c "cat '$exitf' 2>/dev/null" 2>/dev/null | tr -dc '0-9' || true)"
    if [ -n "$rc" ]; then
      kubectl exec "$pod" -c "$CONTAINER" -- sed -n "$((printed + 1)),\$p" "$logf" 2>/dev/null || true
      return "$rc"
    fi
    # The sentinel is written only on a clean finish. If the pod has died WITHOUT it (evicted,
    # OOM-killed, node lost), there is nothing left to wait for — detect a terminal/absent phase
    # and stop, instead of polling a dead pod forever. Require two consecutive observations so a
    # transient `kubectl get` blip can't end the follow prematurely.
    phase="$(kubectl get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    case "${phase:-gone}" in
      Running|Pending|Unknown) gone=0 ;;
      *)
        gone=$((gone + 1))
        if [ "$gone" -ge 2 ]; then
          echo "▶ pod $pod is no longer running (phase='${phase:-gone}') and wrote no exit code —"
          echo "    the run was killed mid-flight (eviction / OOM / lost node). Investigate with:"
          echo "    kubectl describe pod $pod | grep -iE 'reason|message|evict|oom'"
          return 137
        fi
        ;;
    esac
    sleep 3
  done
}

# ---- action dispatch -------------------------------------------------------------------------
ACTION="${1:-${ACTION:-run}}"
case "$ACTION" in run|shell|logs|teardown|list) ;; *) echo "ERROR: unknown action '$ACTION' (run|shell|logs|teardown|list)."; exit 1 ;; esac

if [ "$ACTION" = "list" ]; then
  rows="$(list_test_jobs)"
  if [ -z "$rows" ]; then echo "No local-dev test jobs running."; exit 0; fi
  echo "Local-dev test jobs (follow: make test-logs JOB=<name> ; stop: make test-teardown JOB=<name>):"
  printf '%s\n' "$rows" | awk -F'\t' 'NF{printf "  %-44s active=%-4s created=%s\n",$1,($2==""?"0":$2),$3}'
  exit 0
fi

if [ "$ACTION" = "teardown" ]; then
  if [ "${JOB:-}" = "all" ]; then
    echo "▶ Tearing down ALL local-dev test jobs"
    kubectl delete jobs -l "$TEST_JOB_SELECTOR" --ignore-not-found
    exit 0
  fi
  j="$(select_test_job)" || exit 1
  [ -z "$j" ] && { echo "No test jobs to tear down."; exit 0; }
  echo "▶ Tearing down test job $j"
  kubectl delete job "$j" --ignore-not-found
  exit 0
fi

if [ "$ACTION" = "logs" ]; then
  j="$(select_test_job)" || exit 1
  [ -z "$j" ] && { echo "No test jobs to follow."; exit 0; }
  pod="$(pod_of_job "$j")"
  [ -z "$pod" ] && { echo "ERROR: no pod found for job $j (still scheduling, or torn down)."; exit 1; }
  rd="${REMOTE_DIR:-/opt/Megatron-Bridge}"
  echo "▶ Following $j ($pod) — Ctrl-C stops following; the tests keep running."
  rc=0; follow_logs "$pod" "$rd/$LOGF_NAME" "$rd/$EXITF_NAME" || rc=$?
  echo "▶ Tests on $j finished (rc=$rc)."
  exit "$rc"
fi

# ===================================  ACTION = run  ===========================================

# ---- identity + config (overridable via env) -------------------------------------------------
if [ -z "${FLAMINGO_USERNAME:-}" ]; then
  FLAMINGO_USERNAME="$(kubectl auth whoami -o jsonpath='{.status.userInfo.username}' 2>/dev/null \
    | sed -e 's/^[^:]*://' -e 's/@.*//' || true)"
fi

# Short random DNS-1123-safe suffix (lowercase alphanumeric) so each launch gets a unique job/pod
# name — lets multiple test jobs (e.g. different suites or branches) run side by side without
# colliding. Pure bash, no external deps.
rand_suffix() {
  local chars=abcdefghijklmnopqrstuvwxyz0123456789 s=""
  for _ in 1 2 3 4 5; do s+="${chars:RANDOM%36:1}"; done
  printf '%s' "$s"
}

# Default to <username>-mbridge-test-<rand>. An explicit JOB_NAME is used verbatim (no suffix).
# The shell action instead defaults to a STABLE per-user name so re-running it re-syncs into the
# same live pod (the iterative loop) instead of provisioning a second one.
if [ "$ACTION" = shell ]; then
  JOB_NAME="${JOB_NAME:-${FLAMINGO_USERNAME:-${USER:-mbridge}}-mbridge-shell}"
else
  JOB_NAME="${JOB_NAME:-${FLAMINGO_USERNAME:-${USER:-mbridge}}-mbridge-test-$(rand_suffix)}"
fi
PRIORITY="${PRIORITY:-interactive}"
IMAGE_REF="${IMAGE_REF:-ghcr.io/alignmentresearch/megatron-bridge:latest}"
MANIFEST="${MANIFEST:-k8s/test-pod.yaml}"
# Pod-ready timeout. A first-time pull of this (large NGC-based) image on a fresh node can take
# many minutes; if it exceeds this, kubectl wait fails and the trap tears the pod down
# (ContainerCreating -> Terminating). 30 min covers a cold pull.
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
# Image pull policy. Always, matching the CI pod. This is NOT a full re-download per run: the
# kubelet fetches the manifest and compares digests, reusing cached layers when they match, so
# the cost is a registry round-trip. IMAGE_REF is a MUTABLE tag (:latest) — under IfNotPresent a
# node that cached an older :latest would silently test stale dependencies, and two runs landing
# on different nodes could test different images, with nothing in the output to say so. The one
# real trade-off is that a registry outage or an expired pull secret fails the pod even when a
# usable image is cached locally; set IMAGE_PULL_POLICY=IfNotPresent to ride that out.
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-Always}"
# Where the working tree is synced inside the pod. We overlay it onto /opt/Megatron-Bridge — the
# image's WORKDIR, where Dockerfile.ci already copied the code and where uv's project environment
# (/opt/venv, via UV_PROJECT_ENVIRONMENT) is already resolved — so `uv run` reuses that venv
# instead of re-resolving. The rsync below has no --delete, so image artifacts are preserved,
# including 3rdparty/Megatron-LM: uv resolves megatron-core from that submodule, and a local
# checkout with an uninitialized submodule must not wipe the image's copy.
REMOTE_DIR="${REMOTE_DIR:-/opt/Megatron-Bridge}"

# GPU count for the test pod (default 2 = CI parity, and what both launchers pin internally via
# CUDA_VISIBLE_DEVICES="0,1"). CPU/memory scale with it. PYTEST_K, when set, becomes a pytest -k
# filter so you can target a single test without editing the launcher.
TEST_GPUS="${TEST_GPUS:-2}"
TEST_CPU="${TEST_CPU:-$(( TEST_GPUS * 4 ))}"
# Memory: 80G/GPU = 160G for 2 GPUs, matching the CI pod
# (.github/k8s/pytest-multigpu-job.yaml, memory limit 160G).
TEST_MEM="${TEST_MEM:-$(( TEST_GPUS * 80 ))G}"
# Ephemeral storage (request AND limit; see k8s/test-pod.yaml for why an explicit limit is
# required — the cluster injects a too-small 50Gi/GPU default otherwise). The unit suites are far
# lighter than the functional ones: no torch_dist checkpoints and a single small HF tokenizer
# download (see .github/workflows/README.md), so the dominant consumers are the synced tree, the
# submodule, and the run log. 100Gi matches the CI pod; still scaled per GPU for headroom.
_test_disk_gi=$(( TEST_GPUS * 50 )); [ "$_test_disk_gi" -lt 100 ] && _test_disk_gi=100
TEST_DISK="${TEST_DISK:-${_test_disk_gi}Gi}"
PYTEST_K="${PYTEST_K:-}"

# Unit suite: "" (both), "core", or "diffusion".
TEST_SUITE="${TEST_SUITE:-}"
case "$TEST_SUITE" in
  ""|core|diffusion) ;;
  *) echo "ERROR: TEST_SUITE must be empty, 'core', or 'diffusion' (got '$TEST_SUITE')."; exit 1 ;;
esac

# Filter labels applied to the Job and its pod (k8s label values can't be empty / must be
# DNS-safe). Filter with e.g. `kubectl get jobs -l megatron-bridge.farai/user=<you>`.
USER_LABEL="${FLAMINGO_USERNAME:-unknown}"
SUITE_LABEL="${TEST_SUITE:-both}"
[ "$ACTION" = shell ] && SUITE_LABEL="shell"

REPO_ROOT="$(git rev-parse --show-toplevel)"

# ---- remaining preflight ---------------------------------------------------------------------
test -f "$REPO_ROOT/$MANIFEST" || { echo "ERROR: $MANIFEST not found."; exit 1; }

echo "▶ Running Megatron-Bridge unit tests on the cluster"
echo "    suite      : $SUITE_LABEL"
echo "    image      : $IMAGE_REF"
echo "    job/context: $JOB_NAME @ $(kubectl config current-context)"
echo "    sync source: $REPO_ROOT (working tree) -> $REMOTE_DIR"

# ---- teardown only on EARLY failure ----------------------------------------------------------
# Once the tests are launched DETACHED on the pod they own their lifecycle: a Ctrl-C or dropped
# connection must NOT kill them (re-attach with `logs`, clean up with `teardown`). So the trap
# only tears the Job down if we exit before that hand-off (e.g. the pod never scheduled).
LAUNCHED=0
cleanup() {
  [ "$LAUNCHED" = 1 ] && return 0
  kubectl delete job "$JOB_NAME" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- create the test job (or, for shell, reuse a live one) -----------------------------------
# The shell action's job name is stable, so a running pod from an earlier `shell` is picked up
# and only re-synced — that is the iterative loop. A reused pod must never be torn down by the
# early-failure trap: it predates this invocation.
REUSED=0
if [ "$ACTION" = shell ]; then
  POD="$(kubectl get pods -l job-name="$JOB_NAME" --field-selector=status.phase=Running \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [ -n "$POD" ]; then
    REUSED=1
    LAUNCHED=1
    echo "▶ Reusing running shell pod $POD (job $JOB_NAME)"
  fi
fi

if [ "$REUSED" != 1 ]; then
  kubectl delete job "$JOB_NAME" --ignore-not-found >/dev/null 2>&1 || true
  sed -e "s|__JOB_NAME__|${JOB_NAME}|g" \
      -e "s|__PRIORITY__|${PRIORITY}|g" \
      -e "s|__IMAGE__|${IMAGE_REF}|g" \
      -e "s|__USER_LABEL__|${USER_LABEL}|g" \
      -e "s|__SUITE_LABEL__|${SUITE_LABEL}|g" \
      -e "s|__TEST_GPUS__|${TEST_GPUS}|g" \
      -e "s|__TEST_CPU__|${TEST_CPU}|g" \
      -e "s|__TEST_MEM__|${TEST_MEM}|g" \
      -e "s|__TEST_DISK__|${TEST_DISK}|g" \
      -e "s|__IMAGE_PULL_POLICY__|${IMAGE_PULL_POLICY}|g" \
      "$REPO_ROOT/$MANIFEST" | kubectl create -f -

  # ---- wait for the pod to be running --------------------------------------------------------
  echo "▶ Waiting for test pod (timeout ${READY_TIMEOUT}s; needs ${TEST_GPUS} GPUs via kueue)…"
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  POD=""
  while :; do
    POD="$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    [ -n "$POD" ] && break
    [ "$(date +%s)" -ge "$deadline" ] && {
      echo "ERROR: test pod never scheduled (kueue admission / GPU quota?). Check: kubectl describe job $JOB_NAME"
      exit 1
    }
    sleep 3
  done
  kubectl wait --for=condition=ready "pod/$POD" --timeout="${READY_TIMEOUT}s"
fi

# ---- sync the working tree into the pod ------------------------------------------------------
echo "▶ Syncing working tree into $POD:$REMOTE_DIR (rsync; incremental + resumable)"
kubectl exec "$POD" -c "$CONTAINER" -- mkdir -p "$REMOTE_DIR"
# Prefer rsync over a kubectl-exec transport (the pod has no SSH). REMOTE_DIR is the image's
# /opt/Megatron-Bridge, which ALREADY contains the large 3rdparty/Megatron-LM submodule — so an
# incremental rsync moves only changed files, with --timeout to fail fast on a stalled stream and
# a retry, rather than re-sending the whole tree over one non-resumable pipe. No --delete: we
# OVERLAY onto the image's code, preserving its submodule and any build artifacts.
_krsync="$(mktemp)"
cat > "$_krsync" <<KRS
#!/usr/bin/env bash
shift   # rsync passes a placeholder hostname as \$1; drop it and exec into the known pod
exec kubectl exec -i -c "$CONTAINER" "$POD" -- "\$@"
KRS
chmod +x "$_krsync"
# rsync must exist on BOTH ends: the transport above execs `rsync --server` inside the pod, so a
# local-only check would burn three attempts and 15s of sleeps before falling back. Dockerfile.ci
# does not install rsync, so whether this path is available depends entirely on the NGC base
# image — probe once and say which path we took, rather than discovering it from a stack of
# retry warnings.
_synced=0
_have_rsync=0
if ! command -v rsync >/dev/null 2>&1; then
  echo "▶ rsync not on PATH locally; using the tar stream"
elif ! kubectl exec "$POD" -c "$CONTAINER" -- sh -c 'command -v rsync >/dev/null 2>&1'; then
  echo "▶ rsync not present in the container image; using the tar stream"
else
  _have_rsync=1
fi
if [ "$_have_rsync" = 1 ]; then
  for _attempt in 1 2 3; do
    # .git is NOT excluded, deliberately: parts of the suite shell out to git — the
    # examples/**/slurm_conversion.sh wrappers run `git rev-parse --show-toplevel`, and
    # test_mcore_commit runs `git ls-tree HEAD 3rdparty/Megatron-LM`. CI gets a real repo because
    # it clones; without .git those tests fail with "not a git repository" (exit 128).
    # .claude/.agents are agent-tooling symlink farms (also inside the submodule); the image
    # carries REAL directories at those paths, which neither rsync nor tar can replace with a
    # symlink — and the tests never read them. Exclude at any depth.
    if rsync -rlptz --timeout=180 \
         --filter=':- .gitignore' --exclude='.venv' --exclude='__pycache__' \
         --exclude='.claude' --exclude='.agents' \
         -e "$_krsync" "$REPO_ROOT"/ "mbridge:$REMOTE_DIR"/; then
      _synced=1; break
    fi
    echo "▶ rsync sync attempt $_attempt stalled/failed; retrying in 5s…" >&2
    sleep 5
  done
fi
rm -f "$_krsync"
if [ "$_synced" != 1 ]; then
  echo "▶ Syncing via tar stream"
  # TWO passes, for the same reason as the image build: `git ls-files --cached` reports the
  # submodule as a single gitlink, so a one-pass list would ship an EMPTY 3rdparty/Megatron-LM.
  # Here that would not error — we extract over the image's own tree without --delete, so its
  # copy survives — but the pod would then silently test the IMAGE's submodule commit instead of
  # the one in your working tree. --recurse-submodules expands it; git refuses to combine that
  # with --others, so untracked files come from a second, disjoint pass.
  # COPYFILE_DISABLE=1 is load-bearing on macOS: bsdtar otherwise emits an AppleDouble "._<name>"
  # companion for every file carrying extended attributes. Those land in the repo tree on the pod,
  # and code that globs sources and calls read_text() — e.g. flat_perf_recipe_names() in
  # scripts/performance/utils/utils.py — then reads "._foo.py" and dies with
  # "UnicodeDecodeError: invalid start byte", failing ~20 otherwise-passing tests with an error
  # that points nowhere near the cause. Ignored by GNU tar, so it is safe to set unconditionally.
  #
  # The .git trees are included on purpose. Parts of the suite shell out to git — the
  # examples/**/slurm_conversion.sh wrappers run `git rev-parse --show-toplevel`, and
  # test_mcore_commit runs `git ls-tree HEAD 3rdparty/Megatron-LM`. CI gets a real repo because it
  # clones; without .git those tests fail with "not a git repository" (exit 128). git never lists
  # .git itself (neither --cached nor --others), so it has to be added explicitly. The submodule's
  # .git is a pointer FILE ("gitdir: ../../.git/modules/3rdparty/Megatron-LM") whose relative path
  # resolves correctly once .git/modules is in place, so both parts must ship together.
  # The grep drops .claude/.agents at any depth for the same reason as the rsync excludes
  # above: symlink farms colliding with real directories in the image.
  ( cd "$REPO_ROOT" && { git ls-files -z --cached --recurse-submodules; \
                         git ls-files -z --others --exclude-standard; \
                         find .git \( -type f -o -type l \) -print0; \
                         [ -e 3rdparty/Megatron-LM/.git ] && printf '3rdparty/Megatron-LM/.git\0'; } \
      | grep -zvE '(^|/)\.(claude|agents)(/|$)' \
      | COPYFILE_DISABLE=1 tar --null -T - -czf - ) \
    | kubectl exec -i "$POD" -c "$CONTAINER" -- tar --warning=no-unknown-keyword -xzf - -C "$REMOTE_DIR"
fi

# ---- shell action: attach (or print how to exec) instead of launching the suites -------------
if [ "$ACTION" = shell ]; then
  LAUNCHED=1   # the live pod is the deliverable; never torn down on exit — use `teardown`
  echo "▶ Shell pod ready: $POD (job $JOB_NAME, ${TEST_GPUS} GPUs)"
  echo "    re-sync tree   : re-run this action (pod is reused, rsync is incremental)"
  echo "    one-off command: kubectl exec $POD -c $CONTAINER -- bash -lc 'cd $REMOTE_DIR && <cmd>'"
  echo "    teardown       : JOB=$JOB_NAME bash tools/test_on_flamingo.sh teardown"
  if [ -t 0 ] && [ -t 1 ]; then
    exec kubectl exec -it "$POD" -c "$CONTAINER" -- bash -c "cd '$REMOTE_DIR' && exec bash -l"
  fi
  exit 0
fi

# ---- launch the tests DETACHED on the pod ----------------------------------------------------
# Write the test commands to a file on the pod, then start them under setsid with their own log +
# exit-code sentinel. setsid + redirected stdio means the run outlives both this `kubectl exec`
# returning AND any later disconnect.
echo "▶ Writing test runner to $POD:$REMOTE_DIR/$RUNNER_NAME"
kubectl exec -i "$POD" -c "$CONTAINER" -- sh -c "cat > '$REMOTE_DIR/$RUNNER_NAME'" <<'POD_SCRIPT'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export CI=true
# Stream test results live: unbuffered Python stdout. The launchers already pass -v/-s to pytest.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# PYTEST_K -> a -k filter to target specific tests. The launchers hardcode their pytest
# invocation, so this is injected via PYTEST_ADDOPTS, which pytest shlex-splits (the quoted
# expression survives as one argument).
[ -n "${PYTEST_K:-}" ] && export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -k \"${PYTEST_K}\""
cd "$REMOTE_DIR"

nvidia-smi || true

rc=0
# Report ephemeral usage after each suite so the pod's disk request/limit can be sized from data
# rather than guessed, and a future eviction is easy to attribute.
disk_report() {
  echo "  [disk] after: $1"
  printf '    %s\n' \
    "/tmp:                  $(du -sh /tmp 2>/dev/null | cut -f1)" \
    "$REMOTE_DIR:           $(du -sh "$REMOTE_DIR" 2>/dev/null | cut -f1)" 2>/dev/null || true
}
run() {
  echo "▶ $*"
  if ! "$@"; then rc=1; echo "  ✗ suite FAILED: $*"; fi
  disk_report "$*"
}

# The same launcher scripts the CI workflow runs, so local and CI stay in parity. Each pins
# CUDA_VISIBLE_DEVICES="0,1" internally.
case "${TEST_SUITE:-}" in
  core)      run bash tests/unit_tests/Launch_Unit_Tests_Core.sh ;;
  diffusion) run bash tests/unit_tests/Launch_Unit_Tests_Diffusion.sh ;;
  *)
    run bash tests/unit_tests/Launch_Unit_Tests_Core.sh
    run bash tests/unit_tests/Launch_Unit_Tests_Diffusion.sh
    ;;
esac

if [ "$rc" -ne 0 ]; then echo "✗ One or more suites failed."; else echo "✓ All suites passed."; fi
exit $rc
POD_SCRIPT

LOGF="$REMOTE_DIR/$LOGF_NAME"
EXITF="$REMOTE_DIR/$EXITF_NAME"
echo "▶ Launching tests (detached on the pod — survives disconnects)"
kubectl exec "$POD" -c "$CONTAINER" -- sh -c "
  cd '$REMOTE_DIR' && rm -f '$EXITF' '$LOGF'
  setsid env TEST_SUITE='${TEST_SUITE}' REMOTE_DIR='${REMOTE_DIR}' PYTEST_K='${PYTEST_K}' \
    sh -c 'bash \"$RUNNER_NAME\" > \"$LOGF_NAME\" 2>&1; echo \$? > \"$EXITF_NAME\"' \
    </dev/null >/dev/null 2>&1 &
  sleep 1   # let setsid fork into its own session before this exec session closes
"
LAUNCHED=1   # tests now own their lifecycle; the EXIT trap no longer tears the Job down

if [ "${FOLLOW:-1}" != "1" ]; then
  echo "▶ Launched in the background as job $JOB_NAME (FOLLOW=0)."
  echo "    follow:   make test-logs JOB=$JOB_NAME"
  echo "    teardown: make test-teardown JOB=$JOB_NAME"
  exit 0
fi

echo "▶ Following test output — Ctrl-C stops following (tests keep running; re-attach: make test-logs JOB=$JOB_NAME)"
rc=0; follow_logs "$POD" "$LOGF" "$EXITF" || rc=$?   # || …: a non-zero test exit must not trip set -e before teardown
echo "▶ Tests finished (rc=$rc, suite: $SUITE_LABEL)"
if [ "${KEEP:-0}" = "1" ]; then
  echo "▶ KEEP=1 — leaving job $JOB_NAME up (teardown: make test-teardown JOB=$JOB_NAME)"
else
  echo "▶ Tearing down test job $JOB_NAME"
  kubectl delete job "$JOB_NAME" --ignore-not-found --wait=false >/dev/null 2>&1 || true
fi
exit "$rc"
