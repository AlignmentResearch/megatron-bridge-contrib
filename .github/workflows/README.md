## GitHub Actions Workflows

This fork's CI is five workflows. Two are inherited from upstream; three are FAR.AI additions.
See [README.farai.md](../../README.farai.md) for why the other 24 upstream workflows were removed.

| Workflow | Trigger | Role |
|---|---|---|
| `pre-commit.yml` | every PR / push to `farai/main` | lint + format (all pre-commit hooks) |
| `gpu-tests-dispatch.yml` | `@flamingo run …` PR comment | validates the comment and dispatches the requested workflow |
| `gpu-tests.yml` | dispatched | runs the unit-test suites on the H100 cluster; owns the `gpu-tests/<slug>` commit status |
| `gpu-tests-gate.yml` | every PR | auto-passes `gpu-tests/default` for PRs that only touch `.testignore`-matched paths |
| `detect-secrets.yml` | every PR | upstream secret scanning (uses `config/.secrets.baseline`) |
| `link-check.yml` | every PR, weekly | upstream external-link checking over `docs/**/*.md` |

### pre-commit

Runs **all** hooks in `.pre-commit-config.yaml` against the whole tree (ruff, ruff-format, the
`*.py` file fixers, and the local markdown-filename check). Local parity:

```sh
make lint          # or: pre-commit run --all-files
```

Note this invokes `pre-commit` directly rather than `uv run --group dev pre-commit`. The dev
group pulls `nvidia-resiliency-ext`, which publishes Linux-only wheels and cannot install on
macOS/arm64; no hook needs that environment.

### GPU tests: triggering a run

`gpu-tests-dispatch.yml` is the default-branch comment listener. To run a workflow from a PR
branch, leave a new top-level PR comment in this format:

```text
@flamingo run <workflow> [args...]
```

The first token after `run` is the workflow to dispatch (a file in `.github/workflows/`, with or
without the `.yml`/`.yaml` extension, as it exists on the PR branch). Everything after it is
passed verbatim to the workflow as the `wf_args` input. For `gpu-tests` the valid forms are:

```text
@flamingo run gpu-tests             # both unit suites  -> gpu-tests/default
@flamingo run gpu-tests core        # core only         -> gpu-tests/core
@flamingo run gpu-tests diffusion   # diffusion only    -> gpu-tests/diffusion
```

Any other first argument makes `gpu-tests.yml` **fail loudly**, so a typo'd suite name can't
silently downgrade to a narrower run.

**Who can trigger:** only comments whose author's GitHub association is `OWNER`, `MEMBER`, or
`COLLABORATOR` are honored. The workflow also ignores edited comments, non-PR issue comments, and
comments that do not start with `@flamingo run `. Forked PR branches are refused (the dispatched
workflow uses repository secrets). As feedback the dispatcher reacts to the comment: 👍 when the
workflow was dispatched, 👎 when it could not be (bad comment, forked PR, or the
`workflow_dispatch` API call failed — e.g. the target workflow isn't registered yet; see below).
Because the comment lives on a PR, this requires `pull-requests: write` (`issues: write` alone is
not sufficient for reactions on PR comments).

GitHub only evaluates `issue_comment` workflows from the repository default branch, so
`gpu-tests-dispatch.yml` must exist on `farai/main` before comments can trigger it. The
dispatcher is intentionally small: it validates the comment, resolves the PR head branch/SHA, and
starts the requested workflow with `workflow_dispatch` at the PR branch ref, passing `wf_args`,
`pr_number`, `head_sha`, and `head_repo`. It writes **no** commit status; the dispatched workflow
owns its own pending → success/failure/error status against `head_sha`.

### Suites

| Invocation | What runs | Script | Commit status |
|---|---|---|---|
| `@flamingo run gpu-tests` | both suites, as independent steps | both scripts below | `gpu-tests/default` |
| `… gpu-tests core` | everything under `tests/unit_tests` except `diffusion/` | `tests/unit_tests/Launch_Unit_Tests_Core.sh` | `gpu-tests/core` |
| `… gpu-tests diffusion` | `tests/unit_tests/diffusion` | `tests/unit_tests/Launch_Unit_Tests_Diffusion.sh` | `gpu-tests/diffusion` |

Each suite gets its **own** status context, so a single-suite run does **not** satisfy the
required `gpu-tests/default` context — they are opt-in speed-ups while iterating; merge still
needs a full run (or the gate, below). Any `gpu-tests/<slug>` context can be added to branch
protection / rulesets as a required check.

Both launchers pin `CUDA_VISIBLE_DEVICES="0,1"` internally, so the pod requests 2 GPUs.

To run the same suites **before** opening a PR, use `make test-unit-remote` — it rsyncs your local
working tree onto an ephemeral GPU pod on the same cluster and runs the same launchers, so it
catches what CI would catch without burning a PR round-trip. Its pods are labelled
`component: local-dev` (CI's are `component: ci`), and the two never interfere. See
[README.farai.md](../../README.farai.md#5-remote-test-runner). `make test-unit` runs them
directly on the current machine, which requires 2 local GPUs.

**Not run here:** `tests/functional_tests/` (65 H100 launch scripts) and the entire `gb200/`
tree. The functional suites reference NVIDIA's pre-populated `/home/TestData` volume, and GB200
is ARM64 hardware we do not have.

### GPU tests gate & `.testignore`

`gpu-tests-gate.yml` runs on every PR and auto-marks the required `gpu-tests/default` status
**successful** when *every* changed file matches a pattern in the top-level `.testignore`
(gitignore syntax) — so documentation-only / non-code PRs merge without dispatching a GPU run. If
**any** changed file is unmatched, it writes no status and a real GPU run is still required.

- **Security rule:** `.testignore` is read from the PR's **BASE branch**, not the PR head — a PR
  cannot exempt its own files by editing `.testignore` in the same PR; changes to it only take
  effect once merged.
- **Keep entries conservative:** anything that can affect runtime behavior or the unit suites
  must NOT be listed.
- **Sync obligation:** the gate's `REQUIRED_CONTEXTS` list (currently just `gpu-tests/default`)
  must be kept in sync with the contexts marked required in branch protection / rulesets — a
  required context the gate doesn't know about would block ignore-only PRs forever; one it marks
  but isn't required is harmless.

### What `gpu-tests.yml` does

It runs the unit-test suites against the PR commit on the H100 Kubernetes cluster:

1. Computes the slug/name and status context from `wf_args` (suite validation happens here).
2. Writes a pending `gpu-tests/<slug>` commit status to the PR head SHA.
3. Checks out the PR head commit on the GitHub-hosted runner and installs/configures `kubectl`.
4. Creates a Kubernetes job from `.github/k8s/pytest-multigpu-job.yaml` and waits for readiness.
5. Clones Megatron-Bridge (with submodules) at the PR commit inside the pod, into
   `/opt/megatron-bridge-ci` — deliberately not `/opt/Megatron-Bridge`, where the image bakes its
   build-time copy of the tree.
6. Runs the requested suite(s), each as its own step so they report independently.
7. Copies `.coverage` back and uploads it as the `<slug>-test-outputs` artifact.
8. Tears down the Kubernetes job and marks the status success/failure/error.

For same-repository PRs, edits to `gpu-tests.yml` and `.github/k8s/` are picked up from the PR
branch.

#### Hugging Face access

Unlike upstream, these runs do **not** set `TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE`. A static
audit of `tests/unit_tests` found exactly one live Hub call — `training/test_tokenizer.py`
downloads the ungated `bert-base-uncased` tokenizer (a few hundred KB, no weights). Every other
`from_pretrained` / `snapshot_download` / `load_dataset` site is mocked, and the real-looking
model IDs in the recipe tests are config strings that are never loaded. So no `HF_TOKEN` and no
seeded cache are required. If Hub outages ever make this flaky, bake that tokenizer into the
image and restore the offline environment variables.

#### Dispatching workflows that only exist on a PR branch

`workflow_dispatch` via the REST API only works once GitHub has **registered** the workflow,
which happens after it has run at least once from any trigger. A brand-new workflow file added
only on a PR branch is not registered yet, so the API would return 404 and the dispatcher would
react 👎.

To make a workflow dispatchable from its own PR (before it lands on the default branch), give it
a self-registration trigger and guard the real jobs so they only run on dispatch:

```yaml
on:
  workflow_dispatch:
    inputs: { ... }
  pull_request:
    paths:
      - .github/workflows/<this-file>.yml

jobs:
  <job>:
    if: github.event_name == 'workflow_dispatch'
    ...
```

When a PR adds or edits the file, the `pull_request` event creates a run with all jobs skipped
(near-zero cost), which registers the workflow. After that registering run completes,
`@flamingo run <workflow> ...` can dispatch the PR branch's version. `gpu-tests.yml` uses exactly
this pattern.

### CI secrets

`gpu-tests.yml` requires these repository (or organization) **Actions secrets**
(Settings → Secrets and variables → Actions):

| Secret | What it is | Notes |
|---|---|---|
| `K8S_TOKEN` | token for the `ci-job-manager` service account in the `ci` namespace of the H100 cluster (RBAC-scoped to managing test jobs) | regenerate with `kubectl create token ci-job-manager -n ci --duration=<…>` and update the secret when it expires |
| `CI_GITHUB_PAT` | PAT used to clone this private fork (and the `AlignmentResearch/flamingo` kubeconfig repo) inside the run | repo read scope on both repositories |

No `HF_TOKEN` is needed — see [Hugging Face access](#hugging-face-access) above.

### Container image

The pod runs `ghcr.io/alignmentresearch/megatron-bridge:latest`, built from upstream's
`docker/Dockerfile.ci`. The diffusion unit suite needs hash-locked WAN codecs: bases that gate
those behind `ARG INSTALL_DIFFUSION_DEPS` (off by default, so the codecs stay out of shipped
framework images) get the arg passed automatically, while older bases install them unconditionally
via `uv sync --all-extras --all-groups`. The build tooling probes the Dockerfile and passes the
flag only when it exists. Apart from an rsync layer there is no fork-specific Dockerfile. Build
and push with:

```sh
make image-remote         # on the cluster (preferred)
make image-local          # see `make help` for IMAGE_TAG / BASE_IMAGE / CACHE_REF
```

The pod also needs the `docker` imagePullSecret in the `ci` namespace to have read access to that
GHCR package.
