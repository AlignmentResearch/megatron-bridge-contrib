# FAR.AI Fork of NVIDIA-NeMo/Megatron-Bridge

This repository is FAR.AI's private fork of
[NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), a PyTorch-native bridge between Hugging
Face and Megatron-Core. The fork tracks the upstream `main` branch while carrying FAR.AI-specific patches on top.

**Getting started:** everything fork-specific is on this page. Upstream documentation
([README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/](docs/)) still applies unchanged unless a row in
[FAR.AI Patches](#farai-patches) says otherwise.


## Why We Fork

Our fork extends NVIDIA's upstream repo in two categories:

**Infrastructure patches** — changes needed to develop and test on FAR.AI's Kubernetes cluster instead of NVIDIA's
internal CI fleet.

**Research patches** — experimental training techniques under active development at FAR.AI that are not yet suitable
for upstream contribution.

The patch set here is deliberately small. Megatron-Bridge moves fast upstream, and every delta is a merge conflict
waiting to happen, so prefer configuration and out-of-tree code over in-tree edits wherever possible.


## Branch Structure

| Branch | Role |
|---|---|
| `farai/main` | **Default branch.** Upstream `main` plus the FAR.AI patches below. |
| `main` | Clean mirror of upstream `NVIDIA-NeMo/Megatron-Bridge@main`. Do not commit here. |

New work branches off `farai/main` and merges back via pull request. Upstream changes are periodically merged from
`main` into `farai/main` — see [Keeping the Fork in Sync](#keeping-the-fork-in-sync-with-upstream).


## FAR.AI Patches

Every deliberate divergence from upstream, newest last. Elaboration follows the table for rows that need it.

| # | Change | Type | Files | Landed |
|---|---|---|---|---|
| 1 | [Pruned NVIDIA-only GitHub workflows](#1-pruned-nvidia-only-github-workflows) | infra | `.github/` | `7768fb52` |
| 2 | [Lint CI on GitHub-hosted runners](#2-lint-ci-on-github-hosted-runners) | infra | `.github/workflows/pre-commit.yml` | this commit |
| 3 | [GPU CI on the flamingo cluster](#3-gpu-ci-on-the-flamingo-cluster) | infra | `.github/workflows/gpu-tests*.yml`, `.github/k8s/`, `.testignore` | this commit |
| 4 | [Makefile](#4-makefile) | infra | `Makefile` | this commit |
| 5 | [Remote test runner](#5-remote-test-runner) | infra | `tools/test_on_flamingo.sh`, `k8s/test-pod.yaml` | this commit |
| 6 | [Remote image build](#6-remote-image-build) | infra | `tools/build_image_on_flamingo.sh`, `k8s/build-image-pod.yaml` | this commit |
| 7 | [rsync in the CI image](#7-rsync-in-the-ci-image) | infra | `docker/Dockerfile.ci` | this commit |

### 1. Pruned NVIDIA-only GitHub workflows

Upstream ships 26 workflows built around infrastructure this fork does not have: the `copy-pr-bot` GitHub app (which
creates the `pull-request/<N>` branches every upstream CI trigger keys on), self-hosted H100 and GB200 runner pools,
NVIDIA container registries, and a long list of org secrets. 24 of them could never run here.

Several were not merely inert. `cicd-approve-test-queue.yml` ran on a `*/5` cron — roughly 288 empty workflow runs a
day. `dependabot.yml` called a reusable workflow that pushes directly to the default branch and deletes remote
branches using a `PAT` secret; harmless only for as long as no such secret exists.

Two upstream workflows survive because they run on GitHub-hosted runners with no org secrets:

| Workflow | Trigger | Role |
|---|---|---|
| `detect-secrets.yml` | every PR | secret scanning; depends on `.github/workflows/config/.secrets.baseline` |
| `link-check.yml` | every PR, weekly | external link checking across `docs/**/*.md` |

`link-check.yml` was rescoped to the Sphinx sources (upstream also pointed it at the Fern `.mdx` tree). `CODEOWNERS`
was removed — every rule targeted NVIDIA teams that do not exist in this org, so it silently matched nobody.

### 2. Lint CI on GitHub-hosted runners

Upstream ran `pre-commit` inside its `cicd-main.yml` pipeline, which went with the prune. `pre-commit.yml` restores it
on `ubuntu-latest` for every PR and every push to `farai/main`, running the same hooks as
`pre-commit run --all-files` locally.

### 3. GPU CI on the flamingo cluster

Upstream's GPU CI is a large dynamic matrix over 65 functional launch scripts on H100 plus 45 on GB200. This fork runs
**only the unit test suite**, which is what we actually depend on:

| Suite | Launcher | GPUs |
|---|---|---|
| core | `tests/unit_tests/Launch_Unit_Tests_Core.sh` | 2 |
| diffusion | `tests/unit_tests/Launch_Unit_Tests_Diffusion.sh` | 2 |

Runs are triggered by a PR comment and execute in an ephemeral Kubernetes job on FAR.AI's H100 cluster:

```text
@flamingo run gpu-tests              # both suites  -> gpu-tests/default
@flamingo run gpu-tests core         # core only    -> gpu-tests/core
@flamingo run gpu-tests diffusion    # diffusion    -> gpu-tests/diffusion
```

See [`.github/workflows/README.md`](.github/workflows/README.md) for the full mechanics, required secrets, and the
`.testignore` gate that lets documentation-only PRs merge without a GPU run.

The functional tests (`tests/functional_tests/`) and the entire `gb200/` tree are **not** wired into CI here. They
depend on NVIDIA's pre-populated `/home/TestData` volume and, for GB200, on ARM64 hardware.

### 4. Makefile

A top-level `Makefile` collects the commands that would otherwise be long `uv run` invocations — docs builds, unit
tests (local and remote), the container image build, and lint. Run `make help` for the full list.

### 5. Remote test runner

The unit suites need 2 GPUs, which a laptop does not have, so `tools/test_on_flamingo.sh` runs them on an ephemeral
GPU pod on the flamingo cluster against your **local working tree** — uncommitted changes included. It uses the same
image and the same launcher scripts as CI, so local and CI stay in parity.

Everything below (and `make image-build-remote`) needs a kubectl context pointing at flamingo. That takes **three** merged
kubeconfigs — `~/.kube/config` alone defines only local clusters, the `h100-*` contexts live in `far-config`, and the
`h100` cluster definition itself comes from the [flamingo](https://github.com/AlignmentResearch/flamingo) repo. Without
all three, `kubectl` fails with "context was not found":

```sh
export KUBECONFIG=$HOME/.kube/config:$HOME/.kube/far-config:$HOME/Code/flamingo/kubeconfig
kubectl get pods        # sanity check before running anything below
```

```sh
make test-unit-remote             # both suites
make test-unit-core-remote        # core only
make test-unit-diffusion-remote   # diffusion only
make test-list                    # detached jobs still running
make test-logs [JOB=…]            # re-attach to a run
make test-teardown [JOB=…]        # stop one (JOB=all stops all)
```

Runs are **detached on the pod**, so a dropped connection or Ctrl-C does not kill them; re-attach with `test-logs`.
The pods carry `app.kubernetes.io/component: local-dev` and the script's selector only ever matches that, so
`test-teardown JOB=all` can never touch a CI run. Authentication is your own kubectl context — no secrets involved.

### 6. Remote image build

Both CI and the remote test runner pull `ghcr.io/alignmentresearch/megatron-bridge:latest`, built from upstream's
`docker/Dockerfile.ci`. That build compiles DeepEP and runs a full `uv sync` — on the order of 90 minutes cold, and
too heavy for most laptops. `tools/build_image_on_flamingo.sh` runs it on the cluster's shared BuildKit instead:

```sh
make image-build-remote                  # build on the cluster, push to GHCR (preferred)
make image-build-local                   # build on this machine's Docker daemon
make image-build-remote IMAGE_TAG=dev    # any variable from `make help` can be overridden
```

**Tagging.** A build publishes an immutable sha tag and a moving branch tag — `:3995ef57a` and
`:farai-main`. It never writes `:ci` or `:latest`, because those are what CI pods and other people
pull and a routine build must not repoint them:

| Tag | Meaning |
|---|---|
| `:<sha>` / `:<sha>-dirty` | exactly this commit; `-dirty` when the tree had uncommitted changes |
| `:<branch>` | moving pointer for the branch |
| `:ci` | what CI pods pull — promotion only |
| `:latest` | current known-good build — promotion only |

```sh
make image-build-remote PROMOTE=ci            # also tag :ci, in the same atomic push
make image-build-remote PROMOTE="ci latest"   # also tag both
make image-promote-ci SHA_TAG=3995ef57a       # retag an existing build, no rebuild
```

`PROMOTE` accepts only `ci` and `latest`, so a typo fails rather than creating a junk tag, and it
refuses to run from a dirty tree — a `-dirty` image is reproducible from no commit, so pointing CI
at one makes a failure untraceable. Override with `ALLOW_DIRTY_PROMOTE=1`. Dirty *builds* are fine;
only moving the shared tags is gated. The standalone `promote-*` targets use
`docker buildx imagetools create`, which copies the manifest registry-side — seconds, no download,
but it needs local GHCR credentials with `write:packages`.

`image-build-remote` needs only a kubectl context pointing at flamingo; the GHCR push token is read in-cluster from your
`api-keys-<username>` secret, so no credentials are handled locally. `image-build-local` needs `GH_TOKEN` (or `GITHUB_PAT`)
in your environment, since `Dockerfile.ci` takes it as a build secret.

There is deliberately **no fork-specific Dockerfile** — beyond the rsync layer in patch 7 we need no extra layers, so
adding one would create a merge-conflict surface for nothing.

The diffusion unit suite needs hash-locked WAN codecs. Bases that gate those behind
`ARG INSTALL_DIFFUSION_DEPS` (off by default, to keep CVE-carrying codecs out of shipped framework images) get the arg
passed automatically; older bases install the same deps unconditionally via `uv sync --all-extras --all-groups` and
declare no such arg. Both the Makefile and the build script probe the Dockerfile and pass the flag only when it
exists, so the tooling works unchanged across bases — which matters, since it is cherry-picked between branches whose
Dockerfiles differ substantially.

Because `Dockerfile.ci` builds from the build context rather than cloning a git ref, the image is built from your
**working tree**: uncommitted changes are included, and there is no branch to push first.

### 7. rsync in the CI image

**This is the only patch to an upstream-maintained file**, so it is the one most likely to conflict on an upstream
sync. It appends a single `apt-get install rsync` layer to `docker/Dockerfile.ci`, immediately before the final
`COPY`, so it does not invalidate the expensive DeepEP / `uv sync` cache above it.

The remote test runner syncs your working tree over a `kubectl exec` transport, and rsync has to exist on **both**
ends — it executes `rsync --server` inside the container. Without it the runner falls back to a tar stream that
re-sends the entire tree (~258 MB, including `.git`) on every run. The runner probes both ends and picks the
available path automatically, so this is a performance patch, not a correctness one.

If an upstream merge conflicts here, re-applying is a four-line addition at the end of the file.


## Development

Upstream's [CONTRIBUTING.md](CONTRIBUTING.md) covers the general workflow. Fork-specific notes:

### Environment

```sh
uv sync                  # base environment
uv run pre-commit install
```

**macOS caveat:** `uv sync --group dev` fails on Apple Silicon because `nvidia-resiliency-ext` publishes Linux-only
wheels. Use `pre-commit` directly (`pre-commit run --all-files`) rather than `uv run --group dev pre-commit ...` when
working on a Mac.

### Tests

Unit tests need **2 GPUs** — 143 of 442 test files touch CUDA and 73 use `torch.distributed`, and only a handful are
guarded by skip markers, so a GPU-less run fails rather than skipping. Locally, on a GPU box:

```sh
make test-unit             # both suites
make test-unit-core        # core only
make test-unit-diffusion   # diffusion only
```

### Docs

Upstream's Sphinx site builds cleanly. `docs/fern/` is a parallel, NVIDIA-hosted docs pipeline we do not use.

```sh
make docs-html   # build to docs/_build/html
make docs-live   # live-reload server on :8001
```

Building requires the Megatron-LM submodule, since `uv` resolves `megatron-core` from it even for a docs-only group:

```sh
git submodule update --init --recursive
```

> **Note:** always pass `-d` (the `make` targets do). Sphinx defaults its doctree cache to `<outdir>/.doctrees`, which
> puts a ~7 GB cache *inside* the 131 MB site — fine locally, fatal for any publish step with a size limit.


## Keeping the Fork in Sync with Upstream

Add the NVIDIA repo as a remote named `upstream` once:

```sh
git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
```

Then to pull in new upstream commits:

```sh
git fetch upstream
git checkout main
git merge --ff-only upstream/main     # keep `main` a clean mirror
git push origin main

git checkout farai/main
git merge main
```

Resolve any conflicts, run the tests, then push `farai/main`. Feature branches should be rebased or merged against the
updated `farai/main` after each sync.

Conflicts concentrate in `.github/` — upstream actively develops the workflows we deleted. When a merge reintroduces
one, delete it again rather than reconciling it; the [FAR.AI Patches](#farai-patches) table above is the record of
what should and should not be present.
