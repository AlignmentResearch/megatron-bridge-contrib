#!/usr/bin/env python3
# Copyright (c) 2026, FAR.AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record this fork's upstream base, and check that its pinned submodules agree with it.

The SAME file is meant to be dropped unchanged into every FAR.AI fork that carries patches on an
upstream — it is not specific to any one repository. A repo may be a dependency, a consumer of
dependencies, or both, and each role is a mode of the same computation.

A fork commit `D` of project P is compatible with a commit `N` of a project that pins P when:

    compatible(N, D)  <=>  base(D) == pin_P(base(N))
    base(X)           =    merge-base(X, upstream/<branch>)
    pin_P(commit)     =    git ls-tree <commit> <path of P>   (in P's UPSTREAM consumer)

`base` is computable from git, but only in a full clone with the upstream remote fetched.
Consumers commonly declare submodules `shallow = true`, so the dependency usually cannot be asked
— hence each fork records its own base in `.fork-base.json`, and `--check` re-derives it so the
record cannot drift silently.

`--check` enforces four things:

  1. `.fork-base.json` still matches `merge-base(HEAD, upstream/<branch>)`.
  2. Every pinned submodule is compatible: a patched dependency's recorded base equals the commit
     upstream pins for it at our base; an unpatched one's gitlink equals upstream's exactly.
  3. A dependency's manifest describes the fork we actually pin (`fork_repo`) and the upstream we
     actually expect (`upstream_repo`). Two forks can share an upstream base while carrying
     entirely different patch sets, so the base alone does not identify a fork.
  4. A dependency's own nested submodules pin the same commits we do, matched by URL rather than
     path (layouts differ between repos). The base rule alone cannot catch this: a patch may move
     a nested gitlink without moving the dependency's base.

Repos with no submodules simply have nothing to do for 2-4.

The manifest deliberately records only fields that are stable across the patch series. Recording
the fork head or the patch SHAs would force a regeneration on every commit, and could never be
accurate inside the very commit that carries them.

Usage:
    tools/fork_base.py --write        # regenerate .fork-base.json from git
    tools/fork_base.py --check        # verify the manifest AND the pins
    tools/fork_base.py --check --json # machine-readable report
    tools/fork_base.py --print        # emit the computed base, for scripts
"""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
from pathlib import Path


MANIFEST = ".fork-base.json"
SCHEMA = 1

OK = "ok"
MISMATCH = "mismatch"
UNCHECKED = "unchecked"  # could not verify — a failure unless --allow-skips
SKIPPED = "skipped"  # structurally nothing to check — never a failure


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return its stripped stdout."""
    out = subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=cwd)
    return out.stdout.strip()


def _git_ok(*args: str, cwd: Path | None = None) -> bool:
    """Return True if a git command exits zero."""
    return subprocess.run(["git", *args], capture_output=True, cwd=cwd).returncode == 0


def repo_root() -> Path:
    """Return the repository root as a Path."""
    return Path(_git("rev-parse", "--show-toplevel"))


def remote_url(remote: str) -> str:
    """Return a remote's URL, normalized to https form without a trailing .git."""
    url = _git("remote", "get-url", remote)
    if url.startswith("git@"):  # git@github.com:org/repo.git -> https://github.com/org/repo
        url = "https://" + url[4:].replace(":", "/", 1)
    return url.removesuffix(".git")


def normalize(url: str) -> str:
    """Normalize a URL for comparison: https form, no .git suffix, lowercased."""
    if url.startswith("git@"):
        url = "https://" + url[4:].replace(":", "/", 1)
    return url.removesuffix(".git").rstrip("/").lower()


def compute_base(upstream_remote: str, upstream_branch: str, rev: str = "HEAD") -> str:
    """Return the commit this fork is rebased onto: merge-base(rev, <remote>/<branch>)."""
    ref = f"{upstream_remote}/{upstream_branch}"
    if not _git_ok("rev-parse", "--verify", ref):
        sys.exit(
            f"ERROR: {ref} not found. This needs a full clone with the upstream remote fetched:\n"
            f"       git remote add {upstream_remote} <upstream-url>\n"
            f"       git fetch --filter=blob:none {upstream_remote} {upstream_branch}"
        )
    return _git("merge-base", rev, ref)


def parse_gitmodules(text: str) -> list[dict]:
    """Parse .gitmodules content into a list of {path, url} entries."""
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    mods = [
        {"path": cfg.get(s, "path"), "url": cfg.get(s, "url")}
        for s in cfg.sections()
        if cfg.has_option(s, "path") and cfg.has_option(s, "url")
    ]
    return sorted(mods, key=lambda m: m["path"])


def read_gitmodules(path: Path) -> list[dict]:
    """Parse a .gitmodules file from disk, returning [] when it does not exist."""
    return parse_gitmodules(path.read_text()) if path.exists() else []


def show_gitmodules(root: Path, rev: str) -> list[dict]:
    """Parse .gitmodules as of a given revision, returning [] when absent there."""
    try:
        return parse_gitmodules(_git("show", f"{rev}:.gitmodules", cwd=root))
    except subprocess.CalledProcessError:
        return []


def gitlink(root: Path, rev: str, path: str) -> str | None:
    """Return the commit a tree pins for a submodule path, or None if absent."""
    try:
        line = _git("ls-tree", rev, path, cwd=root)
    except subprocess.CalledProcessError:
        return None
    parts = line.split()  # "160000 commit <sha>\t<path>"
    return parts[2] if len(parts) >= 3 and parts[1] == "commit" else None


def is_populated(path: Path) -> bool:
    """Return True if a submodule directory exists and is not empty."""
    return path.is_dir() and any(path.iterdir())


def load_manifest(path: Path) -> tuple[dict | None, str]:
    """Load a manifest, returning (data, error) with error empty on success."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"{MANIFEST} not found"
    except json.JSONDecodeError as exc:
        return None, f"{MANIFEST} is not valid JSON: {exc}"
    except OSError as exc:
        return None, f"{MANIFEST} could not be read: {exc}"
    if not isinstance(data, dict):
        return None, f"{MANIFEST} must contain a JSON object"
    if data.get("schema") != SCHEMA:
        return None, (
            f"{MANIFEST} schema {data.get('schema')!r}, expected {SCHEMA} — this tool is too old for that manifest"
        )
    return data, ""


def write_manifest(root: Path, upstream_remote: str, upstream_branch: str) -> dict:
    """Regenerate this repo's manifest from git and write it to disk."""
    data = {
        "schema": SCHEMA,
        # Keyed on (fork_repo, upstream_base): two forks can share an upstream base while
        # carrying different patch sets, so the base alone does not identify a fork.
        "fork_repo": remote_url("origin"),
        "upstream_repo": remote_url(upstream_remote),
        "upstream_branch": upstream_branch,
        "upstream_base": compute_base(upstream_remote, upstream_branch),
    }
    (root / MANIFEST).write_text(json.dumps(data, indent=2) + "\n")
    return data


def check_nested(root: Path, dep_path: str, own_pins: dict[str, tuple[str, str]]) -> list[str]:
    """Compare a dependency's nested submodule pins against this repo's own pins.

    Matching is by normalized URL, not by path: repos lay the same project out at different
    paths, so a path-based match would find nothing and pass vacuously.
    """
    dep_root = root / dep_path
    problems = []
    for nested in read_gitmodules(dep_root / ".gitmodules"):
        key = normalize(nested["url"])
        if key not in own_pins:
            continue  # a project this repo does not pin — nothing to agree with
        our_path, our_sha = own_pins[key]
        their_sha = gitlink(dep_root, "HEAD", nested["path"])
        if their_sha is None:
            problems.append(f"nested {nested['path']}: no gitlink in the dependency tree")
        elif their_sha != our_sha:
            problems.append(
                f"nested {nested['path']} pins {their_sha[:12]}, but this repo pins "
                f"{our_sha[:12]} at {our_path} — the dependency was developed against a "
                "different commit than this repo resolves"
            )
    return problems


def check_pin(
    root: Path,
    sub: dict,
    base_n: str,
    upstream_urls: dict[str, str],
    own_pins: dict[str, tuple[str, str]],
) -> dict:
    """Check a single pinned submodule and return a result row."""
    path, url = sub["path"], sub["url"]
    pinned = gitlink(root, "HEAD", path)
    expected = gitlink(root, base_n, path)
    row: dict = {"path": path, "url": url, "pinned": pinned, "upstream_pin": expected}

    if expected is None:
        row.update(
            status=SKIPPED,
            reasons=["not present upstream at our base — nothing to be compatible with"],
        )
        return row

    if not is_populated(root / path):
        row.update(
            status=UNCHECKED,
            reasons=[
                "submodule not initialized — cannot verify. Check out with `submodules: recursive` (shallow is fine)"
            ],
        )
        return row

    problems: list[str] = []
    note = ""
    manifest_path = root / path / MANIFEST

    if not manifest_path.exists():
        # No manifest => an unpatched dependency: the pin must equal upstream's exactly.
        row["base"] = pinned
        if pinned == expected:
            note = "unpatched; pin matches upstream"
        else:
            problems.append(f"unpatched, but pin != upstream pin ({expected[:12]})")
    else:
        data, err = load_manifest(manifest_path)
        if err:
            row.update(status=MISMATCH, reasons=[err])
            return row
        assert data is not None
        base_d = data.get("upstream_base", "")
        row["base"] = base_d

        fork_repo = data.get("fork_repo", "")
        if not fork_repo:
            problems.append(f"{MANIFEST} has no fork_repo — cannot confirm which fork this is")
        elif normalize(fork_repo) != normalize(url):
            problems.append(f"fork_repo {fork_repo} != .gitmodules url {url}")

        # The fork must track the upstream that upstream pins here, not merely *an* upstream.
        upstream_repo = data.get("upstream_repo", "")
        want = upstream_urls.get(path)
        if not upstream_repo:
            problems.append(f"{MANIFEST} has no upstream_repo")
        elif want is None:
            note = "upstream .gitmodules has no entry for this path; upstream_repo unverified"
        elif normalize(upstream_repo) != normalize(want):
            problems.append(f"upstream_repo {upstream_repo} != upstream's url {want}")

        if base_d != expected:
            problems.append(f"base {base_d[:12] or '?'} != upstream pin {expected[:12]}")
        elif not problems and not note:
            note = "patched; base matches upstream pin"

    problems += check_nested(root, path, own_pins)
    row.update(status=MISMATCH if problems else OK, reasons=problems or [note or "ok"])
    return row


def check_pins(root: Path, base_n: str) -> list[dict]:
    """Check every submodule this repo pins."""
    subs = read_gitmodules(root / ".gitmodules")
    upstream_urls = {m["path"]: m["url"] for m in show_gitmodules(root, base_n)}

    # Every project THIS repo pins, keyed by normalized URL, for the nested comparison.
    own_pins: dict[str, tuple[str, str]] = {}
    for m in subs:
        sha = gitlink(root, "HEAD", m["path"])
        if sha:
            own_pins[normalize(m["url"])] = (m["path"], sha)

    return [check_pin(root, s, base_n, upstream_urls, own_pins) for s in subs]


def report(base_n: str, manifest_err: str, rows: list[dict], allow_skips: bool) -> None:
    """Print a human-readable report."""
    mark = {OK: "OK  ", MISMATCH: "FAIL", UNCHECKED: "????", SKIPPED: "SKIP"}
    print(f"Base: {base_n[:12]}")
    print(f"  [{'FAIL' if manifest_err else 'OK  '}] {MANIFEST}")
    if manifest_err:
        for line in manifest_err.splitlines():
            print(f"         {line}")

    if rows:
        print("\nPinned dependencies:")
        width = max(len(r["path"]) for r in rows)
        for r in rows:
            head, *rest = r["reasons"]
            print(f"  [{mark[r['status']]}] {r['path']:<{width}}  {head}")
            for extra in rest:
                print(f"  {'':<{width + 9}}{extra}")
    else:
        print("\nNo submodules pinned.")

    bad = [r for r in rows if r["status"] == MISMATCH]
    unchecked = [r for r in rows if r["status"] == UNCHECKED]
    print()
    if bad:
        print(f"{len(bad)} incompatible pin(s). A dependency is compatible when its upstream base")
        print("equals the commit upstream pins for it at our base, and its nested pins agree with")
        print("ours. Rebase the dependency, or move this repo's base to one whose pins match.")
    if unchecked:
        print(
            f"{len(unchecked)} dependency/ies could not be checked "
            f"({'ignored' if allow_skips else 'treated as failures'})."
        )
    if not bad and not unchecked and not manifest_err:
        print("Manifest is current and all pinned dependencies are compatible.")


def cmd_check(root: Path, args: argparse.Namespace) -> int:
    """Verify this repo's manifest and its pins; return a process exit code."""
    data, err = load_manifest(root / MANIFEST)
    if err:
        sys.exit(f"ERROR: {err}. Generate it with: tools/fork_base.py --write")
    assert data is not None

    branch = args.upstream_branch or data.get("upstream_branch", "main")
    base_n = compute_base(args.upstream_remote, branch)
    recorded = data.get("upstream_base", "")

    manifest_err = ""
    if recorded != base_n:
        manifest_err = (
            f"stale: recorded {recorded or '<missing>'}, actual merge-base {base_n}\n"
            "This normally means the fork was rebased without regenerating the manifest.\n"
            f"Fix with: make fork-base   (then commit {MANIFEST})"
        )

    rows = check_pins(root, base_n)

    if args.json:
        print(json.dumps({"base": base_n, "manifest_error": manifest_err, "submodules": rows}, indent=2))
    else:
        report(base_n, manifest_err, rows, args.allow_skips)

    failing = {MISMATCH} if args.allow_skips else {MISMATCH, UNCHECKED}
    return 1 if manifest_err or any(r["status"] in failing for r in rows) else 0


def main() -> int:
    """Run the requested mode and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the manifest from git")
    mode.add_argument("--check", action="store_true", help="verify the manifest and the pins")
    mode.add_argument("--print", dest="show", action="store_true", help="print the computed base")
    parser.add_argument("--json", action="store_true", help="--check: emit JSON")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="--check: do not fail on deps that could not be checked (default: they fail)",
    )
    parser.add_argument("--upstream-remote", default="upstream", help="remote holding upstream")
    parser.add_argument("--upstream-branch", default="", help="upstream branch (default: main)")
    args = parser.parse_args()

    root = repo_root()
    if args.show:
        return print(compute_base(args.upstream_remote, args.upstream_branch or "main")) or 0
    if args.write:
        data = write_manifest(root, args.upstream_remote, args.upstream_branch or "main")
        base, branch = data["upstream_base"], data["upstream_branch"]
        print(f"Wrote {MANIFEST}: upstream_base {base[:12]} ({branch})")
        return 0
    return cmd_check(root, args)


if __name__ == "__main__":
    sys.exit(main())
