#!/usr/bin/env bash
# Bring newer upstream commits into this fork by MERGING them.
#
# Invoked by `make sync-upstream`. Creates sync/upstream-<YYYYMMDD>-<short-sha>, merges the target
# upstream commit into it, and regenerates .fork-base.json.
#
# Why a merge and not a rebase: force-pushing the shared branch is banned, so a rebase of
# farai/main cannot land. A merge advances the base without rewriting anything — upstream's
# commits keep their SHAs and simply become reachable from our HEAD, so merge-base moves forward
# to the newest upstream commit we contain, which is what the manifest records.
#
# This stops before pushing. The branch is opened as a PR for review and CI, and MUST be landed
# with a merge commit — see the note this prints at the end.
set -euo pipefail

REMOTE=${UPSTREAM_REMOTE:-upstream}
BRANCH=${UPSTREAM_BRANCH:-main}
TARGET_REF=${UPSTREAM_REF:-}
DRY_RUN=${DRY_RUN:-0}

die() { echo "ERROR: $*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first"
git remote get-url "$REMOTE" >/dev/null 2>&1 || \
  die "no '$REMOTE' remote. Add it: git remote add $REMOTE <upstream-url>"

START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$START_BRANCH" != "HEAD" ] || die "detached HEAD — check out the branch you want to sync"

echo "Fetching $REMOTE/$BRANCH ..."
git fetch --quiet "$REMOTE" "$BRANCH"

TARGET=$(git rev-parse --verify "${TARGET_REF:-$REMOTE/$BRANCH}^{commit}") || \
  die "cannot resolve ${TARGET_REF:-$REMOTE/$BRANCH}"
OLD_BASE=$(git merge-base HEAD "$REMOTE/$BRANCH")

# The target must be a real upstream commit, or the new base would be a fiction.
git merge-base --is-ancestor "$TARGET" "$REMOTE/$BRANCH" || \
  die "$(git rev-parse --short=9 "$TARGET") is not an ancestor of $REMOTE/$BRANCH"

if git merge-base --is-ancestor "$TARGET" HEAD; then
  echo "Already contains $(git rev-parse --short=9 "$TARGET") — nothing to sync."
  exit 0
fi

SYNC_BRANCH="sync/upstream-$(date +%Y%m%d)-$(git rev-parse --short=9 "$TARGET")"

cat <<INFO

  source branch : $START_BRANCH
  current base  : $(git rev-parse --short=9 "$OLD_BASE")  ($(git log -1 --format=%as "$OLD_BASE"))
  target base   : $(git rev-parse --short=9 "$TARGET")  ($(git log -1 --format=%as "$TARGET"))
  upstream delta: $(git rev-list --count "$OLD_BASE..$TARGET") commits
  our commits   : $(git rev-list --count --no-merges "$OLD_BASE..HEAD")
  sync branch   : $SYNC_BRANCH

INFO

if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN=1 — stopping before any change."; exit 0; fi

git checkout -q -b "$SYNC_BRANCH"
echo "Merging $(git rev-parse --short=9 "$TARGET") into $SYNC_BRANCH ..."
if ! git merge --no-ff "$TARGET" -m "Merge upstream $(git rev-parse --short=9 "$TARGET") into $START_BRANCH"; then
  cat <<'RECOVER'

Merge stopped on a conflict. Resolve it, then:

    git add <files> && git commit          # completes the merge
    make fork-base                         # regenerate the manifest
    git add .fork-base.json && git commit -m "chore: record new upstream base"

Resolve deliberately: taking --ours wholesale keeps our version of a file and silently discards
upstream's changes to it, while the recorded base still claims we contain that upstream commit.

To abandon:  git merge --abort && git checkout - && git branch -D <sync branch>
RECOVER
  exit 1
fi

python3 tools/fork_base.py --write
if [ -n "$(git status --porcelain -- .fork-base.json)" ]; then
  git add .fork-base.json
  git commit -q -m "chore(fork-base): record upstream base $(git rev-parse --short=9 "$TARGET")"
fi

echo
python3 tools/fork_base.py --check || die "post-sync check failed — do not push this branch"

cat <<NEXT

Done. Review, then:

    git push -u origin $SYNC_BRANCH
    # open a PR against $START_BRANCH

Land it with "Create a merge commit". Squashing collapses the merge, so upstream's commits never
enter our history and merge-base does not move.

NEXT
