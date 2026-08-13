#!/usr/bin/env bash
# Replay this fork's patches onto a newer upstream commit.
#
# Invoked by `make sync-upstream`. Creates sync/upstream-<YYYYMMDD>-<short-sha>, rebases the patch
# series onto the target, and regenerates .fork-base.json.
#
# Why a rebase and not a merge: the compatibility rule assumes our tree is "upstream@base plus our
# patches". A merge moves the base while conflict resolution may quietly drop upstream hunks, so
# the recorded base would overstate how much upstream we actually have. Replaying makes the claim
# literally true.
#
# This deliberately stops before pushing. None of GitHub's merge buttons produce a linear replay
# (a merge commit is non-linear, squash collapses the series, and "rebase and merge" would
# duplicate the patches onto the branch tip), so the sync branch is opened as a PR for review and
# CI, and lands by force-pushing the reviewed SHA.
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

if [ "$TARGET" = "$OLD_BASE" ]; then
  echo "Already based on $(git rev-parse --short=9 "$TARGET") — nothing to sync."
  exit 0
fi

# Forward-only. Rebasing onto older upstream would silently revert upstream fixes.
git merge-base --is-ancestor "$OLD_BASE" "$TARGET" || \
  die "target $(git rev-parse --short=9 "$TARGET") is not a descendant of the current base
       $(git rev-parse --short=9 "$OLD_BASE"). A sync must move the base forward."

SYNC_BRANCH="sync/upstream-$(date +%Y%m%d)-$(git rev-parse --short=9 "$TARGET")"
PATCHES=$(git rev-list --count "$OLD_BASE..HEAD")
MERGES=$(git rev-list --merges --count "$OLD_BASE..HEAD")

cat <<INFO

  source branch : $START_BRANCH
  current base  : $(git rev-parse --short=9 "$OLD_BASE")  ($(git log -1 --format=%as "$OLD_BASE"))
  target base   : $(git rev-parse --short=9 "$TARGET")  ($(git log -1 --format=%as "$TARGET"))
  upstream delta: $(git rev-list --count "$OLD_BASE..$TARGET") commits
  patches       : $PATCHES ($MERGES merge commits)
  sync branch   : $SYNC_BRANCH

INFO

if [ "$MERGES" -gt 0 ]; then
  echo "NOTE: $MERGES merge commit(s) in the series. A plain rebase flattens them; each dropped"
  echo "      conflict resolution has to be redone. Consider linearizing first."
  echo
fi

if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN=1 — stopping before any change."; exit 0; fi

git checkout -q -b "$SYNC_BRANCH"
echo "Replaying $PATCHES commits onto $(git rev-parse --short=9 "$TARGET") ..."
if ! git rebase --onto "$TARGET" "$OLD_BASE"; then
  cat <<'RECOVER'

Rebase stopped on a conflict. Resolve it, then:

    git add <files> && git rebase --continue     # repeat until done
    make fork-base                               # regenerate the manifest
    git add .fork-base.json && git commit -m "chore: record new upstream base"

To abandon:  git rebase --abort && git checkout - && git branch -D <sync branch>
RECOVER
  exit 1
fi

python3 tools/fork_base.py --write
if [ -n "$(git status --porcelain -- .fork-base.json)" ]; then
  git add .fork-base.json
  git commit -q -m "chore(fork-base): rebase onto upstream $(git rev-parse --short=9 "$TARGET")"
fi

echo
python3 tools/fork_base.py --check || die "post-sync check failed — do not push this branch"

cat <<NEXT

Done. Review, then:

    git push -u origin $SYNC_BRANCH
    # open a PR against $START_BRANCH for review + CI, then land it with:
    git push --force-with-lease origin $SYNC_BRANCH:$START_BRANCH

NEXT
