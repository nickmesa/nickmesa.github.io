#!/usr/bin/env bash
# Commit the data files in ./public (a worktree of the glm-data branch) and push.
# Usage: tools/publish.sh "<commit message>"
#
# Hourly updates rewrite the same JSON files, so commits that share a message
# (one per UTC day for real-time runs) are folded into one to keep the branch
# small. Only the glm-data branch is ever force-pushed; main is never touched.
set -euo pipefail
msg="$1"
branch="glm-data"

cd public
git add -A
if git diff --cached --quiet; then
  echo "nothing to publish"
  exit 0
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
if [ "$(git log -1 --format=%s 2>/dev/null || true)" = "$msg" ]; then
  before="$(git rev-parse HEAD)"
  git commit --quiet --amend -m "$msg"
  # replace the commit only if the branch is still where this run found it
  git push --force-with-lease="$branch:$before" origin "$branch"
else
  git commit --quiet -m "$msg"
  git push origin "$branch"
fi
