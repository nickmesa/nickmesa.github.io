#!/usr/bin/env bash
# Publish ./public-env (a worktree of the glm-env branch: IR sheets and vorticity) and push.
# Usage: tools/publish_env.sh "<commit message>"
#
# The branch holds a few hundred megabytes of images, so it is kept as a single
# commit with no history: each publish replaces that commit. Only the glm-env
# branch is ever force-pushed.
set -euo pipefail
msg="$1"
branch="glm-env"

cd public-env
git add -A
if git rev-parse --verify -q HEAD >/dev/null && git diff --cached --quiet; then
  echo "nothing to publish"
  exit 0
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
commit="$(git commit-tree "$(git write-tree)" -m "$msg")"
if before="$(git rev-parse --verify -q HEAD)"; then
  # replace the commit only if the branch is still where this run found it
  git push --force-with-lease="$branch:$before" origin "$commit:refs/heads/$branch"
else
  git push origin "$commit:refs/heads/$branch"
fi
git reset -q --soft "$commit" 2>/dev/null || git update-ref "refs/heads/$branch" "$commit"
