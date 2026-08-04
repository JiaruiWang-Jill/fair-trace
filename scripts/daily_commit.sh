#!/bin/bash
# Daily batch commit+push for the fair-trace repo. Run via crontab at 10pm local time.
# Safe to run with nothing to commit — it just no-ops.
set -euo pipefail

REPO_DIR="/Users/jiarui/niw_github/fair-trace"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

cd "$REPO_DIR"

git add -A

if git diff --cached --quiet; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to commit."
    exit 0
fi

CHANGED_FILES=$(git diff --cached --name-only | tr '\n' ' ')
git commit -m "Daily batch update ($(date '+%Y-%m-%d %H:%M %Z'))

Files changed: ${CHANGED_FILES}"

git push origin main

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Committed and pushed: ${CHANGED_FILES}"
