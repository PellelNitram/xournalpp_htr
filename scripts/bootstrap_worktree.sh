#!/usr/bin/env bash
# Bootstrap a linked git worktree by symlinking gitignored files from the
# main checkout. Idempotent: safe to run on every Claude session start.
set -euo pipefail

worktree_root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$worktree_root"

git_dir=$(git rev-parse --git-dir)
common_dir=$(git rev-parse --git-common-dir)
git_dir_abs=$(cd "$git_dir" && pwd)
common_dir_abs=$(cd "$common_dir" && pwd)

[ "$git_dir_abs" = "$common_dir_abs" ] && exit 0

main_worktree=$(git worktree list --porcelain | awk '/^worktree / {print $2; exit}')
[ -z "$main_worktree" ] && exit 0
[ "$main_worktree" = "$worktree_root" ] && exit 0

if [ ! -e ".env" ] && [ -e "$main_worktree/.env" ]; then
    ln -s "$main_worktree/.env" ".env"
    echo "bootstrap_worktree: linked .env from $main_worktree"
fi
