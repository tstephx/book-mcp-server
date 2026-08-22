#!/usr/bin/env bash
set -euo pipefail

# Git pre-push hook: runs this repository's canonical offline checks
# (ruff lint + the unit test suite, the same sequence ci.yml's `lint` and
# `test` jobs run) before any push from this checkout (or its linked
# worktrees -- hooks resolve from the shared git common dir). Installed
# because hosted CI is billing-blocked account-wide (taylor-dev-core
# issue #81) and the owner decided not to restore it. The workflow file
# stays in place and resumes automatically if billing is ever fixed.
#
# Integration tests (`make test-integration`, needs OPENAI_API_KEY and a
# real library DB) are deliberately NOT run here -- CI itself skips them
# by the same pyproject.toml addopts default, since no OPENAI_API_KEY is
# available there either.
#
# git exports GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE/GIT_PREFIX to hook
# processes. Left in place, any subprocess this gate spawns that itself
# shells out to git can get silently redirected into this repo's real
# .git instead of a scratch repo it creates -- confirmed live the same
# day in two sibling repos (briefcase PR #34 / commit b603539;
# epub-pdf-splitter's pdf-parser-base worktree). Unset before running
# anything.
#
# Bypass for a genuine emergency: `git push --no-verify` -- git's own
# standard escape hatch, deliberately not re-blocked here. A bypassed
# push leaves no local trace, so treat it like a red hosted check used to
# be treated: fix or revert before further material work on main.
#
# Install (once per clone):
#   ln -sf ../../scripts/pre-push-verify.sh .git/hooks/pre-push

unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

# Portability regression guard: tracked .claude/**, .mcp.json, and CLAUDE.md
# must not gain a fresh-clone-breaking /Users/... path. The allowlist below
# covers two reviewed exceptions: the taylor-dev-core marketplace's
# version-pinned directory source in .claude/settings.json (fleet portability
# audit, 2026-08-21 -- centrally managed by fleet promotion tooling, degrades
# gracefully if absent), and .mcp.json's shared book-library DB default
# (same external-data default documented in CLAUDE.md's Environment
# Variables table, overridable via AGENTIC_PIPELINE_DB) -- .mcp.json is
# currently .gitignore'd so this entry is a no-op today, kept so the guard
# doesn't need editing if that file is ever tracked. Extend the allowlist,
# don't remove this check, if a new legitimate exception is reviewed.
printf 'pre-push: checking tracked .claude/**, .mcp.json, and CLAUDE.md for unallowlisted /Users/ paths...\n'
portability_targets=()
while IFS= read -r f; do
  portability_targets+=("$repository_root/$f")
done < <(git -C "$repository_root" ls-files -- '.claude' '.mcp.json' 'CLAUDE.md')

portability_hits=""
if [[ ${#portability_targets[@]} -gt 0 ]]; then
  portability_hits="$(grep -Hn '/Users/' "${portability_targets[@]}" 2>/dev/null \
    | grep -vE '/Users/taylorstephens/Dev/_Workspace/\.harness-releases/taylor-dev-core/|/Users/taylorstephens/Library/Application Support/book-library/library\.db' \
    || true)"
fi

if [[ -n "$portability_hits" ]]; then
  printf 'pre-push: machine-specific /Users/ path(s) found outside the allowlist -- push blocked:\n%s\n' "$portability_hits" >&2
  printf 'pre-push: if this is a legitimate retained exception, extend the allowlist pattern in this script.\n' >&2
  exit 1
fi
printf 'pre-push: portability check passed.\n'

printf 'pre-push: syncing locked dependencies (uv sync --locked --python 3.12 --extra dev)...\n'
uv sync --locked --python 3.12 --extra dev

printf 'pre-push: running make lint && make test (bypass: git push --no-verify)\n'
if ! { make lint && make test; }; then
  printf 'pre-push: lint or test failed -- push blocked. Fix the failure (or --no-verify for a genuine emergency) and retry.\n' >&2
  exit 1
fi
printf 'pre-push: lint and test passed -- push allowed.\n'
