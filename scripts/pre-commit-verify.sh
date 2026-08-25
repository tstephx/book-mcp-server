#!/bin/sh
# Local secret-scanning gate — blocks a commit that stages a live credential.
# Wiring (one-time per clone, .git/hooks is untracked):
#   ln -sf ../../scripts/pre-commit-verify.sh .git/hooks/pre-commit
set -e
cd "$(git rev-parse --show-toplevel)"

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "gitleaks not found on PATH (brew install gitleaks) — refusing to commit without a secret scan." >&2
    exit 1
fi

gitleaks protect --staged --redact

# Reminder to update CLAUDE.md when structure changes (folded in from the
# prior local-only .git/hooks/pre-commit convenience hook this replaces).
if git diff --cached --name-only | grep -qE '\.(py)$'; then
  echo ""
  echo "📝 Reminder: If you added tools, CLI commands, or changed architecture,"
  echo "   consider updating CLAUDE.md (last updated: $(grep -o '20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]' CLAUDE.md | tail -1))"
  echo ""
fi
