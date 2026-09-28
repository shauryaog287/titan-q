#!/bin/bash
# SessionStart hook for Claude Code on the web (cloud research sessions, see CLOUD.md).
# Builds a Python 3.12 venv in .venv from requirements.txt. torch comes from the CPU
# index; if that host is blocked, torch is skipped (the decision layer, broker and
# backtest run without it; train_models.py's neural SDE and PPO parts need it).
# Idempotent: packages already installed are not reinstalled.
# On the trading PC (not a cloud session) it does nothing.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
VENV="$CLAUDE_PROJECT_DIR/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p python3.12 "$VENV"
fi
export VIRTUAL_ENV="$VENV"

grep -v -E '^(torch|#)' requirements.txt > "$VENV/requirements-no-torch.txt"
uv pip install -q -r "$VENV/requirements-no-torch.txt"

TORCH_PIN="$(grep -E '^torch==' requirements.txt || true)"
if [ -n "$TORCH_PIN" ] && ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
  uv pip install -q "$TORCH_PIN" --index-url https://download.pytorch.org/whl/cpu \
    || echo "session-start: torch not installed (download.pytorch.org unreachable)" >&2
fi

# make "python" the project venv for the rest of the session
echo "export VIRTUAL_ENV=\"$VENV\"" >> "$CLAUDE_ENV_FILE"
echo "export PATH=\"$VENV/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
