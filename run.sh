#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8088}"

# Pick the Python interpreter that has torch, fastapi, and uvicorn installed
PY_BIN=""
for candidate in \
  "$(command -v python3 2>/dev/null || true)" \
  "/opt/anaconda3/bin/python3" \
  "$HOME/anaconda3/bin/python3" \
  "$HOME/miniconda3/bin/python3"
do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    if "$candidate" -c "import uvicorn, fastapi, torch" 2>/dev/null; then
      PY_BIN="$candidate"
      break
    fi
  fi
done

if [[ -z "$PY_BIN" ]]; then
  echo "Error: Could not find a python3 environment with uvicorn, fastapi, and torch installed." >&2
  exit 1
fi

echo "Starting FaultSense AI on http://127.0.0.1:$PORT using $PY_BIN"
exec "$PY_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
