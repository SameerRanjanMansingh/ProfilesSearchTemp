#!/usr/bin/env bash
set -euo pipefail

# move into server folder
cd server

# If pip exists, install; if not, skip (Railway build phase installs dependencies)
if python -m pip --version >/dev/null 2>&1; then
  python -m pip install -r requirements.txt || true
fi

# Run the app (correct import path)
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
