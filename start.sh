#!/usr/bin/env bash
set -euo pipefail


# install requirements (safe if already installed by Railway)
if [ -f requirements.txt ]; then
  pip install -r server.requirements.txt
fi

# run the app bound to the PORT Railway provides
exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-8000}"
