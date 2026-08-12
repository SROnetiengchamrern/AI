#!/usr/bin/env bash
# Render start script — do NOT use "gunicorn app:app" for this project.
set -euo pipefail
exec uvicorn app:create_app --factory --host 0.0.0.0 --port "${PORT:-10000}"
