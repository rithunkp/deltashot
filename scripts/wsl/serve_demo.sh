#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
echo "Starting DeltaShot demo from $repo_dir on port ${1:-8765}"
exec .venv_wsl/bin/python -u -m scripts.local_demo --host 127.0.0.1 --port "${1:-8765}"
