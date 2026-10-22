#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Rithun/Github/deltashot
mkdir -p results
.venv_wsl/bin/python -c 'import compressai, torch; print("compressai=" + compressai.__version__); print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available()))' | tee results/wsl_environment_check.txt
