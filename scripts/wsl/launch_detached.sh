#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 JOB_NAME COMMAND [ARG ...]" >&2
  exit 2
fi

repo=/mnt/c/Rithun/Github/deltashot
job_name=$1
shift
cd "$repo"
mkdir -p results
# A separate session prevents WSL from reaping the child when the Windows-side
# launcher invocation exits.
nohup setsid -f "$@" > "results/${job_name}.log" 2>&1 < /dev/null &
echo "launched ${job_name}; poll results/${job_name}.log and result artifacts"
