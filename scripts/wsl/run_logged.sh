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
exec "$@" > "results/${job_name}.log" 2>&1
