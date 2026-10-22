#!/usr/bin/env bash
set -euxo pipefail

MODE="${1:?usage: job_entrypoint.sh dryrun|calibrate|full}"
case "$MODE" in
  dryrun|calibrate|full) ;;
  *) echo "Unknown job mode: $MODE" >&2; exit 2 ;;
esac

BUNDLE_REPO="${HF_BUNDLE_REPO:?Set HF_BUNDLE_REPO via Job env}"
ARTIFACT_REPO="${HF_ARTIFACT_REPO:?Set HF_ARTIFACT_REPO via Job env}"
ROOT="${DELTASHOT_BUNDLE_DIR:-/workspace/deltashot_bundle}"
mkdir -p "$ROOT/results"
cd "$ROOT"
LOG="$ROOT/results/hf_job.log"
exec > >(tee -a "$LOG") 2>&1
export DELTASHOT_ARTIFACT_REPO="$ARTIFACT_REPO"

upload_artifacts() {
  python - "$ARTIFACT_REPO" "$ROOT" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

repo, root = sys.argv[1], Path(sys.argv[2])
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
for folder, path in ((root / "checkpoints", "checkpoints"), (root / "results", "results")):
    if folder.is_dir() and any(folder.iterdir()):
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), path_in_repo=path,
                          commit_message=f"DeltaShot job artifacts: {path}")
PY
}

on_exit() {
  local exit_code="$1"
  trap - EXIT
  set +e
  echo "[cleanup] Uploading logs and partial artifacts (job exit=$exit_code)."
  upload_artifacts
  local upload_code=$?
  if [[ $upload_code -ne 0 ]]; then
    echo "[cleanup] Artifact upload failed with exit=$upload_code."
  else
    echo "[cleanup] Artifact upload finished."
  fi
  exit "$exit_code"
}
trap 'on_exit $?' EXIT

echo "[step 1] Validate downloaded bundle and prepare artifact log."
test -f bundle_manifest.json
test -f results/baselines_classical_full_test.jsonl

echo "[step 2] Check the container PyTorch/CUDA runtime."
python - <<'PY'
import os
import torch

available = torch.cuda.is_available()
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={available}")
if available:
    print(f"gpu={torch.cuda.get_device_name(0)}")
if os.environ.get("DELTASHOT_EXPECT_CUDA") == "1" and not available:
    raise RuntimeError("T4 flavor expected CUDA, but torch.cuda.is_available() is false")
PY

echo "[step 3] Install DeltaShot runtime and evaluation dependencies."
python -m pip install --no-cache-dir compressai==1.2.8 pytorch-msssim pytest pillow beautifulsoup4

echo "[step 4] Ensure the private Model artifact repository exists before epoch uploads."
python - "$ARTIFACT_REPO" <<'PY'
import os
import sys
from huggingface_hub import HfApi

HfApi(token=os.environ["HF_TOKEN"]).create_repo(
    repo_id=sys.argv[1], repo_type="model", private=True, exist_ok=True
)
print(f"Artifact Model repo ready: {sys.argv[1]}")
PY

if [[ "$MODE" != "dryrun" ]]; then
  echo "[step 5] Restore prior checkpoints and results from the artifact repository."
  python - "$ARTIFACT_REPO" "$ROOT" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo, root = sys.argv[1], Path(sys.argv[2])
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
files = api.list_repo_files(repo_id=repo, repo_type="model")
if any(path.startswith(("checkpoints/", "results/")) for path in files):
    with tempfile.TemporaryDirectory(prefix="deltashot-artifacts-") as temporary:
        snapshot_download(repo_id=repo, repo_type="model", token=True, local_dir=temporary,
                          allow_patterns=["checkpoints/**", "results/**"])
        for folder_name in ("checkpoints", "results"):
            source, destination = Path(temporary) / folder_name, root / folder_name
            if not source.is_dir():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if folder_name == "results" and item.name == "hf_job.log":
                    continue
                target = destination / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
PY
fi

echo "[step 6] Run requested pipeline mode=$MODE with automatic device selection."
if [[ "$MODE" == "dryrun" ]]; then
  python -m scripts.run_pipeline --device auto --stages preflight --skip-pytest
elif [[ "$MODE" == "calibrate" ]]; then
  python -m scripts.run_pipeline --device auto --budget-hours 0.1667 --calibration-steps 8 --calibrate-only
else
  python -m scripts.run_pipeline --device auto --budget-hours 3.5 --resume
fi
