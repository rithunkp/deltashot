"""Small runtime helpers shared by training and evaluation entrypoints."""
import random
import os
from pathlib import Path

import numpy as np
import torch


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader"])


def upload_epoch_artifacts(run_id: str, checkpoint_dir: Path, metrics_path: Path | None, epoch: int) -> None:
    """Upload the completed epoch immediately when running inside an HF Job."""
    repo_id = os.environ.get("DELTASHOT_ARTIFACT_REPO")
    if not repo_id:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("DELTASHOT_ARTIFACT_REPO is set but HF_TOKEN is unavailable")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(checkpoint_dir),
        path_in_repo=f"checkpoints/{run_id}",
        commit_message=f"Checkpoint {run_id} epoch {epoch}",
    )
    if metrics_path is not None and metrics_path.exists():
        api.upload_file(
            repo_id=repo_id, repo_type="model", path_or_fileobj=str(metrics_path),
            path_in_repo=f"metrics/{run_id}.json",
            commit_message=f"Metrics {run_id} epoch {epoch}",
        )
