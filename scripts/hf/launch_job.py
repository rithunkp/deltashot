"""Submit a DeltaShot Hugging Face Job using an argv list, never a shell string."""
import argparse
import inspect
import os
import sys

from huggingface_hub import HfApi


IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
ROOT_IN_JOB = "/workspace/deltashot_bundle"
FLAVOR_RATES = {"cpu-upgrade": 0.03, "t4-small": 0.40}
MODE_SETTINGS = {
    "dryrun": ("60m", 1.0),
    "calibrate": ("15m", 0.25),
    "full": ("4.5h", 4.5),
}


def build_job_command(bundle_repo, mode):
    bootstrap = (
        "import os,subprocess,sys; "
        "subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub'],check=True); "
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id=sys.argv[1],repo_type='dataset',token=True,local_dir={ROOT_IN_JOB!r}); "
        f"os.chdir({ROOT_IN_JOB!r}); "
        "os.execvp('bash',['bash','scripts/hf/job_entrypoint.sh',sys.argv[2]])"
    )
    return ["python", "-c", bootstrap, bundle_repo, mode]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-repo", required=True)
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--mode", choices=MODE_SETTINGS, default="calibrate")
    parser.add_argument("--flavor", choices=FLAVOR_RATES, default="t4-small")
    return parser.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required; it will not be printed.")

    signature = inspect.signature(HfApi.run_job)
    required_parameters = {"command", "flavor", "timeout", "secrets", "token"}
    if not required_parameters.issubset(signature.parameters):
        raise SystemExit(f"Installed huggingface_hub run_job API is incompatible: {signature}")

    timeout, hours = MODE_SETTINGS[args.mode]
    rate = FLAVOR_RATES[args.flavor]
    maximum_cost = rate * hours
    cost_label = "about $0.01" if maximum_cost < 0.01 else f"${maximum_cost:.2f}"
    print(f"Mode={args.mode} flavor={args.flavor} timeout={timeout}; maximum cost estimate {cost_label}.")
    if input("Submit this Hugging Face Job? Type YES to continue: ") != "YES":
        print("Cancelled; no job submitted.")
        return

    # This argv vector avoids PowerShell and shell parsing of a command string.
    command = build_job_command(args.bundle_repo, args.mode)
    expected_cuda = "1" if args.flavor == "t4-small" else "0"
    api = HfApi(token=token)
    job = api.run_job(
        image=IMAGE,
        command=command,
        env={
            "HF_BUNDLE_REPO": args.bundle_repo,
            "HF_ARTIFACT_REPO": args.artifact_repo,
            "DELTASHOT_BUNDLE_DIR": ROOT_IN_JOB,
            "DELTASHOT_EXPECT_CUDA": expected_cuda,
        },
        secrets={"HF_TOKEN": token},
        flavor=args.flavor,
        timeout=timeout,
        name=f"deltashot-{args.mode}-{args.flavor}",
        token=token,
    )
    print(f"Submitted Hugging Face Job: {job}")


if __name__ == "__main__":
    main()
