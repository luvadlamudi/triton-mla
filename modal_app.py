"""Run tests and benchmarks on a cloud GPU via Modal.

    modal run modal_app.py --cmd "pytest -q"
    modal run modal_app.py --cmd "triton-mla-bench --batch 1 32 128"
    MODAL_GPU=H100 modal run modal_app.py --cmd "nvidia-smi"
"""

import os

import modal

GPU = os.environ.get("MODAL_GPU", "B200")
REPO = "/root/triton-mla"
IGNORE = [".venv", ".venv/**", ".git", ".git/**", "build", "build/**", "**/__pycache__", "**/*.egg-info", ".pytest_cache"]

cache = modal.Volume.from_name("triton-mla-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "numpy", "pytest")
    .env(
        {
            "PYTHONPATH": f"{REPO}/src",
            "TRITON_CACHE_DIR": "/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
        }
    )
    .add_local_dir(".", remote_path=REPO, ignore=IGNORE)
)

app = modal.App("triton-mla", image=image)


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def run(cmd: str):
    import subprocess

    cmd = cmd.replace("triton-mla-bench", "python -m triton_mla.bench")
    result = subprocess.run(cmd, shell=True, cwd=REPO)
    cache.commit()
    return result.returncode


@app.local_entrypoint()
def main(cmd: str = "nvidia-smi && pytest -q"):
    code = run.remote(cmd)
    if code:
        raise SystemExit(code)
