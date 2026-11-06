"""Download every checkpoint the experiments use into the Hugging Face cache ($HF_HOME).

Run where there is network access; the SLURM jobs then run with HF_HUB_OFFLINE=1:
    python scripts/download_models.py
"""

import sys
import time

from huggingface_hub import snapshot_download

WEIGHTS = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model", "*.py"]

MODELS = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "mistralai/Timestral-28M",
    "google/timesfm-2.5-200m-pytorch",
]


def main() -> int:
    failed = []
    for repo in MODELS:
        start = time.time()
        try:
            path = snapshot_download(repo, allow_patterns=WEIGHTS)
            print(f"ok   {repo} ({time.time() - start:.0f}s) -> {path}", flush=True)
        except Exception as exc:  # keep going; report at the end
            failed.append(repo)
            print(f"FAIL {repo}: {exc}", flush=True)
    print("failed:", failed if failed else "none", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
