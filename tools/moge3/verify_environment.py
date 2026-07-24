from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the isolated MoGe-3 server environment")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = assert_safe_path(args.report, safe_root=args.safe_root, writable=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import accelerate
    import spconv
    import spconv.pytorch as spconv_torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_rows = []
    for index in range(torch.cuda.device_count()):
        # SpConv initializes per CUDA context. DDP uses one process per device;
        # the actual convolution check is therefore run by smoke_ssr.py/accelerate.
        properties = torch.cuda.get_device_properties(index)
        gpu_rows.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )

    cache_variables = {
        name: os.environ.get(name)
        for name in (
            "TMPDIR",
            "HF_HOME",
            "TORCH_HOME",
            "PIP_CACHE_DIR",
            "XDG_CACHE_HOME",
            "CONDA_PKGS_DIRS",
        )
    }
    for name, value in cache_variables.items():
        if value is None:
            raise RuntimeError(f"Required cache environment variable is unset: {name}")
        assert_safe_path(value, safe_root=args.safe_root, writable=True)

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": __import__("torchvision").__version__,
        "spconv": spconv.__version__,
        "accelerate": accelerate.__version__,
        "spconv_module": spconv_torch.__name__,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpu_rows,
        "cache_environment": cache_variables,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
