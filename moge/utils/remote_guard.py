from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_SAFE_ROOT = Path("/mnt/data/home/zhuzichao")


def _resolved(path: Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def assert_safe_path(
    path: Path,
    *,
    safe_root: Path = DEFAULT_SAFE_ROOT,
    must_exist: bool = False,
    writable: bool = False,
) -> Path:
    """Resolve a server path and reject every path outside the user-owned root."""
    safe_root = _resolved(safe_root)
    path = Path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    resolved = _resolved(path)
    try:
        relative = resolved.relative_to(safe_root)
    except ValueError as exc:
        raise PermissionError(f"Path escapes safe root {safe_root}: {resolved}") from exc
    if relative == Path("."):
        raise PermissionError("The safe root itself is too broad to use as an operation target")
    if writable:
        dataset_root = safe_root / "datasets"
        try:
            resolved.relative_to(dataset_root)
        except ValueError:
            pass
        else:
            raise PermissionError(f"Dataset paths are read-only: {resolved}")
    return resolved


def assert_read_path(
    path: Path,
    *,
    safe_root: Path = DEFAULT_SAFE_ROOT,
    allowed_read_roots: Iterable[Path] = (),
    must_exist: bool = True,
) -> Path:
    """
    Validate a read-only input.

    Inputs normally live below ``safe_root``.  A caller may additionally grant
    exact read-only roots (for example a mounted dataset); this never expands
    the set of writable locations accepted by :func:`assert_safe_path`.
    """
    path = Path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    resolved = _resolved(path)
    roots = (_resolved(safe_root), *(_resolved(root) for root in allowed_read_roots))
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise PermissionError(f"Read path is outside allowed roots ({allowed}): {resolved}")
    return resolved


def validate_paths(
    read_paths: Iterable[Path],
    write_paths: Iterable[Path],
    safe_root: Path = DEFAULT_SAFE_ROOT,
    allowed_read_roots: Iterable[Path] = (),
) -> dict:
    reads = [
        str(
            assert_read_path(
                path,
                safe_root=safe_root,
                allowed_read_roots=allowed_read_roots,
                must_exist=True,
            )
        )
        for path in read_paths
    ]
    writes = [
        str(assert_safe_path(path, safe_root=safe_root, writable=True))
        for path in write_paths
    ]
    return {
        "safe_root": str(_resolved(safe_root)),
        "allowed_read_roots": [str(_resolved(path)) for path in allowed_read_roots],
        "read": reads,
        "write": writes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MoGe-3 server paths before any training/evaluation operation."
    )
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--allowed-read-root", type=Path, action="append", default=[])
    parser.add_argument("--read", type=Path, action="append", default=[])
    parser.add_argument("--write", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            validate_paths(
                args.read,
                args.write,
                args.safe_root,
                args.allowed_read_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
