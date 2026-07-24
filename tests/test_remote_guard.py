from pathlib import Path

import pytest

from moge.utils.remote_guard import assert_safe_path


def test_guard_accepts_project_path(tmp_path: Path):
    safe_root = tmp_path / "zhuzichao"
    target = safe_root / "project" / "run"
    target.mkdir(parents=True)
    assert assert_safe_path(target, safe_root=safe_root, must_exist=True) == target


def test_guard_rejects_parent_escape(tmp_path: Path):
    safe_root = tmp_path / "zhuzichao"
    safe_root.mkdir()
    with pytest.raises(PermissionError):
        assert_safe_path(tmp_path / "other", safe_root=safe_root)


def test_guard_rejects_dataset_write(tmp_path: Path):
    safe_root = tmp_path / "zhuzichao"
    dataset = safe_root / "datasets" / "sample"
    dataset.mkdir(parents=True)
    with pytest.raises(PermissionError):
        assert_safe_path(dataset, safe_root=safe_root, writable=True)
