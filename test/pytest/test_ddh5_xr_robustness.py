"""Robustness tests for ddh5_xr: stale locks, corrupt files, odd folder names."""

import time

import numpy as np
import pytest

from labcore.data.ddh5_xr import (
    DDH5Writer_swmr,
    ddh5_schema,
    ddh5_to_xarray,
    find_data,
)


def _write(folder, n=20):
    folder.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, n)
    structure = {
        "x": {"dtype": np.float64, "shape": ()},
        "y": {"dtype": np.float64, "shape": (), "axes": ["x"]},
    }
    with DDH5Writer_swmr(structure, filepath=folder / "data.ddh5") as w:
        w.add_data(x=x, y=x ** 2)
    return folder / "data.ddh5"


def test_load_ignores_stale_lock_file(tmp_path):
    path = _write(tmp_path / "2026-01-01T120000_abcd-ok")
    (path.parent / "~data.lock").write_bytes(b"")

    t0 = time.monotonic()
    ds = ddh5_to_xarray(path, file_timeout=2)
    assert time.monotonic() - t0 < 2
    assert "y" in ds.data_vars
    assert ds.sizes["x"] == 20


def test_load_does_not_create_lock_file(tmp_path):
    path = _write(tmp_path / "2026-01-01T120000_abcd-ok")
    ddh5_to_xarray(path)
    assert not (path.parent / "~data.lock").exists()


def test_schema_matches_after_refactor(tmp_path):
    path = _write(tmp_path / "2026-01-01T120000_abcd-ok")
    schema = ddh5_schema(path)
    assert schema.dependents == ["y"]
    assert schema.independents == ["x"]
    assert schema.nrecords == 20


def test_corrupt_file_fails_fast(tmp_path):
    folder = tmp_path / "2026-01-01T120000_abcd-bad"
    folder.mkdir()
    (folder / "data.ddh5").write_bytes(b"\x01" * 4096)

    t0 = time.monotonic()
    with pytest.raises(OSError):
        ddh5_to_xarray(folder / "data.ddh5", file_timeout=10)
    assert time.monotonic() - t0 < 2


def test_reader_file_locking_opt_out(tmp_path, monkeypatch):
    import h5py
    from labcore.data import ddh5_xr

    path = _write(tmp_path / "2026-01-01T120000_abcd-ok")
    seen = []
    real_file = h5py.File

    def spy(*args, **kwargs):
        seen.append(kwargs.get("locking", "default"))
        return real_file(*args, **kwargs)

    monkeypatch.setattr(ddh5_xr.h5py, "File", spy)
    ddh5_to_xarray(path)
    assert seen and all(v == "default" for v in seen)   # unchanged default

    seen.clear()
    monkeypatch.setattr(ddh5_xr.DDH5SWMRReader, "file_locking", False)
    ddh5_to_xarray(path)
    assert seen and all(v is False for v in seen)


def test_find_data_skips_folders_without_timestamp(tmp_path):
    good = tmp_path / "2026-01-01" / "2026-01-01T120000_abcd-ok"
    good.mkdir(parents=True)
    (good / "data.ddh5").write_bytes(b"")
    bad = tmp_path / "2026-01-01" / "scratch_copy"
    bad.mkdir()
    (bad / "data.ddh5").write_bytes(b"")

    found = find_data(tmp_path)
    assert list(found) == [good]
