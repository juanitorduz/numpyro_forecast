"""Tests for the dataset loaders."""

import hashlib
import io
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import NoReturn

import pytest

from numpyro_forecast import datasets
from numpyro_forecast.datasets import (
    bart_available,
    load_bart_hierarchical,
    load_breakfast_at_the_frat,
    load_victoria_electricity,
)


def test_bart_available_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed download (any exception) makes ``bart_available`` return False."""

    def boom() -> NoReturn:
        raise RuntimeError("download failed")

    monkeypatch.setattr(datasets, "_load_counts", boom)
    assert bart_available() is False


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_load_bart_hierarchical_rejects_oversized_window() -> None:
    """An over-large training window must fail fast, not silently wrap."""
    with pytest.raises(ValueError, match="exceeds available history"):
        load_bart_hierarchical(train_days=10**6)


def test_load_victoria_electricity_shapes_and_values() -> None:
    """The bundled CSV loads into aligned demand/temperature arrays."""
    demand, temperature = load_victoria_electricity()
    assert demand.shape == (1_344, 1)  # eight weeks of hourly data
    assert temperature.shape == (1_344,)
    assert float(demand[0, 0]) == pytest.approx(3.794, abs=1e-3)
    assert float(temperature[0]) == pytest.approx(18.05, abs=1e-3)


# --- Breakfast at the Frat -----------------------------------------------------

FIXTURE_WORKBOOK = Path(__file__).parent / "data" / "breakfast_at_the_frat_sample.xlsx"
"""A three-sheet workbook with the real layout (title row, header, a few rows)."""
CACHED_WORKBOOK = datasets._default_cache_dir() / "breakfast_at_the_frat.xlsx"


def _no_download(url: str, timeout: float) -> NoReturn:
    msg = f"unexpected download of {url} (timeout {timeout})"
    raise AssertionError(msg)


class _StalledResponse(io.BytesIO):
    def read(self, size: int | None = -1, /) -> bytes:
        raise TimeoutError("timed out")


def _fixture_digest() -> str:
    return hashlib.sha256(FIXTURE_WORKBOOK.read_bytes()).hexdigest()


def test_breakfast_loader_downloads_once_and_lowercases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first call downloads with a timeout into the cache, later calls reuse it, and everything is lowercase."""
    pl = pytest.importorskip("polars")
    calls: list[tuple[str, float]] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(FIXTURE_WORKBOOK.read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets, "BREAKFAST_AT_THE_FRAT_SHA256", _fixture_digest())
    frat = load_breakfast_at_the_frat(cache_dir=str(tmp_path))
    assert calls == [(datasets.BREAKFAST_AT_THE_FRAT_URL, datasets.DOWNLOAD_TIMEOUT)]
    assert (tmp_path / "breakfast_at_the_frat.xlsx").exists()
    for frame in (frat.transactions, frat.products, frat.stores):
        assert frame.columns == [name.lower() for name in frame.columns]
        for name, dtype in frame.schema.items():
            if dtype == pl.String:
                assert frame[name].str.to_lowercase().equals(frame[name])
    assert frat.transactions.shape == (4, 12)
    assert frat.transactions.schema["week_end_date"] == pl.Date
    assert frat.transactions["feature"].to_list() == [0, 1, 0, 0]
    assert frat.products["category"].to_list() == ["cold cereal", "cold cereal", "bag snacks"]
    assert frat.stores["seg_value_name"].to_list() == ["mainstream", "mainstream", "upscale"]
    assert frat.stores["store_id"].is_duplicated().any()  # the loader keeps the workbook as is
    load_breakfast_at_the_frat(cache_dir=tmp_path)
    assert len(calls) == 1  # cached: no second download


def test_breakfast_loader_rejects_wrong_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file with the wrong digest fails fast, names the digest and never downloads."""
    monkeypatch.setattr(urllib.request, "urlopen", _no_download)
    (tmp_path / "breakfast_at_the_frat.xlsx").write_bytes(b"not a workbook")
    with pytest.raises(ValueError, match=r"unexpected workbook digest .* delete the file"):
        load_breakfast_at_the_frat(cache_dir=tmp_path)


def test_breakfast_loader_discards_a_corrupt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download with the wrong digest is deleted, so the next call downloads again."""

    def corrupt_urlopen(url: str, timeout: float) -> io.BytesIO:
        return io.BytesIO(b"truncated")

    monkeypatch.setattr(urllib.request, "urlopen", corrupt_urlopen)
    with pytest.raises(ValueError, match="the file was discarded"):
        load_breakfast_at_the_frat(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_breakfast_loader_propagates_a_stalled_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read timeout raises and leaves no workbook in the cache."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _StalledResponse(b""))
    with pytest.raises(TimeoutError):
        load_breakfast_at_the_frat(cache_dir=tmp_path)
    assert not (tmp_path / "breakfast_at_the_frat.xlsx").exists()


def test_breakfast_loader_requires_dataframes_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``fastexcel`` surfaces the ``dataframes`` install hint through ``require``."""
    monkeypatch.setattr(urllib.request, "urlopen", _no_download)
    shutil.copyfile(FIXTURE_WORKBOOK, tmp_path / "breakfast_at_the_frat.xlsx")
    monkeypatch.setattr(datasets, "BREAKFAST_AT_THE_FRAT_SHA256", _fixture_digest())
    monkeypatch.setitem(sys.modules, "fastexcel", None)
    with pytest.raises(ImportError, match=r"pip install numpyro_forecast\[dataframes\]"):
        load_breakfast_at_the_frat(cache_dir=tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not CACHED_WORKBOOK.exists(), reason="the workbook is not cached locally")
def test_breakfast_loader_real_workbook_shapes() -> None:
    """The cached real workbook loads with the documented sizes (no network in CI)."""
    frat = load_breakfast_at_the_frat()
    assert frat.transactions.shape == (524_950, 12)
    assert frat.products.shape == (58, 6)
    assert frat.stores.shape == (79, 9)
    assert frat.transactions["store_num"].n_unique() == 77
