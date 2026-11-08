"""Tests for the M5 loader on a tiny synthetic copy of the competition files."""

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from numpyro_forecast import datasets
from numpyro_forecast.datasets import M5_DAYS, M5_FILES, M5_MARKER, load_m5

pytest.importorskip("polars")

N_WEEKS = -(-M5_DAYS // 7)


def _synthetic_files() -> dict[str, str]:
    """Build the five M5 files for two series, with a known price gap and one event."""
    days = [f"d_{day}" for day in range(1, M5_DAYS + 1)]
    keys = "item_id,dept_id,cat_id,store_id,state_id"
    rows = [
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("FOODS_3_090", "FOODS_3", "FOODS", "TX_2", "TX"),
    ]
    train = [f"{keys},{','.join(days[:1_941])}"]
    test = [f"{keys},{','.join(days[1_941:])}"]
    for n, row in enumerate(rows):
        train.append(",".join(row) + "," + ",".join(str(n + day % 3) for day in range(1_941)))
        test.append(",".join(row) + "," + ",".join(str(10 * (n + 1)) for _ in range(28)))
    calendar = [
        "date,wm_yr_wk,weekday,wday,month,year,event_name_1,event_type_1,"
        "event_name_2,event_type_2,snap_CA,snap_TX,snap_WI"
    ]
    start = np.datetime64("2011-01-29")
    for day in range(M5_DAYS):
        date = start + day
        event = "SuperBowl,Sporting" if day == 8 else "NA,NA"
        calendar.append(f"{date},{11101 + day // 7},Saturday,1,1,2011,{event},NA,NA,{day % 2},0,0")
    prices = ["store_id,item_id,wm_yr_wk,sell_price"]
    for week in range(N_WEEKS):
        if week != 1:
            prices.append(f"CA_1,HOBBIES_1_001,{11101 + week},{1.0 + week}")
        prices.append(f"TX_2,FOODS_3_090,{11101 + week},2.5")
    weights = ["Level_id,Agg_Level_1,Agg_Level_2,Dollar_Sales,weight", "Level1,Total,X,100.0,1.0"]
    return {
        "sales_train_evaluation.csv": "\n".join(train) + "\n",
        "sales_test_evaluation.csv": "\n".join(test) + "\n",
        "calendar.csv": "\n".join(calendar) + "\n",
        "sell_prices.csv": "\n".join(prices) + "\n",
        "weights_evaluation.csv": "\n".join(weights) + "\n",
    }


def _write_files(directory: Path) -> None:
    for name, text in _synthetic_files().items():
        (directory / name).write_text(text)
    (directory / M5_MARKER).write_text(datasets.M5_SHA256)


def _synthetic_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in _synthetic_files().items():
            archive.writestr(name, text)
    return buffer.getvalue()


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    """Pin the digest of ``payload`` and serve it as the download; return the fetched URLs."""
    calls = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets, "M5_SHA256", hashlib.sha256(payload).hexdigest())
    return calls


def test_load_m5_reads_extracted_files(tmp_path: Path) -> None:
    """Sales, prices, keys, calendar and weights line up on the synthetic files."""
    _write_files(tmp_path)
    m5 = load_m5(cache_dir=tmp_path)
    assert m5.sales.shape == (M5_DAYS, 2)
    assert m5.sales.dtype == np.float32
    assert m5.sales[1_941:, 1].tolist() == [20.0] * 28
    assert m5.keys["id"].to_list() == ["HOBBIES_1_001_CA_1", "FOODS_3_090_TX_2"]
    assert np.isnan(m5.price[7:14, 0]).all()
    assert m5.price[0:7, 0].tolist() == [1.0] * 7
    assert m5.price[14:21, 0].tolist() == [3.0] * 7
    assert m5.price[:, 1].tolist() == [2.5] * M5_DAYS
    assert m5.calendar.height == M5_DAYS
    assert m5.calendar["event_name_1"].null_count() == M5_DAYS - 1
    assert m5.calendar["event_name_1"][8] == "SuperBowl"
    assert m5.weights.columns == [
        "Level_id",
        "Agg_Level_1",
        "Agg_Level_2",
        "Dollar_Sales",
        "weight",
    ]


def test_load_m5_downloads_verifies_and_unpacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive is fetched once, checked against its digest and unpacked; a bad digest is discarded."""
    payload = _synthetic_archive()
    calls = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets, "M5_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="unexpected digest"):
        load_m5(cache_dir=tmp_path / "bad")
    assert not (tmp_path / "bad" / "m5.zip").exists()
    assert not (tmp_path / "bad" / "m5.zip.part").exists()

    monkeypatch.setattr(datasets, "M5_SHA256", hashlib.sha256(payload).hexdigest())
    m5 = load_m5(cache_dir=tmp_path / "good")
    assert m5.sales.shape == (M5_DAYS, 2)
    assert all((tmp_path / "good" / name).exists() for name in M5_FILES)
    load_m5(cache_dir=tmp_path / "good")
    assert len(calls) == 2


def test_load_m5_rejects_non_https_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only https archives are fetched."""
    monkeypatch.setattr(datasets, "M5_URL", "file:///etc/passwd")
    with pytest.raises(ValueError, match="https"):
        load_m5(cache_dir=tmp_path)


def test_load_m5_redoes_an_interrupted_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extraction that fails in the last member writes no marker and is redone in full."""
    calls = _serve(monkeypatch, _synthetic_archive())
    original = zipfile.ZipFile.extractall

    def failing_extractall(
        self: zipfile.ZipFile, path: str | Path, members: tuple[str, ...]
    ) -> None:
        original(self, path, members=members[:-1])
        Path(path, members[-1]).write_text("Level_id,Agg_Level_1\n")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", failing_extractall)
    with pytest.raises(OSError, match="No space left"):
        load_m5(cache_dir=tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["m5.zip"]

    monkeypatch.setattr(zipfile.ZipFile, "extractall", original)
    m5 = load_m5(cache_dir=tmp_path)
    assert m5.weights.height == 1
    assert (tmp_path / M5_MARKER).read_text() == datasets.M5_SHA256
    assert len(calls) == 1


def test_load_m5_reextracts_files_without_a_matching_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncated files with no marker, or a marker of another digest, are replaced from the archive."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    (tmp_path / "m5.zip").write_bytes(payload)
    for name, text in _synthetic_files().items():
        (tmp_path / name).write_text("".join(text.splitlines(keepends=True)[:2]))
    assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)

    (tmp_path / "sales_train_evaluation.csv").write_text("truncated\n")
    (tmp_path / M5_MARKER).write_text("0" * 64)
    assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)
    assert calls == []


def test_load_m5_replaces_an_archive_with_another_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``m5.zip`` already in the cache is verified and downloaded again on a mismatch."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    with zipfile.ZipFile(tmp_path / "m5.zip", "w") as archive:
        for name in M5_FILES:
            archive.writestr(name, "col\nNOT THE PINNED ARCHIVE\n")
    m5 = load_m5(cache_dir=tmp_path)
    assert m5.sales.shape == (M5_DAYS, 2)
    assert (tmp_path / "m5.zip").read_bytes() == payload
    assert len(calls) == 1


def test_load_m5_removes_the_partial_download_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that fails mid-read leaves neither the ``.part`` file nor the archive."""
    payload = _synthetic_archive()

    class FailingResponse(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() > len(payload) // 2:
                raise TimeoutError("The read operation timed out")
            return super().read(1_024)

    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda url, timeout: FailingResponse(payload)
    )
    with pytest.raises(TimeoutError):
        load_m5(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
