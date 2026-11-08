"""Tests for the M5 loader on a tiny synthetic copy of the competition files."""

import hashlib
import io
import json
import stat
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


def test_load_m5_reads_extracted_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sales, prices, keys, calendar and weights line up on the synthetic files."""
    _serve(monkeypatch, _synthetic_archive())
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
    calls = _serve(monkeypatch, _synthetic_archive())
    digest = datasets.M5_SHA256
    monkeypatch.setattr(datasets, "M5_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="unexpected digest"):
        load_m5(cache_dir=tmp_path / "bad")
    assert list((tmp_path / "bad").iterdir()) == []

    monkeypatch.setattr(datasets, "M5_SHA256", digest)
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
    assert "sizes" not in json.loads((tmp_path / M5_MARKER).read_text())

    monkeypatch.setattr(zipfile.ZipFile, "extractall", original)
    m5 = load_m5(cache_dir=tmp_path)
    assert m5.weights.height == 1
    assert json.loads((tmp_path / M5_MARKER).read_text())["digest"] == datasets.M5_SHA256
    assert len(calls) == 1


def test_load_m5_reextracts_files_without_a_matching_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncated files, a marker of another digest and an unreadable marker are extracted again."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    (tmp_path / "m5.zip").write_bytes(payload)
    for name, text in _synthetic_files().items():
        (tmp_path / name).write_text("".join(text.splitlines(keepends=True)[:2]))
    assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)

    marker = json.loads((tmp_path / M5_MARKER).read_text())
    (tmp_path / M5_MARKER).write_text(json.dumps({**marker, "digest": "0" * 64}))
    load_m5(cache_dir=tmp_path)
    assert json.loads((tmp_path / M5_MARKER).read_text()) == marker

    (tmp_path / M5_MARKER).write_text("not json")
    load_m5(cache_dir=tmp_path)
    assert json.loads((tmp_path / M5_MARKER).read_text()) == marker
    assert calls == []


def _write_unmarked_cache(directory: Path, payload: bytes) -> None:
    """Write a cache as an earlier version left it: the archive and the files, no marker."""
    (directory / "m5.zip").write_bytes(payload)
    for name, text in _synthetic_files().items():
        (directory / name).write_text(text)


def _forbid_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def extractall(self: zipfile.ZipFile, *args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected extraction")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", extractall)


def test_load_m5_adopts_a_cache_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete cache with a matching archive and no marker is used without a new extraction."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    _write_unmarked_cache(tmp_path, payload)
    _forbid_extraction(monkeypatch)
    assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)
    assert json.loads((tmp_path / M5_MARKER).read_text())["digest"] == datasets.M5_SHA256
    assert calls == []


def test_load_m5_reads_a_read_only_cache_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only cache with a matching archive and no marker is read without writing to it."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    _write_unmarked_cache(tmp_path, payload)
    _forbid_extraction(monkeypatch)
    tmp_path.chmod(0o555)
    try:
        assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)
        assert load_m5(cache_dir=tmp_path).weights.height == 1
    finally:
        tmp_path.chmod(0o755)
    assert not (tmp_path / M5_MARKER).exists()
    assert calls == []


def test_load_m5_cache_hit_needs_no_archive_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete cache loads after the archive is removed and with no network."""
    _serve(monkeypatch, _synthetic_archive())
    load_m5(cache_dir=tmp_path)
    (tmp_path / "m5.zip").unlink()

    def offline(url: str, timeout: float) -> io.BytesIO:
        raise OSError("network unreachable")

    monkeypatch.setattr(datasets.urllib.request, "urlopen", offline)
    assert load_m5(cache_dir=tmp_path).sales.shape == (M5_DAYS, 2)


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


def test_load_m5_restores_a_deleted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file removed from a complete cache is extracted again from the archive."""
    calls = _serve(monkeypatch, _synthetic_archive())
    load_m5(cache_dir=tmp_path)
    (tmp_path / "sell_prices.csv").unlink()
    assert load_m5(cache_dir=tmp_path).price.shape == (M5_DAYS, 2)
    assert len(calls) == 1


def test_load_m5_keeps_the_archive_when_the_replacement_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``m5.zip`` with another digest stays in place until a new archive is verified."""
    _serve(monkeypatch, _synthetic_archive())
    load_m5(cache_dir=tmp_path)
    previous = (tmp_path / "m5.zip").read_bytes()
    monkeypatch.setattr(datasets, "M5_SHA256", "f" * 64)

    def offline(url: str, timeout: float) -> io.BytesIO:
        raise OSError("network unreachable")

    monkeypatch.setattr(datasets.urllib.request, "urlopen", offline)
    with pytest.raises(OSError, match="network unreachable"):
        load_m5(cache_dir=tmp_path)
    assert (tmp_path / "m5.zip").read_bytes() == previous
    assert list(tmp_path.glob("*.part")) == []


def test_load_m5_restores_a_truncated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file truncated after a complete extraction is extracted again from the archive."""
    calls = _serve(monkeypatch, _synthetic_archive())
    load_m5(cache_dir=tmp_path)
    header = _synthetic_files()["weights_evaluation.csv"].splitlines(keepends=True)[0]
    (tmp_path / "weights_evaluation.csv").write_text(header)
    assert load_m5(cache_dir=tmp_path).weights.height == 1
    assert len(calls) == 1


def test_load_m5_archive_follows_the_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The downloaded archive gets the same permission bits as the extracted files."""
    _serve(monkeypatch, _synthetic_archive())
    load_m5(cache_dir=tmp_path)
    archive_mode = stat.S_IMODE((tmp_path / "m5.zip").stat().st_mode)
    assert archive_mode == stat.S_IMODE((tmp_path / M5_FILES[0]).stat().st_mode)


def test_load_m5_redoes_an_extraction_cut_before_the_data_reached_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero-filled files of the right size next to an in-progress marker are extracted again."""
    payload = _synthetic_archive()
    calls = _serve(monkeypatch, payload)
    _write_unmarked_cache(tmp_path, payload)
    weights = tmp_path / "weights_evaluation.csv"
    weights.write_bytes(b"\0" * weights.stat().st_size)
    (tmp_path / M5_MARKER).write_text(
        json.dumps({"digest": datasets.M5_SHA256, "extracting": True})
    )
    assert load_m5(cache_dir=tmp_path).weights.height == 1
    assert calls == []
