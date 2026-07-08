"""Unit tests for the reference-keyed snapshot generator ``scripts/api_snapshot.py``.

The multi-version docs build prunes a release's API reference pages by matching
page stems against snapshot keys, so the generator must key symbols by the
curated ``reference:`` entry names, must distinguish "not in this package tree"
(skipped, hence pruned) from a broken environment (raise), and must emit the
snapshot schema great-docs reads. These tests pin that behavior against the
installed working tree.
"""

import importlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "api_snapshot.py"


@pytest.fixture(scope="module")
def api_snapshot() -> ModuleType:
    """Load ``scripts/api_snapshot.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("api_snapshot", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["api_snapshot"] = module
    spec.loader.exec_module(module)
    return module


class TestReferenceEntries:
    def test_collects_all_sections_in_order(
        self, api_snapshot: ModuleType, tmp_path: Path
    ) -> None:
        config = tmp_path / "great-docs.yml"
        config.write_text(
            "reference:\n"
            "  - title: A\n"
            "    contents:\n"
            "      - forecaster.Forecaster\n"
            "      - name: evaluate.backtest\n"
            "  - title: B\n"
            "    contents:\n"
            "      - metrics.crps_empirical\n",
            encoding="utf-8",
        )
        assert api_snapshot.reference_entries(config) == [
            "forecaster.Forecaster",
            "evaluate.backtest",
            "metrics.crps_empirical",
        ]

    def test_empty_without_reference_section(
        self, api_snapshot: ModuleType, tmp_path: Path
    ) -> None:
        config = tmp_path / "great-docs.yml"
        config.write_text("parser: numpy\n", encoding="utf-8")
        assert api_snapshot.reference_entries(config) == []


class TestResolveEntry:
    def test_resolves_module_attribute(self, api_snapshot: ModuleType) -> None:
        from numpyro_forecast.forecaster import Forecaster

        assert api_snapshot.resolve_entry("forecaster.Forecaster") is Forecaster

    def test_dotless_entry_resolves_against_the_package_root(
        self, api_snapshot: ModuleType
    ) -> None:
        import numpyro_forecast

        assert api_snapshot.resolve_entry("Forecaster") is numpyro_forecast.Forecaster

    def test_missing_module_path_is_missing(self, api_snapshot: ModuleType) -> None:
        assert api_snapshot.resolve_entry("nonexistent.thing") is api_snapshot._MISSING

    def test_missing_attribute_is_missing(self, api_snapshot: ModuleType) -> None:
        assert api_snapshot.resolve_entry("forecaster.NotAThing") is api_snapshot._MISSING

    def test_foreign_import_error_propagates(
        self, api_snapshot: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing third-party dependency must fail loudly, not shrink the snapshot."""

        def fake_import(name: str, package: str | None = None) -> object:
            raise ModuleNotFoundError("boom", name="blackjax")

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(ModuleNotFoundError, match="boom"):
            api_snapshot.resolve_entry("contrib.blackjax.BlackjaxNUTSKernel")


class TestSymbolKind:
    def test_classifies_class_function_attribute(self, api_snapshot: ModuleType) -> None:
        assert api_snapshot.symbol_kind(int) == "class"
        assert api_snapshot.symbol_kind(len) == "function"
        assert api_snapshot.symbol_kind(42) == "attribute"


class TestBuildSnapshot:
    def test_symbols_keyed_by_entry_with_kind_and_bases(self, api_snapshot: ModuleType) -> None:
        entries = [
            "forecaster.Forecaster",
            "evaluate.backtest",
            "exceptions.BacktestWindowError",
            "functional.svi.not_a_symbol",
            "gone.module.Thing",
        ]
        snapshot, skipped = api_snapshot.build_snapshot(entries, "9.9.9")
        assert snapshot["version"] == "9.9.9"
        assert snapshot["package_name"] == "numpyro_forecast"
        symbols = snapshot["symbols"]
        assert set(symbols) == {
            "forecaster.Forecaster",
            "evaluate.backtest",
            "exceptions.BacktestWindowError",
        }
        assert symbols["forecaster.Forecaster"]["kind"] == "class"
        assert symbols["evaluate.backtest"] == {"name": "evaluate.backtest", "kind": "function"}
        assert symbols["exceptions.BacktestWindowError"]["bases"] == [
            "NumpyroForecastError",
            "ValueError",
        ]
        assert skipped == ["functional.svi.not_a_symbol", "gone.module.Thing"]


class TestMain:
    def test_writes_snapshot_for_the_working_tree(
        self, api_snapshot: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "snap.json"
        monkeypatch.setattr(sys, "argv", ["api_snapshot.py", "--output", str(output)])
        assert api_snapshot.main() == 0

        snapshot = json.loads(output.read_text(encoding="utf-8"))
        pyproject = tomllib.loads(
            (api_snapshot.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert snapshot["version"] == pyproject["project"]["version"]
        # Every reference entry resolves against the working tree, so nothing is pruned
        # (test_docs_reference.py guarantees the entries stay resolvable).
        assert set(snapshot["symbols"]) == set(
            api_snapshot.reference_entries(api_snapshot.CONFIG_PATH)
        )
        assert all("." in name for name in snapshot["symbols"])
