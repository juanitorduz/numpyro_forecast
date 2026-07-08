"""Unit tests for the example-page metadata helpers in ``scripts/build_docs.py``.

The Examples index card grid is driven by front matter that ``build_docs.py``
derives from each committed notebook: the card description comes from the
notebook-level ``description`` metadata (falling back to the intro paragraph)
and the thumbnail from the stored figure outputs (a ``thumbnail`` cell tag
falling back to the first figure). These tests pin that derivation logic.
"""

import base64
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_docs.py"

# A valid 1x1 PNG, used as a stand-in for a stored matplotlib figure.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


@pytest.fixture(scope="module")
def build_docs() -> ModuleType:
    """Load ``scripts/build_docs.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("build_docs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_docs"] = module
    spec.loader.exec_module(module)
    return module


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _code_cell(png: str | list[str] | None, tags: list[str] | None = None) -> dict[str, Any]:
    outputs = [] if png is None else [{"output_type": "display_data", "data": {"image/png": png}}]
    return {
        "cell_type": "code",
        "metadata": {"tags": tags} if tags is not None else {},
        "source": [],
        "outputs": outputs,
    }


class TestNotebookDescription:
    def test_metadata_key_wins(self, build_docs: ModuleType) -> None:
        notebook = {
            "metadata": {"description": "Hand-written card copy."},
            "cells": [_markdown_cell("# Title\n\nIntro sentence. More text.")],
        }
        assert build_docs.notebook_description(notebook) == "Hand-written card copy."

    def test_fallback_extracts_intro_first_sentence(self, build_docs: ModuleType) -> None:
        source = (
            "# Title with `numpyro_forecast`\n\n"
            "This ports [**a blog post**](https://example.com) to `numpyro_forecast`. "
            "Second sentence."
        )
        notebook = {"metadata": {}, "cells": [_markdown_cell(source)]}
        assert (
            build_docs.notebook_description(notebook)
            == "This ports a blog post to numpyro_forecast."
        )

    def test_empty_when_no_intro(self, build_docs: ModuleType) -> None:
        notebook = {"metadata": {}, "cells": [_markdown_cell("# Title only")]}
        assert build_docs.notebook_description(notebook) == ""


class TestExtractThumbnail:
    def test_tagged_cell_wins_over_first_image(
        self, build_docs: ModuleType, tmp_path: Path
    ) -> None:
        tagged_png = base64.b64encode(b"tagged-figure").decode("ascii")
        notebook = {
            "metadata": {},
            "cells": [
                _code_cell(_PNG_B64),
                _code_cell(tagged_png, tags=["thumbnail"]),
            ],
        }
        dest = tmp_path / "thumb.png"
        assert build_docs.extract_thumbnail(notebook, dest)
        assert dest.read_bytes() == b"tagged-figure"

    def test_fallback_to_first_image(self, build_docs: ModuleType, tmp_path: Path) -> None:
        notebook = {
            "metadata": {},
            "cells": [_code_cell(None), _code_cell(_PNG_B64), _code_cell(_PNG_B64)],
        }
        dest = tmp_path / "thumb.png"
        assert build_docs.extract_thumbnail(notebook, dest)
        assert dest.read_bytes() == _PNG_BYTES

    def test_list_payload_is_joined(self, build_docs: ModuleType, tmp_path: Path) -> None:
        mid = len(_PNG_B64) // 2
        notebook = {"metadata": {}, "cells": [_code_cell([_PNG_B64[:mid], _PNG_B64[mid:]])]}
        dest = tmp_path / "thumb.png"
        assert build_docs.extract_thumbnail(notebook, dest)
        assert dest.read_bytes() == _PNG_BYTES

    def test_nothing_written_without_figures(self, build_docs: ModuleType, tmp_path: Path) -> None:
        notebook = {"metadata": {}, "cells": [_code_cell(None), _markdown_cell("text")]}
        dest = tmp_path / "thumb.png"
        assert not build_docs.extract_thumbnail(notebook, dest)
        assert not dest.exists()


class TestNewestSemver:
    def test_max_by_numeric_tuple(self, build_docs: ModuleType) -> None:
        assert build_docs.newest_semver(["0.1.3", "0.1.10", "0.0.9"]) == "0.1.10"

    def test_non_semver_tags_ignored(self, build_docs: ModuleType) -> None:
        assert build_docs.newest_semver(["v1.0", "foo", "0.2.0", "1.0"]) == "0.2.0"

    def test_none_without_release_tags(self, build_docs: ModuleType) -> None:
        assert build_docs.newest_semver([]) is None
        assert build_docs.newest_semver(["foo", "v1.2.3"]) is None


class TestInjectVersions:
    def test_dev_first_then_pinned_release(self, build_docs: ModuleType) -> None:
        original = 'site_url: "https://example.org/proj/"\nparser: numpy\n'
        patched = yaml.safe_load(
            build_docs.inject_versions(original, "0.1.3", ".great-docs-cache/x.json")
        )
        assert patched["versions"] == [
            {"label": "dev", "tag": "dev", "prerelease": True},
            {
                "label": "0.1.3",
                "tag": "0.1.3",
                "latest": True,
                "api_snapshot": ".great-docs-cache/x.json",
            },
        ]
        assert patched["site_url"] == "https://example.org/proj/"
        assert patched["parser"] == "numpy"


class TestEnsureReleaseSnapshot:
    def test_cached_snapshot_short_circuits(
        self, build_docs: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(build_docs, "CACHE_SNAPSHOT_DIR", tmp_path)
        cached = tmp_path / "0.1.3.json"
        cached.write_text("{}", encoding="utf-8")

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("cached snapshots must not shell out")

        monkeypatch.setattr(build_docs.subprocess, "run", explode)
        assert build_docs.ensure_release_snapshot("0.1.3") == cached

    def test_failed_worktree_add_leaves_no_tempdir(
        self, build_docs: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`git worktree remove` cannot clean a dir the failed add never registered."""
        monkeypatch.setattr(build_docs, "CACHE_SNAPSHOT_DIR", tmp_path / "cache")
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        monkeypatch.setattr(build_docs.tempfile, "mkdtemp", lambda **kwargs: str(worktree))

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            if cmd[:3] == ["git", "worktree", "add"]:
                raise build_docs.subprocess.CalledProcessError(128, cmd)

        monkeypatch.setattr(build_docs.subprocess, "run", fake_run)
        with pytest.raises(build_docs.subprocess.CalledProcessError):
            build_docs.ensure_release_snapshot("0.1.3")
        assert not worktree.exists()


class TestFixAliasRedirects:
    def test_stubs_point_at_absolute_site_url(
        self, build_docs: ModuleType, tmp_path: Path
    ) -> None:
        for name in ("latest", "stable"):
            stub = tmp_path / "v" / name / "index.html"
            stub.parent.mkdir(parents=True)
            stub.write_text('<meta http-equiv="refresh" content="0; url=/">', encoding="utf-8")
        build_docs.fix_alias_redirects(tmp_path, "https://example.org/proj/")
        for name in ("latest", "stable"):
            content = (tmp_path / "v" / name / "index.html").read_text(encoding="utf-8")
            assert 'url=https://example.org/proj/"' in content
            assert 'href="https://example.org/proj/"' in content
            assert 'url=/"' not in content

    def test_missing_trailing_slash_is_added(self, build_docs: ModuleType, tmp_path: Path) -> None:
        stub = tmp_path / "v" / "stable" / "index.html"
        stub.parent.mkdir(parents=True)
        stub.write_text("placeholder", encoding="utf-8")
        build_docs.fix_alias_redirects(tmp_path, "https://example.org/proj")
        assert 'url=https://example.org/proj/"' in stub.read_text(encoding="utf-8")

    def test_missing_stubs_are_tolerated(self, build_docs: ModuleType, tmp_path: Path) -> None:
        build_docs.fix_alias_redirects(tmp_path, "https://example.org/proj/")
        assert not (tmp_path / "v").exists()
