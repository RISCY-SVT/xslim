"""Regression coverage for downstream release-maintenance tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / "release" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_source_archives(tmp_path):
    tool = load_tool("build_release.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("release\n", encoding="utf-8")
    (source / "executable.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (source / "executable.sh").chmod(0o755)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    files = tool.source_files(source)
    tool.write_tar_gz(first, source, files, "xslim-test", 1_800_000_000)
    tool.write_tar_gz(second, source, files, "xslim-test", 1_800_000_000)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["xslim-test/README.md", "xslim-test/executable.sh"]


def test_spdx_document_is_deterministic_and_complete(tmp_path):
    tool = load_tool("build_release.py")
    (tmp_path / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    files = [Path("LICENSE")]
    first = tool.spdx_document(tmp_path, files, "a" * 40, 1_800_000_000)
    second = tool.spdx_document(tmp_path, files, "a" * 40, 1_800_000_000)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["spdxVersion"] == "SPDX-2.3"
    assert first["packages"][0]["licenseDeclared"] == "Apache-2.0"
    assert first["files"][0]["fileName"] == "./LICENSE"


def test_source_archive_excludes_dataset_and_model_payloads(tmp_path):
    tool = load_tool("build_release.py")
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "sample.npy").write_bytes(b"dataset")
    (tmp_path / "model.onnx").write_bytes(b"model")
    assert tool.source_files(tmp_path) == [Path("module.py")]


def test_sdist_normalization_removes_archive_time_variance(tmp_path):
    tool = load_tool("build_release.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "PKG-INFO").write_text("Metadata-Version: 2.4\n", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    tool.write_tar_gz(first, source, [Path("PKG-INFO")], "xslim-test", 1_700_000_000)
    tool.write_tar_gz(second, source, [Path("PKG-INFO")], "xslim-test", 1_800_000_000)
    assert first.read_bytes() != second.read_bytes()
    tool.normalize_sdist(first, 1_900_000_000)
    tool.normalize_sdist(second, 1_900_000_000)
    assert first.read_bytes() == second.read_bytes()


def test_syntax_pass_does_not_claim_api_binding_or_execution():
    tool = load_tool("check_docs.py")
    status, detail = tool.check_snippet("python", "missing_api(unknown_argument=True)\n", True)
    assert status == "pass"
    evidence = tool.snippet_evidence("python", status, detail)
    assert evidence["compiled"] == "pass"
    assert evidence["api_bound"] == "not-run"
    assert evidence["executed_synthetic"] == "not-run"
    assert evidence["executed_e2e"] == "not-run"


def test_documentation_fragment_is_explicit():
    tool = load_tool("check_docs.py")
    status, detail = tool.check_snippet("text", "expected output\n", True)
    evidence = tool.snippet_evidence("text", status, detail)
    assert evidence["documentation_fragment"] == "yes"
    assert evidence["parsed"] == "not-run"
