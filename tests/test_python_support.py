"""Prevent unsupported Python claims and runtime dependency drift."""

from pathlib import Path
import tomllib

from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[1]


def test_metadata_supports_only_the_bounded_qualified_line():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    supported = SpecifierSet(project["requires-python"])
    assert "3.12.3" in supported
    for version in ["3.9.23", "3.10.18", "3.11.13", "3.12.2", "3.13.0", "3.14.0"]:
        assert version not in supported
    classifiers = [c for c in project["classifiers"] if c.startswith("Programming Language :: Python :: 3.")]
    assert classifiers == ["Programming Language :: Python :: 3.12"]


def test_certified_constraints_preserve_numeric_dependencies():
    constraints = set((ROOT / "requirements-certified-python312.txt").read_text().splitlines())
    assert {"numpy==2.5.2", "onnx==1.21.0", "onnxruntime==1.24.3", "torch==2.13.0+cpu", "torchvision==0.28.0+cpu"} <= constraints
    dependencies = (ROOT / "requirements.txt").read_text()
    assert "onnx>=1.21.0,<1.22.0" in dependencies
    assert "onnxruntime>=1.24.0,<=1.24.3" in dependencies
