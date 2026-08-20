# Modified by RISCY-SVT in 2026: distinguish immutable release provenance from downstream development.
"""RISCY-SVT release provenance and publication safety regressions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c"
UPSTREAM_TREE = "05d2c8425ab8587abf401fa5976a08d008fdd719"
UPSTREAM_VERSION = "2.1.2"


def test_upstream_provenance_constants_are_exact():
    upstream = (ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "https://github.com/spacemit-com/xslim" in upstream
    assert UPSTREAM_COMMIT in upstream
    assert UPSTREAM_TREE in upstream
    assert "05d2c842fb4407bed80fb688c533e43079850dd1" not in upstream
    assert f"`{UPSTREAM_VERSION}`" in upstream
    assert (ROOT / "VERSION_NUMBER").read_text().strip() == "2.1.2+riscy.2.dev2"
    assert "v2.1.2-riscy.1" in (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")


def test_downstream_publish_workflow_is_fail_closed():
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    guard = "if: github.repository == 'spacemit-com/xslim'"
    assert workflow.count(guard) == 2
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "Modified by RISCY-SVT" in workflow
    assert "RISCY-SVT/xslim'" not in workflow


def test_development_evidence_is_not_part_of_package_manifest():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft stages" not in manifest
    assert "recursive-include stages" not in manifest
    assert "global-exclude *.py[cod]" in manifest
    assert "prune **/__pycache__" in manifest


def test_release_notes_keep_validated_and_unvalidated_claims_separate():
    notes = (ROOT / "RELEASE_NOTES.md").read_text()
    assert "config_stage64_repro.json" in notes
    assert "config_accuracy_starting_point.json" in notes
    assert "has not passed K1X board or COCO validation" in notes
