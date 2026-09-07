#!/usr/bin/env python3
"""Check Markdown syntax/links; execute only an explicitly requested synthetic example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import yaml


SKIPPED_SCHEMES = {"http", "https", "mailto"}


def markdown_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.md"))
        if ".git" not in path.parts and "stages" not in path.parts
    ]


def links_from_tokens(tokens: list[Any]) -> list[str]:
    links = []
    for token in tokens:
        children = token.children or []
        for child in children:
            if child.type == "link_open":
                target = child.attrGet("href")
                if target:
                    links.append(target)
    return links


def check_link(document: Path, target: str) -> str | None:
    parsed = urlsplit(target)
    if parsed.scheme in SKIPPED_SCHEMES or target.startswith("#"):
        return None
    if parsed.scheme or parsed.netloc:
        return f"unsupported link target: {target}"
    relative = unquote(parsed.path)
    if not relative:
        return None
    destination = (document.parent / relative).resolve()
    if not destination.exists():
        return f"missing local target: {target}"
    return None


def snippet_evidence(language: str, status: str, detail: str) -> dict[str, str]:
    """Syntax success must never imply API binding or workflow execution."""
    parsed = "pass" if status == "pass" else "not-run"
    return {
        "parsed": parsed,
        "compiled": "pass" if detail == "python-compile" else "not-applicable",
        "api_bound": "not-run",
        "executed_synthetic": "not-run",
        "executed_e2e": "not-run",
        "documentation_fragment": "yes" if status == "not-applicable" else "no",
        "not_run_reason": (
            "documentation-fragment" if status == "not-applicable"
            else "syntax-only; external inputs or separate workflow authorization required"
        ),
    }


def check_snippet(language: str, content: str, strict_data: bool) -> tuple[str, str]:
    normalized = language.lower().strip()
    if normalized in {"json"}:
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            if not strict_data:
                return "not-applicable", f"json-fragment: {exc}"
            raise
        return "pass", "json-parse"
    if normalized in {"yaml", "yml"}:
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as exc:
            if not strict_data:
                return "not-applicable", f"yaml-fragment: {exc}"
            raise
        return "pass", "yaml-parse"
    if normalized in {"python", "py"}:
        compile(content, "<documentation-snippet>", "exec")
        return "pass", "python-compile"
    if normalized in {"bash", "sh", "shell"}:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", encoding="utf-8") as script:
            script.write(content)
            script.flush()
            subprocess.run(["bash", "-n", script.name], check=True, capture_output=True, text=True)
        return "pass", "bash-syntax"
    return "not-applicable", normalized or "untyped"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-synthetic", action="store_true", help="Execute only the exact reconstruction_minimal.py example.")
    args = parser.parse_args()
    root = args.root.resolve()
    markdown = MarkdownIt("commonmark")
    link_results = []
    snippet_results = []
    failures = 0
    for document in markdown_files(root):
        relative = document.relative_to(root).as_posix()
        relative_path = Path(relative)
        strict_data = len(relative_path.parts) == 1 or relative_path.parts[0] == "docs"
        tokens = markdown.parse(document.read_text(encoding="utf-8"))
        occurrences: dict[str, int] = {}
        for target in links_from_tokens(tokens):
            error = check_link(document, target)
            external = urlsplit(target).scheme in SKIPPED_SCHEMES
            status = "not-checked-external" if external else ("fail" if error else "pass")
            failures += int(error is not None)
            link_results.append({"document": relative, "target": target, "status": status, "detail": error or ""})
        for index, token in enumerate((item for item in tokens if item.type == "fence"), start=1):
            content_hash = hashlib.sha256(token.content.encode("utf-8")).hexdigest()
            occurrences[content_hash] = occurrences.get(content_hash, 0) + 1
            try:
                status, detail = check_snippet(token.info, token.content, strict_data)
            except Exception as exc:  # The report preserves the exact parser failure.
                status, detail = "fail", f"{type(exc).__name__}: {exc}"
                failures += 1
            evidence = snippet_evidence(token.info, status, detail)
            if args.execute_synthetic and relative == "docs/RECONSTRUCTION_GUIDE.md" and token.info == "python":
                example = root / "samples/reconstruction_minimal.py"
                try:
                    if token.content != example.read_text(encoding="utf-8"):
                        raise ValueError("documented reconstruction code differs from tested source")
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = str(root / "src")
                    subprocess.run([sys.executable, str(example)], check=True, capture_output=True, text=True, env=environment, timeout=60)
                    evidence.update(api_bound="pass", executed_synthetic="pass", not_run_reason="end-to-end detector/model workflow not authorized")
                except Exception as exc:
                    status, detail = "fail", f"{type(exc).__name__}: {exc}"
                    failures += 1
            snippet_results.append(
                {
                    "document": relative,
                    "block_id": f"{relative}#{content_hash[:16]}-{occurrences[content_hash]}",
                    "content_sha256": content_hash,
                    "index": index,
                    "language": token.info,
                    "status": status,
                    "detail": detail,
                    **evidence,
                }
            )
    payload = {
        "schema": "xslim-documentation-check-v2",
        "root": str(root),
        "documents": len(markdown_files(root)),
        "links": link_results,
        "snippets": snippet_results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
