#!/usr/bin/env python3
"""Build deterministic downstream release assets from an immutable source tree."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Iterable, Sequence
import zipfile


VERSION = "2.1.2+riscy.2"
TAG = "v2.1.2-riscy.2"
UPSTREAM_COMMIT = "9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c"
UPSTREAM_TREE = "05d2c8425ab8587abf401fa5976a08d008fdd719"
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "stages",
}
ROOT_DOCUMENTS = {
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DEVELOPMENT_NOTES.md",
    "INSTALL.md",
    "LICENSE",
    "LICENSE_AUDIT.md",
    "MODIFICATIONS.md",
    "NOTICE-RISCY-SVT",
    "QUICKSTART.md",
    "README.md",
    "README_zh.md",
    "RELEASE_NOTES.md",
    "SECURITY.md",
    "SUPPORT.md",
    "THIRD_PARTY_LICENSES.tsv",
    "THIRD_PARTY_NOTICES.md",
    "UPSTREAM.md",
}
DOC_SUFFIXES = {".json", ".md", ".py", ".sh", ".tsv", ".yaml", ".yml"}
EXCLUDED_SUFFIXES = {
    ".bin",
    ".bmp",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".o",
    ".onnx",
    ".ort",
    ".pb",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".so",
    ".tflite",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.is_symlink():
            raise RuntimeError(f"release source contains a symlink: {relative}")
        if path.is_file():
            result.append(relative)
    return result


def documentation_files(files: Iterable[Path]) -> list[Path]:
    selected = []
    for relative in files:
        if relative.as_posix() in ROOT_DOCUMENTS:
            selected.append(relative)
        elif relative.parts[0] in {"doc", "docs"}:
            selected.append(relative)
        elif relative.parts[0] == "samples" and relative.suffix in DOC_SUFFIXES:
            selected.append(relative)
    return selected


def write_tar_gz(
    destination: Path,
    root: Path,
    files: Sequence[Path],
    prefix: str,
    epoch: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative in sorted(files):
                    path = root / relative
                    info = tarfile.TarInfo(f"{prefix}/{relative.as_posix()}")
                    info.size = path.stat().st_size
                    info.mtime = epoch
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
                    with path.open("rb") as source:
                        archive.addfile(info, source)


def spdx_document(root: Path, files: Sequence[Path], source_commit: str, epoch: int) -> dict[str, object]:
    created = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    sha1_values = []
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package-xslim",
        }
    ]
    for index, relative in enumerate(sorted(files), start=1):
        path = root / relative
        sha1 = digest(path, "sha1")
        sha1_values.append(sha1)
        identifier = f"SPDXRef-File-{index:04d}"
        entries.append(
            {
                "SPDXID": identifier,
                "fileName": f"./{relative.as_posix()}",
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": sha1},
                    {"algorithm": "SHA256", "checksumValue": digest(path)},
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package-xslim",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )
    verification = hashlib.sha1("".join(sorted(sha1_values)).encode("ascii")).hexdigest()
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "name": f"xslim-{VERSION}",
        "documentNamespace": f"https://github.com/RISCY-SVT/xslim/spdx/{VERSION}/{source_commit}",
        "creationInfo": {
            "created": created,
            "creators": ["Organization: RISCY-SVT", "Tool: tools/release/build_release.py"],
            "licenseListVersion": "3.25",
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-xslim",
                "name": "xslim",
                "versionInfo": VERSION,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "packageVerificationCode": {"packageVerificationCodeValue": verification},
                "licenseConcluded": "Apache-2.0",
                "licenseDeclared": "Apache-2.0",
                "copyrightText": "Copyright SpacemiT and RISCY-SVT contributors",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/xslim@{VERSION}",
                    }
                ],
            }
        ],
        "files": entries,
        "relationships": relationships,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def assert_archive_safe(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
    rejected = []
    for name in names:
        relative = Path(name)
        if relative.suffix.lower() in EXCLUDED_SUFFIXES or "stages" in relative.parts:
            rejected.append(name)
        if any(part in {".env", ".git-credentials"} for part in relative.parts):
            rejected.append(name)
    if rejected:
        raise RuntimeError(f"release archive contains forbidden payloads: {sorted(set(rejected))}")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (source / "VERSION_NUMBER").read_text(encoding="utf-8").strip() != VERSION:
        raise RuntimeError("VERSION_NUMBER does not match the release builder")
    files = source_files(source)
    if not files:
        raise RuntimeError("release source is empty")

    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": str(args.source_date_epoch),
            "TZ": "UTC",
        }
    )
    with tempfile.TemporaryDirectory(prefix="xslim-release-build-") as temporary:
        build_output = Path(temporary) / "dist"
        subprocess.run(
            [
                args.python,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(build_output),
                str(source),
            ],
            check=True,
            env=environment,
        )
        expected = {
            f"xslim-{VERSION}-py3-none-any.whl",
            f"xslim-{VERSION}.tar.gz",
        }
        produced = {path.name for path in build_output.iterdir() if path.is_file()}
        if produced != expected:
            raise RuntimeError(f"unexpected build outputs: {sorted(produced)}")
        for name in sorted(expected):
            shutil.copyfile(build_output / name, output / name)
            assert_archive_safe(output / name)

    source_name = f"xslim-{VERSION}-source.tar.gz"
    docs_name = f"xslim-{VERSION}-docs.tar.gz"
    sbom_name = f"xslim-{VERSION}.spdx.json"
    write_tar_gz(
        output / source_name,
        source,
        files,
        f"xslim-{VERSION}",
        args.source_date_epoch,
    )
    assert_archive_safe(output / source_name)
    write_tar_gz(
        output / docs_name,
        source,
        documentation_files(files),
        f"xslim-{VERSION}-docs",
        args.source_date_epoch,
    )
    assert_archive_safe(output / docs_name)
    write_json(
        output / sbom_name,
        spdx_document(source, files, args.source_commit, args.source_date_epoch),
    )

    primary_names = sorted(
        [
            f"xslim-{VERSION}-py3-none-any.whl",
            f"xslim-{VERSION}.tar.gz",
            source_name,
            docs_name,
            sbom_name,
        ]
    )
    manifest = {
        "schema": "xslim-riscy-release-manifest-v1",
        "version": VERSION,
        "tag": TAG,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "source_date_epoch": args.source_date_epoch,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "build_tools": {
            "python": sys.version.split()[0],
            "build": package_version("build"),
            "setuptools": package_version("setuptools"),
            "wheel": package_version("wheel"),
        },
        "assets": [
            {"name": name, "bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
            for name in primary_names
        ],
    }
    write_json(output / "release-manifest.json", manifest)
    checksum_names = primary_names + ["release-manifest.json"]
    for algorithm, filename in (("sha256", "SHA256SUMS"), ("sha512", "SHA512SUMS")):
        lines = [f"{digest(output / name, algorithm)}  {name}" for name in sorted(checksum_names)]
        (output / filename).write_text("\n".join(lines) + "\n", encoding="ascii")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--source-tree", required=True)
    result.add_argument("--source-date-epoch", type=int, required=True)
    result.add_argument("--python", default=sys.executable)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
