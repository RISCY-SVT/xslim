"""Shared helpers for the optional RISCY-SVT validation commands."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def read_paths(path: Path) -> List[Path]:
    entries = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise ValueError(f"image list is empty: {path}")
    missing = [str(item) for item in entries if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "image-list entries do not exist: " + ", ".join(missing[:8])
        )
    return entries


def _load_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"preprocess module does not exist: {path}")
    module_name = "xslim_user_preprocess_" + hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load preprocess module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_preprocess(specification: str) -> Callable[[Path], Any]:
    module_text, separator, function_name = specification.rpartition(":")
    if not separator or not module_text or not function_name:
        raise ValueError("preprocess must use script.py:function syntax")
    module = _load_module(Path(module_text))
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(
            f"preprocess attribute is not callable: {specification}"
        )
    return function


def make_feed(
    preprocess: Callable[[Path], Any],
    path: Path,
    input_names: Iterable[str],
    selected_input: str = "",
) -> Dict[str, np.ndarray]:
    value = preprocess(path)
    names = list(input_names)
    if isinstance(value, dict):
        feed = {str(name): np.asarray(item) for name, item in value.items()}
        missing = [name for name in names if name not in feed]
        if missing:
            raise ValueError(
                "preprocess mapping is missing model inputs: "
                + ", ".join(missing)
            )
        return feed
    input_name = selected_input or (names[0] if len(names) == 1 else "")
    if not input_name:
        raise ValueError(
            "a single-array preprocess requires a one-input model or --input-name"
        )
    if input_name not in names:
        raise ValueError(f"model has no input named {input_name!r}")
    return {input_name: np.asarray(value)}


def parse_shape(text: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise ValueError(f"invalid shape {text!r}") from exc
    if not result or any(item < 0 for item in result):
        raise ValueError(f"invalid shape {text!r}")
    return result
