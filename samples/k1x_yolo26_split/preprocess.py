"""Project-exact 640x640 letterbox preprocessing for the sanitized example."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


def _letterbox(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    height, width = image.shape[:2]
    ratio = min(640.0 / width, 640.0 / height)
    resized_width = int(np.rint(width * ratio))
    resized_height = int(np.rint(height * ratio))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    horizontal = 640 - resized_width
    vertical = 640 - resized_height
    left = int(round(horizontal / 2.0 - 0.1))
    top = int(round(vertical / 2.0 - 0.1))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def preprocess_impl(
    path_list: Sequence[str], input_parameter: dict
) -> torch.Tensor:
    """XSlim calibration callback; input_parameter is retained by the API."""
    del input_parameter
    return torch.from_numpy(np.stack([_letterbox(path) for path in path_list]))


def preprocess_one(path: Path) -> np.ndarray:
    """Single-image callback for the optional semantic/audit CLIs."""
    return np.expand_dims(_letterbox(str(path)), axis=0)
