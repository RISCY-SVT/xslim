"""Keep the documented synthetic example bound to the actual public API."""

from pathlib import Path
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "samples/reconstruction_minimal.py"


def test_documented_example_executes_the_public_api():
    example = runpy.run_path(str(EXAMPLE))
    first = example["run_example"]()
    second = example["run_example"]()
    assert first.manifest() == second.manifest()
    np.testing.assert_array_equal(first.hardened_weights["weight"], second.hardened_weights["weight"])
    assert first.final_validation_loss <= first.initial_validation_loss


def test_guide_uses_the_tested_source_verbatim():
    guide = (ROOT / "docs/RECONSTRUCTION_GUIDE.md").read_text(encoding="utf-8")
    displayed = guide.split("<!-- reconstruction-minimal:start -->\n```python\n", 1)[1].split(
        "```\n<!-- reconstruction-minimal:end -->", 1,
    )[0]
    assert displayed == EXAMPLE.read_text(encoding="utf-8")
