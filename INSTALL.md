# Install XSlim

The local maintenance package is `2.1.2+riscy.2.1`. No maintenance tag or
remote release exists. Obtain the wheel/sdist and checksums from the local
operator handoff. The published `v2.1.2-riscy.2` assets remain immutable;
their original Python 3.9 claim is corrected by this maintenance.

## Certified Environment

The executed reference is **CPython 3.12.3, Ubuntu 24.04, Linux x86_64, CPU**.
[requirements-certified-python312.txt](requirements-certified-python312.txt)
pins its complete runtime dependency closure, including NumPy 2.5.2,
ONNX 1.21.0, ONNX Runtime 1.24.3 and Torch 2.13.0+cpu.

Package metadata permits `>=3.12.3,<3.13`. Only 3.12.3 on the platform above
has been executed in this maintenance. Later 3.12 patches are not separately
certified. Python 3.9 is incompatible with required ONNX; Python 3.10/3.11
cannot install the certified NumPy version. Python 3.13 and later, other
platforms, and CUDA are outside this bounded certification.

## Install the Local Wheel

Keep the constraints file beside the wheel and use a dedicated environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -c "import sys; assert sys.version_info[:3] == (3, 12, 3)"
python -m pip install --constraint requirements-certified-python312.txt --extra-index-url https://download.pytorch.org/whl/cpu ./xslim-2.1.2+riscy.2.1-py3-none-any.whl
xslim --version
python -m pip check
```

Expected version: `xslim 2.1.2+riscy.2.1`; `pip check` must report no
broken requirements. Use the exact CPU wheel source when those Torch versions
are unavailable from the default index. An offline installation may use a
verified wheelhouse containing the same pins. Do not silently substitute
another numeric dependency version.

If a different XSlim appears, inspect `python -m pip show xslim` and activate
the intended environment. This downstream fork is not published to PyPI;
`pip install xslim` is not the installation route.

## Install the Local Source Distribution

In a separate Python 3.12.3 environment:

```bash
python3.12 -m venv .venv-sdist
. .venv-sdist/bin/activate
python -m pip install --constraint requirements-certified-python312.txt --extra-index-url https://download.pytorch.org/whl/cpu ./xslim-2.1.2+riscy.2.1.tar.gz
xslim --version
python -m pip check
```

Build tools used for the certified local build are setuptools 83.0.0 and
wheel 0.46.3. The operator receipt also records the Python, pip and build
frontend identities.

## Source Checkout and API Smoke

Use the exact maintenance commit from the local commit receipt. A clone of a
remote still at riscy.2 does not contain these corrections. After checking
the source commit and archive checksums:

```bash
python -m pip install --constraint requirements-certified-python312.txt .
python samples/reconstruction_minimal.py
python -m xslim --help
```

The synthetic example prints finite loss diagnostics and completes within
eight iterations. It requires no model or dataset and writes no ONNX artifact.
The same file is included in the sdist/source archive; it is not a wheel
console entry point. A wheel installation can execute the file from its
matching source archive.

## Verification Scope

The maintenance tests use an isolated copy of the accepted dependency
environment and fresh wheel/sdist installations into separate environments.
Dependency metadata/resolver rejection checks for unsupported Python versions
do not count as executions of those interpreters. See the operator
`python_support_matrix.tsv` for the exact executed and inspected scopes.

Generic PTQ recipes require separately supplied models and calibration data.
They were not run as a maintenance campaign and do not reproduce frozen
B2/C2. Follow [QUICKSTART.md](QUICKSTART.md) and the
[reproduction distinction](docs/K1X_YOLO26_COOKBOOK.md).

For package conflicts, create a fresh environment with the certified
constraints. For other failures, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
