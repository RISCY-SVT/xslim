# Install XSlim

XSlim requires Python 3.9 or newer. A dedicated virtual environment prevents
ONNX, ONNX Runtime, Torch, and NumPy versions from colliding with other tools.

## Install a Release Wheel

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./xslim-2.1.2+riscy.2-py3-none-any.whl
xslim --version
python -m pip check
```

Expected version: `2.1.2+riscy.2`; `pip check` must report no broken
requirements.

If a different version appears, run `python -m pip show xslim`, remove that
package, and reinstall the release wheel by explicit path. This fork is not on
PyPI.

## Install the Source Distribution

```bash
python3 -m venv .venv-sdist
. .venv-sdist/bin/activate
python -m pip install ./xslim-2.1.2+riscy.2.tar.gz
xslim --version
```

## Install from a Verified Source Checkout

```bash
git clone https://github.com/RISCY-SVT/xslim.git
cd xslim
git checkout v2.1.2-riscy.2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Verify release checksums before installation:

```bash
sha256sum -c SHA256SUMS
```

The release tag is annotated but is not claimed to be cryptographically signed.

## Development Install

```bash
python3 -m venv .venv-dev
. .venv-dev/bin/activate
python -m pip install -e .
python -m pytest
```

For release-quality checks install Ruff, mypy, build, twine,
check-wheel-contents, and ShellCheck in the development environment. Do not
install host release tools on a K1X board.

## Platform Notes

- Quantization is an offline host workflow; CPU execution is supported.
- CUDA calibration is used only when the installed Torch build and hardware
  expose CUDA.
- SpaceMIT provider execution requires a separately installed vendor runtime
  and board validation. It is not bundled with XSlim.
- Large ONNX models may use external data. Keep model and data files together.

## Common Failures

`ModuleNotFoundError: onnx` means the package dependencies were not installed
in the active environment. Activate the intended venv and reinstall the wheel.

`pip check` conflicts usually mean XSlim was installed into a shared ML
environment. Create a fresh venv instead of forcing dependency replacement.

For other failures, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
