# Contributing

## Prepare a Development Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest ruff mypy build twine check-wheel-contents
```

## Change Contract

- Keep generic XSlim code model-independent; do not hard-code private model
  tensor names or paths.
- Add tests before changing quantization, reconstruction, exporter, selector,
  or profile behavior.
- Preserve legacy configuration behavior unless a documented migration is
  unavoidable.
- Keep structural validation separate from provider execution claims.
- Do not commit models, weights, predictions, datasets, credentials, or vendor
  binaries.

## Checks

```bash
python -m pytest
ruff check .
python -m compileall -q src tests samples
python -m build
twine check dist/*
```

Run focused strict mypy on public downstream modules and report legacy
whole-tree debt separately. Run ShellCheck on every shipped shell script.

## Documentation

User docs should state purpose, minimal command, expected output, common
failure/fix, then deeper details. Commands and identifiers stay in English.

## Provenance

Preserve upstream attribution and Apache-2.0 notices. Explain whether a change
is upstream-authored, downstream-authored, or independently reimplemented.
Do not silently copy semantic changes from an unreviewed upstream branch.
