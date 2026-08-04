# Package content audit

The final wheel, normalized sdist, and source archive passed these checks:

- relative paths only; no `..` traversal;
- no symlinks, hard links, devices, sockets, or setuid/setgid entries;
- no `.git` metadata;
- no ONNX model, checkpoint, trained weight, vendor runtime, or calibration
  image;
- no private absolute path, credential, token, or private-key material;
- upstream `LICENSE`, `LICENSE_AUDIT.md`, `MODIFICATIONS.md`,
  `NOTICE-RISCY-SVT`, `THIRD_PARTY_NOTICES.md`,
  `THIRD_PARTY_LICENSES.tsv`, and `UPSTREAM.md` are present;
- wheel contains 125 files, sdist 185 files, source archive 197 files;
- final wheel and deterministic sdist install in independent fresh virtual
  environments, all three CLI help paths run, and `pip check` passes.

The upstream small BERT `.npy` sample inputs remain in source/sdist as inherited
Apache-licensed project content. They are neither models nor trained weights.
