# Documentation and Compatibility Errata

The local `2.1.2+riscy.2.1` maintenance source descends from immutable
`v2.1.2-riscy.2`. Its packages, tag and release have not been published.
The original published source and scientific evidence remain unchanged.

## Reconstruction Example (D1)

The riscy.2 reconstruction example supplied unsupported keyword arguments
and omitted required rounders and callbacks. Use the executable, tested
[current guide](RECONSTRUCTION_GUIDE.md). This corrects documentation only;
the reconstruction implementation is unchanged.

## Python Support (D2)

The old `>=3.9` claim conflicts with required ONNX 1.21.x (`>=3.10`) and
the accepted NumPy 2.5.2 dependency (`>=3.12`). Current maintenance support
is conservatively limited to Python 3.12. The certified installation and
dependency lock are described in [INSTALL.md](../INSTALL.md).

## Snippet Verification (D3)

The DEV-002 report counted 106 parsed/compiled snippets, not 106 executed
workflows. Syntax success does not establish API binding, installation,
quantization accuracy, or board execution. Current audits report these
levels separately and preserve reasons for unexecuted recipes.

Generic examples and the historical Stage64 recipe do not reproduce B2/C2.
See the [cookbook](K1X_YOLO26_COOKBOOK.md) for exact-reproduction inputs.
The frozen model-generation authority remains riscy.2 and its accepted
lineage/manifests; this maintenance is not an accuracy improvement.
