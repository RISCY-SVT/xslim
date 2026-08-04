# Third-party notices

XSlim includes an embedded derivative of OpenPPL PPQ under
`src/xslim/ppq_decorator/ppq/`. OpenPPL PPQ is licensed under Apache-2.0:
<https://github.com/OpenPPL/ppq>.

The upstream `LICENSE` file contains the retained OpenPPL 2021 and SpacemiT
2023 notices and is distributed without modification.

Python packages named in `requirements.txt` are external runtime/build
dependencies. They are not copied into the XSlim wheel, sdist, or source
archive. Their versions and license declarations from the captured Stage65A
environment are listed in `THIRD_PARTY_LICENSES.tsv`; recipients must obtain
and comply with those packages separately.

No ONNX model, trained weight, calibration image, COCO image, SpacemiT ONNX
Runtime binary, or other vendor binary is included in this release.
