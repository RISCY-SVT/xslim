# Upstream Provenance

This repository is an unofficial derivative of XSlim.

| Field | Value |
|---|---|
| Upstream project | `spacemit-com/xslim` |
| Upstream URL | <https://github.com/spacemit-com/xslim> |
| Release base commit | `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` |
| Release base tree | `05d2c8425ab8587abf401fa5976a08d008fdd719` |
| Upstream version at base | `2.1.2` |
| Latest public upstream release at reconciliation | `2.1.1` |
| License | Apache-2.0 |

Upstream `main` still matched the release base when `2.1.2+riscy.2` was
prepared. No semantic upstream source change was merged or cherry-picked in
the closure release.

The upstream YoloDecode and YOLO26 ReduceMax branches were already merged into
upstream history before the selected base. Their branch refs remain visible,
but they do not represent a newer unmerged release dependency.

The two-input ReduceMax parsing repair was already present in the upstream
base. RISCY-SVT does not claim authorship. Downstream modifications are listed
in [MODIFICATIONS.md](MODIFICATIONS.md) and [CHANGELOG.md](CHANGELOG.md).

This build is not endorsed by SpacemiT. SpacemiT and XSlim names identify the
origin and compatibility surface only.
