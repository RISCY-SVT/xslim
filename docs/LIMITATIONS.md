# Limitations

Current maintenance is local-only and certifies CPython 3.12.3 on Linux
x86_64 with the accepted dependency lock. See [INSTALL.md](../INSTALL.md)
and [the documentation erratum](MAINTENANCE_ERRATA.md). No new model-generation
or board campaign was run for these documentation/metadata corrections.

## Product and Runtime Claims

XSlim is an offline model transformation tool. It does not prove target
provider placement, kernel selection, latency, stability, thermal behavior, or
application fitness. These require exact target-runtime and hardware evidence.

## K1X SpaceMIT Profile

The structural profile covers a specific signed-S8 Q/DQ split contract. It is
not a general certification for every K1X model or SpaceMIT runtime release.
The float CPU tail is outside the quantized provider partition.

## YOLO26 Direct E2E

YoloDecode source support is present. The frozen validated B2/C2 YOLO26 split
graphs do not use it. A historical direct-E2E YOLO26 experiment produced score
collapse on 100/100 images and remains unreconciled. Use six outputs plus the
exact float tail.

## Reconstruction

The release contains BRECQ-inspired layer-local adaptive rounding
infrastructure. Validation covers seven layer-local single-Conv targets, not
full BRECQ blocks. Residual/C2f, whole-head, QDrop, task-loss reconstruction,
and QAT are not validated.

## Accuracy and Operating Points

Aggregate mAP does not guarantee class-, size-, or threshold-level recall.
Calibration and candidate-selection data are not final-generalization
authority. Each application must select score thresholds and quantify TP/FP/FN
under its own costs and data.

## Custom Settings

Selectors depend on exact post-fusion tensor correspondence. Model export,
opset, fusion, or naming changes can invalidate a selector. Strict matching is
recommended.

## Dependency and Typing Debt

The inherited PPQ/ONNXSlim-integrated tree has legacy static-typing debt and
third-party modules without type markers. Downstream public range,
reconstruction, profile, and validator modules are checked separately under a
strict focused contract.

## Licensing and Support

The release is Apache-2.0 with inherited and downstream notices. It is an
unofficial RISCY-SVT derivative, not endorsed by SpacemiT, and carries no
production SLA.
