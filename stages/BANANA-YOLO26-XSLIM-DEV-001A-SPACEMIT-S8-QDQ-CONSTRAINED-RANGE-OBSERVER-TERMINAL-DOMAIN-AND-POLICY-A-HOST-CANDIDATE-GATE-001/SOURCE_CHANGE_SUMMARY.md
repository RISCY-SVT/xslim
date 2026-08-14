# XSLIM-DEV-001A source summary

This unreleased downstream development line adds a generic strict local selector, constrained signed asymmetric INT8 range search, post-refine qparam locking/finalization, and a structural `spacemit_k1x_s8_qdq_split_v1` validator.

The implementation does not embed private model tensor names. Explicit tensor or bounded subgraph selectors are supplied by the consumer configuration. Contradictory overlaps, strict no-match, unsafe rebinding, infeasible constraints, and exported-profile violations fail closed.

The profile validates graph structure only. It does not claim provider placement, fusion, kernel selection, latency, or board stability.

Version `2.1.2+riscy.2.dev1` is a development artifact. No tag, release, or PyPI publication is part of this stage.

Host qualification selected the terminal-domain-only A1 policy. The generated all-S8
candidate improved H500 by `0.007062946` mAP and full val2017 by `0.007075925`
mAP over the frozen B2 control, with positive paired-bootstrap confidence intervals.
The detailed task evidence remains in the Banana research branch and result packet;
this source repository records only the compact identity summary below.
