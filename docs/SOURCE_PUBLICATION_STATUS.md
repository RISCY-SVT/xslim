# Maintenance Source Publication

D1-D3 are fixed on the existing `riscy/k1x-yolo26` branch. Source publication
to the established GitHub and GitLab repositories is independent of a binary
release. The MAINT-001-PUB post-push receipt is the authority for exact final
branch heads and trees; this document does not contain its own commit hash.

| Surface | Identity or status |
|---|---|
| Tested maintenance source | `f5007ceb086d91cc06da0b19bfbc2ce90908fdd5` |
| Tested source tree | `9903811c33f447974a2d31ce1a16dbbdfc130353` |
| Version | `2.1.2+riscy.2.1` |
| Later current branch HEAD | May include a docs-only publication-status commit; see independent receipt |
| Candidate assets | Local MAINT-001 builds from the tested commit above; not uploaded |
| Maintenance tag/release/PyPI | Not authorized, not attempted |
| Existing riscy.1/riscy.2 | Immutable |

Use [the pinned source installation](../INSTALL.md#source-checkout-and-api-smoke).
Only CPython 3.12.3 on Ubuntu 24.04 Linux x86_64 CPU was executed. Metadata
permits `>=3.12.3,<3.13`; it does not certify other patches or platforms.
Fresh-install evidence uses separate environments with an offline-preseeded
accepted dependency closure, not a fresh internet dependency-resolution test.

All 117 runtime Python modules and numeric dependency requirements remain
unchanged. The synthetic reconstruction example is not full BRECQ or a YOLO
campaign. Generic/Stage64 instructions do not reproduce frozen B2/C2 by themselves.

Native backend sdists differed; original bytes were retained. The explicit
canonicalization packaging step produced byte-identical final assets under
the recorded source/epoch/toolchain. No new build is part of publication.
Original local-only build manifests and historical no-push receipts stay intact.

Remaining debts are not hidden: inherited whole-tree mypy and per-file REUSE,
unverified external links, partial procedural Codex handoff, four missing
external chat histories and no verified second off-host backup. None of these
is a claim of completed five-source assimilation or full disaster recovery.

B2 remains the universal vendor control. C2 retains its existing separate
TIER-1 higher-AP waiver, historical universal FAIL and threshold-selection
requirement. No new waiver, model, runtime default, board action, numerical
tuning, training or co-design is authorized by source publication.
