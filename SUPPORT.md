# Support

Use the RISCY-SVT issue tracker for reproducible defects in this downstream
release. Use the upstream SpacemiT project for behavior that reproduces on the
unmodified upstream base.

Before filing an issue, provide:

```text
xslim version
Python and dependency versions
host OS and architecture
command and sanitized config
input/output names, shapes and dtypes
small public or synthetic reproducer
full error text
```

Do not attach private models, datasets, credentials, vendor runtimes, or board
dumps. Replace private paths and tensor names when they are not essential.

Structural K1X profile validation does not guarantee provider placement or
runtime performance. Provider crashes, partitioning, and board behavior also
require the exact runtime build and board identity.

No production SLA, model-accuracy guarantee, or hardware support contract is
provided.
