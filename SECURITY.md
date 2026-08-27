# Security Policy

## Supported Version

Security fixes are considered for the latest RISCY-SVT release only. This fork
is an offline research/development tool and is not a supported SpacemiT product.

## Report a Vulnerability

Do not include credentials, private models, datasets, or proprietary logs in a
public issue. Use GitHub's private vulnerability reporting for
`RISCY-SVT/xslim` when available, or contact the repository maintainers through
the private channel listed in the repository profile.

Include the affected version, operating system, Python version, minimal
reproducer, impact, and whether untrusted ONNX/config input is required.

## Security Boundaries

XSlim loads ONNX, JSON, Python preprocess modules, and model data from local
paths. Treat all of them as executable or parser-sensitive input. Run
untrusted artifacts in an isolated account/container without credentials or
write access to valuable data.

Release checksums establish byte identity, not trust. Verify provenance and
the annotated Git tag before installing an asset.
