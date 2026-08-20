# DEV-001B Reproducible Package Report

Version: `2.1.2+riscy.2.dev2`

## Wheel

Two clean builds are byte-identical:

```text
filename: xslim-2.1.2+riscy.2.dev2-py3-none-any.whl
size: 365899
SHA-256: 7b2ca5075b90643a89da8c0529b20faf0b87c8cace8a2fddc6a4b01ae421c6d8
```

## Source distribution

The two raw gzip archives differ only in gzip/build metadata and have sizes
391938 and 391958 bytes. Their extracted content trees are exact, and their
normalized archives are byte-identical:

```text
normalized archive SHA-256:
a5bf928823cf459486b611e8d8abea00586fa1903b8d1dd9337f73c6e8d5a9f1
```

Raw SHA-256 values are preserved in stage evidence:

```text
run1: 9228f9f9ac778323246f7af4b196c9fb555fa1dc60aa34a5d356b6b6066edf7b
run2: 7b2082dfc9db44f89de4f736c6ad7f902417fb3b1ad2c7da276d9270eb4b9a13
```

Fresh wheel and sdist environments report the intended metadata version,
install the reconstruction API, pass `pip check`, API/config smoke,
`compileall`, and CLI help smokes. No package was published.
