# Human Decision Options

## Required to resume publication

Provide a GitHub API credential for `Custler` that can create releases for a
tag whose commit changes `.github/workflows`. The diagnostic endpoint reports
the required alternative as Contents write plus Workflows write.

Then authorize a new bounded publication-resume stage to:

1. Reuse the already published immutable tag; do not recreate it.
2. Create the GitHub draft release and attach the prepared bytes.
3. Create the matching GitLab release from the same bytes.
4. Publish the GitHub draft, poll workflow runs, prove PyPI absence, and
   download-verify both release surfaces.

## Metadata debt

A repository administrator may separately enable Issues and set the prepared
description/homepage. This is not a release-safety prerequisite.

Stage65B remains unauthorized.
