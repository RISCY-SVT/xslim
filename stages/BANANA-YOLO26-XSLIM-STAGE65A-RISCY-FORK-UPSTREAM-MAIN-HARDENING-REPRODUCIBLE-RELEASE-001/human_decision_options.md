# Human decision required

GitHub rejected both authorized fork attempts with HTTP 403 because the active
personal access token cannot create a fork in `RISCY-SVT`.

Required recovery:

1. Grant the authenticated account permission to create repositories/forks in
   `RISCY-SVT`, and refresh the GitHub CLI token accordingly.
2. Run:

   ```bash
   gh repo fork spacemit-com/xslim --org RISCY-SVT --clone=false --remote=false
   ```

3. Resume publication from the validated local branch and local annotated tag;
   push normally, create the GitHub release, attach the seven manifested files,
   and verify anonymous download hashes.

Do not create an unrelated standalone repository, force-push, publish to PyPI,
or claim that the direct-E2E YOLO26 path is fixed. No Stage65B work is implied
or authorized.
