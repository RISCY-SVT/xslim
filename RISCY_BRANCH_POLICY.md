# RISCY-SVT XSlim Branch Policy

- `main` is an exact, read-only mirror of upstream `spacemit-com/xslim`.
- `riscy/k1x-yolo26` is the only downstream source-development branch.
- Do not create one branch per validation stage; use sequential, reviewable commits.
- Releases use immutable annotated version tags.
- Stage evidence belongs in result packets and shared logs; use occasional
  immutable evidence tags only when long-term Git reachability is required.
- Never force-push or rewrite a published tag.
- Treat upstream and vendor branches as read-only comparison inputs.
