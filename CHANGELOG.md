# Changelog

## 1.0.0 (2026-11-10)

- **SWE-bench Lite file localization** on all 300 issues, using the published Acc@k definition. Measured rows sit next to the rows reported in the LocAgent paper.
- **Commit-supervised reranker**, trained only on pre-issue git history (pooled and leave-one-repo-out), with a label-leak guard.
- **Local dense retrieval** (Jina-Code-v2 via fastembed, CPU) and reciprocal rank fusion.
- **Incremental indexing** across 12 repositories with Merkle diffs over partial clones, spot-verified against full rebuilds.
- **Packaging:** leaderboard and per-instance explorer site; README rendered from results, with a drift test; methodology, data card and design decisions.
- The commit-history benchmark from v0.1 is kept as a secondary benchmark.
