# Design decisions

### ADR-1: Train only on git history; never on SWE-bench
**Context:** Localization labels are expensive, and evaluating on the same distribution you train on flatters the model.
**Decision:** The reranker's labels are commit message → modified files, from commits before each repository's first SWE-bench issue. SWE-bench is evaluation-only, and `check_no_label_leak` enforces the time boundary.
**Consequence:** The result measures whether *free* supervision transfers to real issues, which is the question a team with no labelling budget actually has.

### ADR-2: Use the published metric definition exactly
**Decision:** Acc@k counts an instance only when every gold file is in the top k, as in LocAgent's Table 4. Reported rows are copied verbatim with their source, and never mixed with measured rows without a label.

### ADR-3: Reproduce a baseline before claiming to beat it
**Decision:** BM25 and Jina-Code-v2 are run here and placed next to their reported numbers. Gaps are reported with likely causes, and nothing is tuned toward a published figure.

### ADR-4: One incremental index per repository, verified
**Context:** 300 base commits across 12 repositories, plus 3,000 training snapshots.
**Decision:** A Merkle diff over git blob ids, so only changed files are re-read. The incremental index is spot-checked for exact equality against a full rebuild.
**Consequence:** Indexing cost scales with change, not repository size, which is the same reason production code-search systems sync this way.

### ADR-5: Dense retrieval is local and fused late
**Decision:** fastembed ONNX on CPU, with no API. Dense reranks a 60-file candidate pool using file skeletons, because full-corpus chunk embedding is days of laptop CPU. Dense scores are fused with reciprocal rank fusion at query time instead of being fed into the reranker, so training needs no embeddings of historical snapshots.
**Consequence:** The reranker stays cheap to train. Fusion is a fixed, parameter-light combination.

### ADR-6: Simple, inspectable models
**Decision:** Gradient-boosted trees over 18 named features, with permutation importance reported. A reviewer can see *why* a file ranked high.

### ADR-7: No hand-typed numbers
**Decision:** The README is rendered from `docs/README.template.md` and `results/*.json`. CI fails if they diverge.
