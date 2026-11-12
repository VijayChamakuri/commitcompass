# Methodology

How every number in the README is produced. The numbers themselves live in [`results/`](../results) and are rendered into the README by `commitcompass readme`.

## 1. Task

Given an issue (natural-language bug report or feature request) and the repository at the commit the fix was based on, rank the repository's Python files so the file the fix edits is near the top. This is *file-level localization*, the first step every coding agent takes before it can edit anything.

## 2. Benchmarks

### 2.1 SWE-bench Lite (primary)

- **Instances:** all 300 instances of `princeton-nlp/SWE-bench_Lite` (test split), downloaded from the Hugging Face datasets-server API. The file's SHA-256 is recorded in `data/swe_cache/dataset.json`.
- **Query:** `problem_statement`, the issue text as filed.
- **Gold:** the `.py` files modified by the reference `patch`. Every Lite instance edits exactly one existing file, so no gold file is excluded.
- **Candidates:** every `.py` file in the repository at `base_commit`.
- **Metric:**
  - **Acc@k = 1 if every gold file ranks within the top k.** This matches the file-level metric in LocAgent (arXiv:2503.09089, Table 4), which is the source of all reported comparison rows.
  - Also reported: MRR, and recall under an 8k-token context budget (each file contributes its best BM25 chunks until the budget is spent).
- **Statistics:** 95% bootstrap CIs over instances (2,000 resamples). Paired bootstrap CIs for differences between methods.

### 2.2 Commit history (secondary)

Commit message → modified `.py` files for flask, requests and httpx (non-merge commits before 2026-10-01, filtered for bots, version bumps, formatting-only changes and short messages). Temporal 70/30 split per repository. A *masked* variant removes words that literally name a gold file. Full report: [HISTORY_BENCHMARK.md](HISTORY_BENCHMARK.md).

## 3. Index and replay

- **Partial clones.** Each of the 12 SWE-bench repositories is cloned with `--filter=blob:none` and sparse-checked-out to `*.py`, so only Python sources are fetched.
- **One incremental index per repository.** Events are processed in time order on a single index:
  - commit-history training queries, which use the parent snapshot;
  - SWE-bench issues, which use `base_commit`.
- **Merkle-tree diff.** Each snapshot is diffed against the previous one with a Merkle tree over `{path: git blob id}`, and only changed files are re-tokenised. Every 25th SWE-bench event, the incremental index is rebuilt from scratch and compared for exact equality.
- **History priors** (churn, recency, co-change) absorb only commits strictly older than the event being scored.

## 4. Methods

**Sparse and structural** (no learning):
- BM25 over whole files, and BM25 over def/class chunks (max per file);
- path and filename overlap;
- symbol-name match;
- a history-only prior;
- a hand-tuned linear hybrid.

**Dense** (local, free):
- Model: `jinaai/jina-embeddings-v2-base-code` via fastembed (ONNX, CPU).
- **Pool:** dense reranks a candidate pool, the union of the top 30 files by BM25 (file) and the top 30 by the hand-tuned hybrid. Both are non-learned. The pool's recall (share of issues whose gold file is in it) is reported.
- **File representation:** each pooled file is embedded once as a *skeleton*: its path, class/def signatures and the first docstring line under each, truncated to 256 tokens. The score is cosine similarity with the first 2,000 characters of the issue.
- **Why not full-corpus chunk retrieval:** that run was not done. What was measured is the pool. The file count and the seconds are the `dense_files_embedded` and `dense_seconds` fields in `results/swebench_lite.json`, and they are rendered into [RESULTS.md](RESULTS.md).
- Vectors are cached per git blob, so a file version is embedded once.

**Commit-supervised reranker** (the contribution):
- Gradient-boosted trees over 18 named features (sparse scores, path, symbol, history priors, test-file flags, file size).
- The model is scikit-learn's `HistGradientBoostingClassifier`, fit in `fit_gbt`: 300 iterations, learning rate 0.05, at most 31 leaves, L2 penalty 1.0. Each row is one candidate file. Negative rows are left at equal weight.
- **Training labels are commit messages and the files those commits modified**, taken from each repository's history strictly before the creation date of its first SWE-bench issue. `clean_message` strips git trailers and cuts the text at 1,500 characters. `classify` drops bots, version bumps, format-only subjects, messages under 5 words, and commits that touch more than 8 Python files. `check_no_label_leak` asserts the time cut, and no SWE-bench label is ever used for training.
- **Tokenizer.** `tokenize` in `textproc.py` splits camelCase and snake_case, keeps the whole identifier, applies a light plural stem, and drops stopwords. The same function is used on queries and on source.
- **Pooled:** trained on all repositories' commit queries.
- **Leave-one-repo-out:** the model scoring repository R never saw R's commits. This tests transfer to unseen codebases.
- **Which feature mattered.** Permutation importance uses average precision on the test issues, 3 repeats, seed 0. The ranked column names are rendered in [RESULTS.md](RESULTS.md).

**Fusion:** reciprocal rank fusion (k = 60) of the reranker or BM25 with the dense scores. Each list contributes `1/(60 + rank)`.

## 5. Comparability with reported numbers

The reported rows are copied, not re-run. Known differences:

- Their retrieval baselines may chunk, truncate and preprocess issues differently. Our BM25 and Jina-Code rows are our own implementations of the same model families, so gaps against their rows are shown, not hidden.
- Their large-language-model rows call a paid model. Scoring here does not. Fetching the dataset and cloning the repositories does use the network.
- Scoring runs locally. Wall times are in `results/swebench_lite.json`. That file does not name the machine.

## 6. What is not claimed

- Retrieval quality is not agent success. No patch is generated or tested here.
- SWE-bench Lite is 12 Python repositories. Results may not transfer to other languages or monorepos.
- Training commit messages are written after the change, and are not issue reports. Leave-one-repo-out measures how much that mismatch costs on unseen repositories.
