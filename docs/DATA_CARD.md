# Data card

## SWE-bench Lite (evaluation only)

| | |
|---|---|
| Source | `princeton-nlp/SWE-bench_Lite`, test split, via the Hugging Face datasets-server API |
| Licence | MIT (SWE-bench); each repository keeps its own licence |
| Size | 300 issues from 12 Python repositories: django 114, sympy 77, matplotlib 23, scikit-learn 23, pytest 17, sphinx 16, astropy 6, requests 6, pylint 6, xarray 5, seaborn 4, flask 3 |
| Fields used | `instance_id`, `repo`, `base_commit`, `created_at`, `problem_statement`, `patch` |
| Gold | files modified by `patch`; every Lite instance modifies exactly one existing `.py` file |
| Integrity | SHA-256 of the downloaded JSONL in `data/swe_cache/dataset.json` |
| Use | **Evaluation only.** No SWE-bench field is used to train or tune any model. The one hand-set method, the hybrid, was tuned before SWE-bench was added. |

## Commit-history training queries

| | |
|---|---|
| Source | non-merge history of the 12 SWE-bench repositories (blob-less partial clones) |
| Window | per repository, the last 250 qualifying commits strictly before its first SWE-bench issue |
| Query / label | cleaned commit message → existing `.py` files it modified |
| Filters | drop root commits, bots, commits touching more than 8 `.py` files, messages under 5 words, version bumps, typo- and format-only commits ([`mine.py`](../src/commitbench/mine.py)) |
| Known noise | messages are written after the change; some modified files are incidental |

## Commit-history benchmark (secondary)

flask, requests and httpx, all non-merge commits before 2026-10-01, under the same filters. See [HISTORY_BENCHMARK.md](HISTORY_BENCHMARK.md).

## Models

| Model | Use | Licence | Runs |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` | dense retrieval | Apache-2.0 | locally via fastembed (ONNX, CPU) |
| scikit-learn `HistGradientBoostingClassifier` | reranker | BSD-3 | locally |
