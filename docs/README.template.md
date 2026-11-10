<!-- Generated from docs/README.template.md by `commitbench readme`. Edit the template, not this file. -->
# commitbench

**Which files does a coding agent actually need?** A code-localization retriever trained *only* on free supervision from git history (commit message → files changed), evaluated on all {{n_test}} SWE-bench Lite issues against published baselines. It runs on a laptop CPU with no API calls.

[![CI](https://github.com/VijayChamakuri/commitbench/actions/workflows/ci.yml/badge.svg)](https://github.com/VijayChamakuri/commitbench/actions/workflows/ci.yml)
[![Leaderboard + explorer](https://img.shields.io/badge/results-live%20explorer-2457c5)](https://vijaychamakuri.github.io/commitbench/)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![mypy](https://img.shields.io/badge/types-mypy-1f5082)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![SWE-bench Lite leaderboard](results/swebench_leaderboard.png)

## Key results

File-level localization on all {{n_test}} SWE-bench Lite issues (Acc@k = gold file in the top k, the same definition as the reported rows). All measured on a laptop CPU with no API calls:

- **Git history alone is useful supervision.** The commit-supervised reranker, trained on {{n_train}} commit messages from before each repository's first issue and never on a SWE-bench label, reaches **Acc@1 {{rr_acc1}}%** and Acc@5 {{rr_acc5}}%. That is **{{d_rr_bm25_acc1}} at Acc@1** over BM25, and {{d_rr_bm25_acc5}} at Acc@5.
- **It transfers to unseen codebases.** With each repository's own history held out (leave-one-repo-out), it still reaches Acc@1 {{loro_acc1}}% and Acc@5 {{loro_acc5}}%.
- **The baseline is reproduced, not just cited.** Our BM25 gets Acc@1 {{bm25_acc1}}% against {{pub_bm25_acc1}}% reported, and Acc@5 {{bm25_acc5}}% against {{pub_bm25_acc5}}%.
- **Fused with a local code-embedding reranker**, the best measured method ({{best_label}}) reaches **Acc@5 {{best_acc5}}%** (95% CI {{best_acc5_ci}}). For context, reported full-corpus dense retrievers reach {{pub_jina_acc5}}–{{pub_cre_acc5}}%, and LLM pipelines and agents on paid APIs reach {{pub_agentless_acc5}}–{{pub_locagent_acc5}}%.
- **Where the gain comes from.** Against a hand-tuned hybrid over the same 18 features, the differences are small and not statistically significant: Acc@1 {{d_rr_hyb_acc1}}, Acc@5 {{d_rr_hyb_acc5}}. Most of the lift over BM25 comes from the features themselves (path, symbol, chunk and history signals). Learning their weights from commit history matches careful hand-tuning with no manual effort, and keeps working on repositories it has never seen.
- **Indexing scales with change, not repo size.** {{events}} snapshots replayed through a Merkle-diff incremental index re-read {{reindexed}} files per update on average. Every spot-check matched a full rebuild ({{verified}}).

> Every number above is rendered from [`results/`](results) by `commitbench readme`; CI fails if the README and results disagree. Reported rows are copied from {{published_source}} and marked as such.

## Why this matters

- **Context is the ceiling on agent quality.** If the right file never reaches the prompt, no model fixes the diff. Localization is the step to measure first.
- **Labels are the bottleneck.** Hand-labelled localization data is scarce. Every merged commit already says "this request → these files", so this repo tests whether that free signal is enough.
- **Agents meet codebases that change every commit.** Retrieval has to stay fresh cheaply, and has to be checked so the incremental index never silently drifts from the truth.

## Quickstart

```bash
git clone https://github.com/VijayChamakuri/commitbench && cd commitbench
uv sync
uv run commitbench swebench --no-dense   # 300 issues, sparse + learned methods, ~45 min on a laptop CPU
```

### Reproduce everything

```bash
make reproduce   # SWE-bench Lite (incl. local dense pass) + commit-history benchmark + site + README
```

| Step | Work | Time (Apple M2, 8 GB, CPU only) |
|---|---|---|
| Replay | {{events}} snapshots over 12 blob-less partial clones (sparse `*.py` checkout) | ~{{replay_min}} min |
| Dense | {{dense_files}} file versions embedded with `jina-embeddings-v2-base-code` (fastembed, ONNX) | ~{{dense_min}} min |
| Evaluate | rerankers (pooled + 12 leave-one-repo-out), bootstrap CIs, charts | ~2 min |

There are no API keys or paid services. Data: SWE-bench Lite via the Hugging Face datasets server; repositories from GitHub.

## Leaderboard

{{leaderboard}}

<sub>Measured rows show the mean and a 95% bootstrap CI over {{n_test}} issues. Reported rows come from {{published_source}} and were not re-run. LLM rows use paid APIs; nothing here does.</sub>

![Acc@k](results/swebench_acc_at_k.png)

## How it works

```mermaid
flowchart LR
  subgraph Data
    H[(git history<br/>12 repos, partial clones)] --> T[training queries<br/>commit message → modified files<br/>strictly before first issue]
    S[(SWE-bench Lite<br/>300 issues)] --> Q[test queries<br/>issue text → gold file]
  end
  subgraph Index["Incremental index per repo"]
    T & Q --> R[replay in time order]
    R --> MK[Merkle diff over git blob ids<br/>re-read only changed files]
    MK --> IX[BM25 file + chunk · symbols · paths]
    R --> HP[history priors<br/>churn · recency · co-change]
  end
  IX & HP --> F[18 features per candidate file]
  F --> GBT[commit-supervised reranker<br/>gradient-boosted trees]
  F --> P[top-30 BM25 ∪ top-30 hybrid pool] --> D[Jina-Code-v2 skeleton embeddings<br/>local ONNX]
  GBT & D --> RRF[reciprocal rank fusion] --> E[Acc@k · MRR · CIs · paired tests]
```

| Module | Responsibility |
|---|---|
| [`swebench.py`](src/commitbench/swebench.py) | fetch + checksum SWE-bench Lite, parse gold files from patches, Acc@k |
| [`swe_replay.py`](src/commitbench/swe_replay.py) | per-repo time-ordered replay, sparse checkout, history priors, index verification |
| [`index.py`](src/commitbench/index.py) · [`merkle.py`](src/commitbench/merkle.py) | mutable BM25 (file + chunk), Merkle diff, incremental update |
| [`features.py`](src/commitbench/features.py) | 18 named features and the non-learned retrievers |
| [`dense.py`](src/commitbench/dense.py) | local code embeddings over a candidate pool, cached per blob |
| [`swe_eval.py`](src/commitbench/swe_eval.py) | reranker training (pooled, leave-one-repo-out), label-leak guard, fusion, CIs, miss analysis |

## Results in depth

### Does git-history supervision transfer?

- **Commit-supervised reranker vs BM25:** Acc@1 {{d_rr_bm25_acc1}}; Acc@5 {{d_rr_bm25_acc5}}; leave-one-repo-out vs BM25 at Acc@5: {{d_loro_bm25_acc5}}.
- **vs the hand-tuned hybrid** (same features, weights set by hand): Acc@1 {{d_rr_hyb_acc1}}; Acc@5 {{d_rr_hyb_acc5}}.
- **Leave-one-repo-out** (never saw the scored repo's history): Acc@5 {{loro_acc5}}% vs {{rr_acc5}}% pooled.
- **Most important features** (permutation importance): {{top_features}}.

### Dense retrieval and fusion

The local Jina-Code-v2 model reranks a candidate pool: the union of the top 30 files by BM25 and the top 30 by the hand-tuned hybrid, both non-learned. The pool contains the gold file for **{{pool_recall}}%** of issues, which caps any method that only reorders it.

- **Jina-Code-v2 alone:** Acc@5 {{jina_acc5}}% (reported full-corpus number: {{pub_jina_acc5}}%).
- **Fused with the commit-supervised reranker (RRF):** Acc@5 {{fused_acc5}}%, which is {{d_fused_rr_acc5}} over the reranker alone and {{d_fused_dense_acc5}} over dense alone.

### Per repository (Acc@5)

{{per_repo_table}}

![Per repository](results/swebench_per_repo.png)

### Where it still fails

{{miss_table}}

Browse all {{n_test}} issues, with our top 10 files and the gold file highlighted, in the [live explorer](https://vijaychamakuri.github.io/commitbench/).

### Reproducing the baselines

Our BM25 reaches Acc@1 {{bm25_acc1}}% and Acc@5 {{bm25_acc5}}%, against {{pub_bm25_acc1}}% and {{pub_bm25_acc5}}% reported. Implementations differ in tokenisation (ours splits camelCase and snake_case and indexes path tokens), in document unit (whole file vs chunks) and in issue preprocessing. The gap is reported, not tuned away. The dense row differs further by design: it reranks a pool with file skeletons rather than retrieving chunks from the full corpus, because full-corpus chunk embedding of 45k file versions is days of laptop CPU (see [ADR-5](docs/DESIGN_DECISIONS.md)).

### Efficiency

| | |
|---|---|
| Snapshots replayed | {{events}} |
| Largest first full index build | {{first_build_max}} s (django: {{django_first}} s for ~{{django_files}} files) |
| Incremental update | {{incr_ms}} ms per snapshot on average, re-reading {{reindexed}} files |
| Incremental == full rebuild | {{verified}} spot-checks |
| Feature computation | {{feat_ms}} ms per query (all candidate files) |

### Secondary benchmark: commit history (flask, requests, httpx)

{{history_table}}

<sub>{{history_n}} test commits (the latest 30% of each repo's history). "Masked" removes words that literally name a gold file. Full report: [docs/HISTORY_BENCHMARK.md](docs/HISTORY_BENCHMARK.md).</sub>

## Limitations

- **Retrieval is not agent success.** No patches are generated or tested here.
- **Twelve Python repositories.** SWE-bench Lite also has single-file gold, so multi-file localization is untested.
- **Training queries are commit messages, not issues.** They're written after the change and are terser. Leave-one-repo-out quantifies the transfer cost.
- **Dense is pool-limited.** It can't recover a gold file the sparse pool missed.

## Roadmap

1. Hard-negative fine-tuning of a small code-embedding model on the same commit-history labels.
2. Downstream agent success: retrieve top-k at the base commit, let an agent edit, run the SWE-bench tests.
3. SWE-bench Verified and multi-file instances; non-Python repositories.
4. Query-time latency budget: early-exit reranking for interactive use.

## Project structure

```
src/commitbench/   swebench, swe_replay, swe_eval, dense, index, merkle, features, history, mine, site, readme
results/           every number in this README (JSON, per-instance JSONL, charts)
docs/              METHODOLOGY · DATA_CARD · DESIGN_DECISIONS · HISTORY_BENCHMARK · README template
site/              GitHub Pages: leaderboard + per-instance explorer
tests/             offline tests on a synthetic git repo, metric/parsing/leak guards, README drift
```

## Citation and licence

Code is MIT-licensed ([LICENSE](LICENSE)); cite via [CITATION.cff](CITATION.cff). SWE-bench is MIT-licensed; each evaluated repository keeps its own licence. Reported baseline numbers: {{published_source}}.
