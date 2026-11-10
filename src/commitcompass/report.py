"""Turn results/summary.json + per_query.jsonl into tables, charts, REPORT.md and README numbers."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from . import chartstyle as cs
from .textproc import tokenize

LABELS = {
    "bm25_file": "BM25 (file)",
    "bm25_chunk": "BM25 (chunk-max)",
    "path_match": "Path / filename match",
    "symbol_match": "Symbol-name match",
    "history_prior": "History prior only",
    "hybrid_hand_tuned": "Hybrid (hand-tuned)",
    "learned_lr": "Learned reranker (logistic)",
    "learned_gbt": "Learned reranker (GBT)",
}
MAIN = ("bm25_file", "hybrid_hand_tuned", "learned_gbt")


def _fmt(d: dict, ci: bool = True) -> str:
    if not ci:
        return f"{d['mean']:.3f}"
    lo, hi = d["ci95"]
    return f"{d['mean']:.3f} <sub>[{lo:.3f}, {hi:.3f}]</sub>"


def pooled_table(vs: dict) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "budget_recall@8k"]
    head = "| Method | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = []
    best = {c: max(vs["pooled"][m][c]["mean"] for m in LABELS) for c in cols}
    for m, label in LABELS.items():
        cells = []
        for c in cols:
            cell = _fmt(vs["pooled"][m][c], ci=c in ("recall@10", "mrr"))
            if abs(vs["pooled"][m][c]["mean"] - best[c]) < 1e-12:
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def compact_table(summary: dict) -> str:
    """Small README table: recall@10 and MRR, as-is vs masked, for the main methods."""
    a, m = summary["variants"]["as_is"]["pooled"], summary["variants"]["masked"]["pooled"]
    out = (
        "| Method | Recall@10 | MRR | Recall @ 8k-token budget | Recall@10 (masked) | MRR (masked) |\n"
        "|---|---|---|---|---|---|\n"
    )
    for k in LABELS:
        out += (
            f"| {LABELS[k]} | {a[k]['recall@10']['mean']:.3f} | {a[k]['mrr']['mean']:.3f} | "
            f"{a[k]['budget_recall@8k']['mean']:.3f} | {m[k]['recall@10']['mean']:.3f} | "
            f"{m[k]['mrr']['mean']:.3f} |\n"
        )
    return out


def per_repo_table(vs: dict) -> str:
    repos = [r for r in vs if r not in ("pooled", "n", "deltas", "permutation_importance")]
    head = "| Method | " + " | ".join(f"{r} R@10 / MRR (n={vs['n'][r]})" for r in repos) + " |\n"
    head += "|---|" + "---|" * len(repos) + "\n"
    rows = []
    for m, label in LABELS.items():
        cells = [f"{vs[r][m]['recall@10']['mean']:.3f} / {vs[r][m]['mrr']['mean']:.3f}" for r in repos]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def deltas_table(vs: dict) -> str:
    out = "| Comparison | Mean Δ | 95% CI (paired bootstrap) |\n|---|---|---|\n"
    for k, d in vs["deltas"].items():
        name, metric = k.split(":")
        base = name.replace("learned_gbt_minus_", "")
        out += f"| GBT − {LABELS[base]}, {metric} | {d['mean']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |\n"
    return out


def charts(summary: dict, results_dir: Path) -> list[str]:
    cs.apply()
    files = []
    # 1) Recall@10 per method, as-is vs masked, with CIs.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    methods = list(LABELS)
    width = 0.38
    for j, (v, color) in enumerate((("as_is", cs.BLUE), ("masked", cs.ORANGE))):
        p = summary["variants"][v]["pooled"]
        means = [p[m]["recall@10"]["mean"] for m in methods]
        err = [
            [p[m]["recall@10"]["mean"] - p[m]["recall@10"]["ci95"][0] for m in methods],
            [p[m]["recall@10"]["ci95"][1] - p[m]["recall@10"]["mean"] for m in methods],
        ]
        xs = [i + (j - 0.5) * width for i in range(len(methods))]
        ax.bar(
            xs,
            means,
            width - 0.04,
            yerr=err,
            error_kw={"elinewidth": 0.9, "ecolor": cs.INK2, "capsize": 0},
            color=color,
            label="Commit message as written" if v == "as_is" else "File names masked out of the message",
        )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=25, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Recall@10 (test split)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=2, handlelength=1)
    ax.set_title("Which files did the commit edit? Recall@10 with 95% CIs")
    fig.tight_layout()
    fig.savefig(results_dir / "recall_at_10.png")
    plt.close(fig)
    files.append("recall_at_10.png")

    # 2) Recall vs context token budget.
    fig, ax = plt.subplots(figsize=(6.8, 4))
    budgets = [2, 4, 8, 16]
    for m, color in zip(MAIN, (cs.NEUTRAL, cs.ORANGE, cs.BLUE), strict=True):
        for v, ls in (("as_is", "-"), ("masked", "--")):
            p = summary["variants"][v]["pooled"][m]
            ax.plot(
                budgets,
                [p[f"budget_recall@{b}k"]["mean"] for b in budgets],
                ls,
                marker="o",
                color=color,
                label=f"{LABELS[m]} ({'as written' if v == 'as_is' else 'masked'})",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}k" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Context budget (approx. tokens)")
    ax.set_ylabel("Changed files with a chunk in context")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Recall under a fixed context budget (dashed: file names masked)")
    fig.tight_layout()
    fig.savefig(results_dir / "budget_curve.png")
    plt.close(fig)
    files.append("budget_curve.png")
    return files


CATEGORY_TEXT = {
    "cross_cutting": "Cross-cutting change (≥4 gold files)",
    "cold_file": "Gold file rarely touched before (≤2 prior edits)",
    "test_only_gold": "Only test files changed",
    "vague_message": "Vague message (<6 distinct content tokens)",
    "vocabulary_mismatch": "Vocabulary mismatch (none of the above)",
}


def categorize(row: dict) -> str:
    if len(row["gold"]) >= 4:
        return "cross_cutting"
    if all(c <= 2 for c in row["gold_churn"]):
        return "cold_file"
    if all(row["gold_is_test"]):
        return "test_only_gold"
    if len(set(tokenize(row["query"]))) < 6:
        return "vague_message"
    return "vocabulary_mismatch"


def failure_analysis(rows: list[dict], method: str = "learned_gbt") -> tuple[str, dict]:
    rows = [r for r in rows if r["variant"] == "as_is"]
    total = Counter(categorize(r) for r in rows)
    misses = [r for r in rows if r["methods"][method]["recall@10"] == 0]
    miss_c = Counter(categorize(r) for r in misses)
    md = (
        f"A *miss* = no gold file in the {LABELS[method]}'s top 10 (message as-is). "
        f"{len(misses)} of {len(rows)} test queries ({len(misses) / len(rows):.1%}) are misses.\n\n"
        "| Category (first rule that matches) | Misses | Share of misses | Miss rate within category |\n"
        "|---|---|---|---|\n"
    )
    for cat, text in CATEGORY_TEXT.items():
        n, t = miss_c.get(cat, 0), total.get(cat, 0)
        md += f"| {text} | {n} | {n / max(1, len(misses)):.0%} | {n / max(1, t):.0%} (of {t}) |\n"
    md += "\n**Concrete misses (real commits):**\n\n"
    seen: set[str] = set()
    examples = []
    for r in sorted(misses, key=lambda r: (r["repo"], r["order"])):
        cat = categorize(r)
        if cat in seen:
            continue
        seen.add(cat)
        examples.append((cat, r))
    for cat, r in examples[:5]:
        first = r["query"].split("\n", 1)[0][:140]
        md += (
            f"- **{CATEGORY_TEXT[cat]}** — `{r['repo']}@{r['sha'][:10]}`: “{first}”  \n"
            f"  gold: {', '.join(f'`{g}`' for g in r['gold'])} · top-3 predicted: "
            f"{', '.join(f'`{p}`' for p in r['top5'][method][:3])}\n"
        )
    return md, {"misses": len(misses), "total": len(rows), "by_category": dict(miss_c)}


def _timing_md(summary: dict) -> str:
    out = (
        "| Repo | Queries replayed | Mean full rebuild | Mean incremental update | Speed-up | "
        "Files re-indexed / update | Merkle nodes visited | Incremental == full rebuild |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    for repo, t in summary["timing"].items():
        n = t["incremental_updates"] + 1
        visited = t["merkle_nodes_visited"] / max(1, t["merkle_nodes_total"])
        out += (
            f"| {repo} | {n} | {t['full_rebuild_mean_seconds'] * 1000:.0f} ms "
            f"({t['full_rebuild_mean_files']:.0f} files) | {t['incremental_mean_seconds'] * 1000:.1f} ms | "
            f"{t['speedup']:.0f}× | {t['incremental_mean_files_reindexed']:.2f} | {visited:.1%} of tree | "
            f"{t['verified_equal']}/{t['verified_total']} snapshots |\n"
        )
    return out


def _dataset_md(summary: dict) -> str:
    out = (
        "| Repo | Commits scanned | Queries kept | Kept | Mean gold files | New files excluded from gold |\n"
        "|---|---|---|---|---|---|\n"
    )
    for repo, s in summary["mine"].items():
        out += (
            f"| {repo} | {s['commits_scanned']} | {s['queries']} | {s['kept_fraction']:.0%} | "
            f"{s['mean_gold_size']:.2f} | {s['new_py_files_excluded_from_gold']} |\n"
        )
    reasons: Counter[str] = Counter()
    for s in summary["mine"].values():
        reasons.update(s["drop_reasons"])
    reasons.pop("kept", None)
    out += "\nDrop reasons (all repos): " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()) + ".\n"
    return out


def write_report(results_dir: Path, root: Path) -> None:
    summary = json.loads((results_dir / "summary.json").read_text())
    with open(results_dir / "per_query.jsonl") as f:
        rows = [json.loads(line) for line in f]
    asis = [r for r in rows if r["variant"] == "as_is"]
    n_cand = sum(r["n_candidates"] for r in asis) / len(asis)
    rand_r10 = sum(min(1.0, 10 / r["n_candidates"]) for r in asis) / len(asis)
    context = (
        f"Mean candidate files per query: **{n_cand:.0f}** (every `.py` file in the snapshot). "
        f"A random ranker would score Recall@10 ≈ **{rand_r10:.3f}**.\n"
    )
    pngs = charts(summary, results_dir)
    fa_md, fa = failure_analysis(rows)
    summary_extra = {"failure_analysis": fa}
    (results_dir / "failure_analysis.json").write_text(json.dumps(summary_extra, indent=2))

    a, m = summary["variants"]["as_is"], summary["variants"]["masked"]
    tables = "## Message as-is\n\n" + pooled_table(a) + "\n## File names masked\n\n" + pooled_table(m)
    tables += "\n## Per repo (as-is)\n\n" + per_repo_table(a)
    (results_dir / "tables.md").write_text(tables)

    imp = list(a["permutation_importance"].items())[:6]
    imp_m = list(m["permutation_importance"].items())[:6]
    report = f"""# commitcompass report

_Auto-generated by `commitcompass report` from `results/summary.json`. Every number below comes from that run._

**Setup.** Real commits from {", ".join(summary["mine"])} up to {summary["cutoff"]}. Query = commit message,
gold = existing `.py` files the commit modified. Each query is scored against the repo **as it was at the
parent commit**. Temporal split per repo: earliest 70% of queries train the learned rerankers, latest 30% test
(n_train={summary["n_train"]}, n_test={summary["n_test"]}). All numbers are on the **test split**; brackets are
95% bootstrap CIs over queries.

## Dataset

{_dataset_md(summary)}
{context}
## Results — commit message as-is

{pooled_table(a)}
`budget_recall@8k` = share of gold files with at least one chunk packed into an ~8k-token context when chunks
are taken in the method's ranked order.

## Results — file names masked

{summary["leakage_rate_gold_stem_in_message"]:.0%} of test commit messages literally name a gold file
(e.g. “sessions.py: …”). Masking removes those words to measure retrieval without the free hint.

{pooled_table(m)}
## Is the learned reranker really better?

Paired bootstrap over the same test queries (message as-is):

{deltas_table(a)}
Masked:

{deltas_table(m)}
## Per repo (as-is, Recall@10 / MRR)

{per_repo_table(a)}
## What the learned reranker relies on

Permutation importance (drop in average precision when a feature is shuffled), test split.

- As-is: {", ".join(f"`{k}` {v:.3f}" for k, v in imp)}
- Masked: {", ".join(f"`{k}` {v:.3f}" for k, v in imp_m)}

## Incremental indexing (Merkle tree)

The index is moved from one parent snapshot to the next by diffing Merkle trees (leaf = git blob id,
directory = hash of children). Only changed files are re-read and re-tokenized. Sampled snapshots were also
rebuilt from scratch and compared field-by-field with the incremental index.

{_timing_md(summary)}
## Failure analysis (learned GBT, test split, as-is)

{fa_md}
## Charts

{chr(10).join(f"![{p}](results/{p})" for p in pngs)}
"""
    # The full commit-history report lives in docs/; README numbers are rendered by readme.py.
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "HISTORY_BENCHMARK.md").write_text(report.replace("](results/", "](../results/"))
    print(f"[report] wrote docs/HISTORY_BENCHMARK.md, {results_dir}/tables.md and {len(pngs)} charts")


def _update_readme(readme: Path, summary: dict, fa: dict, n_cand: float, rand_r10: float) -> None:
    if not readme.exists():
        return
    text = readme.read_text()
    a, m = summary["variants"]["as_is"], summary["variants"]["masked"]
    d = a["deltas"]["learned_gbt_minus_bm25_file:recall@10"]
    dm = m["deltas"]["learned_gbt_minus_bm25_file:recall@10"]
    block = (
        f"Test split: **{summary['n_test']} real commits** (latest 30% of each repo's history), "
        f"95% bootstrap CIs in [REPORT.md](REPORT.md).\n\n"
        + compact_table(summary)
        + f"\nEach query ranks all ~{n_cand:.0f} `.py` files in its snapshot (random Recall@10 ≈ {rand_r10:.2f}).\n"
        + f"\n- Learned reranker vs BM25, Recall@10: **{d['mean']:+.3f}** "
        f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) as-is; **{dm['mean']:+.3f}** "
        f"({dm['ci95'][0]:+.3f} to {dm['ci95'][1]:+.3f}) with file names masked.\n"
        f"- {summary['leakage_rate_gold_stem_in_message']:.0%} of commit messages name a gold file outright — "
        "that is why the masked column exists.\n"
        f"- {fa['misses']} of {fa['total']} test queries ({fa['misses'] / fa['total']:.0%}) have no gold file "
        "in the learned reranker's top 10; see the failure analysis in REPORT.md.\n"
    )
    t = summary["timing"]
    sp = ", ".join(f"{r} {v['speedup']:.0f}×" for r, v in t.items())
    eq = sum(v["verified_equal"] for v in t.values())
    tot = sum(v["verified_total"] for v in t.values())
    block += (
        f"- Merkle-diff incremental re-indexing vs full rebuild: {sp} faster per snapshot; "
        f"incremental index identical to a fresh rebuild on {eq}/{tot} sampled snapshots.\n"
    )
    new = re.sub(
        r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
        lambda _: f"<!-- RESULTS:START -->\n{block}<!-- RESULTS:END -->",
        text,
        flags=re.S,
    )
    readme.write_text(new)


def by_repo(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["repo"]].append(r)
    return out
