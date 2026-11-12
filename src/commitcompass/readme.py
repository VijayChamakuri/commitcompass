"""Render README.md and docs/RESULTS.md from templates and results/*.json.

{{name}} placeholders are filled from the result files. tests/test_readme.py fails
if either generated file differs from a fresh render.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dense import POOL
from .features import FEATURES
from .swe_charts import LABELS

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs" / "README.template.md"
README = ROOT / "README.md"
RESULTS_TEMPLATE = ROOT / "docs" / "RESULTS.template.md"
RESULTS_DOC = ROOT / "docs" / "RESULTS.md"
TABLE_METHODS = [
    "bm25_file",
    "bm25_chunk",
    "hybrid_hand_tuned",
    "commit_reranker_loro",
    "commit_reranker",
    "dense_jina_code",
    "rrf_bm25_dense",
    "rrf_reranker_dense",
]


def p1(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(d: dict) -> str:
    return f"{p1(d['mean'])} <sub>[{p1(d['ci95'][0])}, {p1(d['ci95'][1])}]</sub>"


def delta(d: dict) -> str:
    return f"{100 * d['mean']:+.1f} pts (95% CI {100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f})"


def values() -> dict[str, str]:
    res = json.loads((ROOT / "results" / "swebench_lite.json").read_text())
    hist = json.loads((ROOT / "results" / "summary.json").read_text())
    agg, pub = res["aggregate"], {p["method"]: p for p in res["published"]}
    v: dict[str, Any] = {
        "n_test": str(res["n_test"]),
        "n_train": f"{res['n_train_commit_queries']:,}",
        "n_features": str(len(FEATURES)),
        "n_repos": str(len(res["train_queries_per_repo"])),
        "pool": str(POOL),
        "published_source": res["published_source"],
    }
    rows = ["| Method | Source | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in sorted((m for m in TABLE_METHODS if m in agg), key=lambda m: -agg[m]["acc@5"]["mean"]):
        a = agg[m]
        name = f"**{LABELS[m]}**" if m == res["best_method"] else LABELS[m]
        rows.append(
            f"| {name} | measured here | {ci(a['acc@1'])} | {ci(a['acc@3'])} | {ci(a['acc@5'])} | "
            f"{p1(a['acc@10']['mean'])} | {a['mrr']['mean']:.3f} |"
        )
    for p in res["published"]:
        rows.append(
            f"| {p['method']} | reported, {p['kind']} | {p['acc@1']:.1f} | {p['acc@3']:.1f} | {p['acc@5']:.1f} | – | – |"
        )
    v["leaderboard"] = "\n".join(rows)
    best = res["best_method"]
    v |= {
        "best_label": LABELS[best],
        "best_acc1": p1(agg[best]["acc@1"]["mean"]),
        "best_acc5": p1(agg[best]["acc@5"]["mean"]),
        "best_acc5_ci": f"{p1(agg[best]['acc@5']['ci95'][0])}–{p1(agg[best]['acc@5']['ci95'][1])}",
        "bm25_acc1": p1(agg["bm25_file"]["acc@1"]["mean"]),
        "bm25_acc5": p1(agg["bm25_file"]["acc@5"]["mean"]),
        "hyb_acc1": p1(agg["hybrid_hand_tuned"]["acc@1"]["mean"]),
        "hyb_acc5": p1(agg["hybrid_hand_tuned"]["acc@5"]["mean"]),
        "pub_bm25_acc5": f"{pub['BM25']['acc@5']:.1f}",
        "pub_bm25_acc1": f"{pub['BM25']['acc@1']:.1f}",
        "rr_acc1": p1(agg["commit_reranker"]["acc@1"]["mean"]),
        "rr_acc5": p1(agg["commit_reranker"]["acc@5"]["mean"]),
        "loro_acc5": p1(agg["commit_reranker_loro"]["acc@5"]["mean"]),
        "loro_acc1": p1(agg["commit_reranker_loro"]["acc@1"]["mean"]),
        "pub_jina_acc5": f"{pub['Jina-Code-v2']['acc@5']:.1f}",
        "pub_agentless_acc5": f"{pub['Agentless (Claude-3.5)']['acc@5']:.1f}",
        "pub_cre_acc5": f"{pub['CodeRankEmbed']['acc@5']:.1f}",
        "pub_locagent_acc5": f"{pub['LocAgent (Claude-3.5)']['acc@5']:.1f}",
    }
    if "dense_jina_code" in agg:
        v |= {
            "jina_acc5": p1(agg["dense_jina_code"]["acc@5"]["mean"]),
            "jina_acc1": p1(agg["dense_jina_code"]["acc@1"]["mean"]),
            "fused_acc5": p1(agg["rrf_reranker_dense"]["acc@5"]["mean"]),
            "fused_acc1": p1(agg["rrf_reranker_dense"]["acc@1"]["mean"]),
        }
    d = res["paired_deltas"]
    for key, name in [
        ("commit_reranker - bm25_file : acc@1", "d_rr_bm25_acc1"),
        ("commit_reranker - bm25_file : acc@5", "d_rr_bm25_acc5"),
        ("commit_reranker - hybrid_hand_tuned : acc@5", "d_rr_hyb_acc5"),
        ("commit_reranker - hybrid_hand_tuned : acc@1", "d_rr_hyb_acc1"),
        ("commit_reranker_loro - bm25_file : acc@5", "d_loro_bm25_acc5"),
        ("rrf_reranker_dense - dense_jina_code : acc@5", "d_fused_dense_acc5"),
        ("rrf_reranker_dense - commit_reranker : acc@5", "d_fused_rr_acc5"),
        ("rrf_reranker_dense - commit_reranker : acc@1", "d_fused_rr_acc1"),
    ]:
        if key in d:
            v[name] = delta(d[key])
            if key == "commit_reranker - bm25_file : acc@1":
                v["d_rr_bm25_acc1_pts"] = f"{100 * d[key]['mean']:.1f}"
    v["pool_recall"] = f"{100 * res.get('pool_recall', 0):.1f}"
    pr = res["per_repo_acc@5"]
    show = [m for m in ("bm25_file", "commit_reranker", "dense_jina_code", "rrf_reranker_dense") if m in agg]
    rows = ["| Repository | Issues | " + " | ".join(LABELS[m] for m in show) + " |", "|---|---:|" + "---:|" * len(show)]
    for repo in sorted(pr, key=lambda r: -pr[r]["n"]):
        rows.append(f"| {repo} | {pr[repo]['n']} | " + " | ".join(p1(pr[repo][m]) for m in show) + " |")
    v["per_repo_table"] = "\n".join(rows)
    mc = res["miss_categories"]
    total_miss = sum(c["count"] for c in mc.values())
    rows = [f"| Why {LABELS[best]} missed at k=5 | Issues | Examples |", "|---|---:|---|"]
    for cat, c in mc.items():
        rows.append(f"| {cat} | {c['count']} | {', '.join(f'`{e}`' for e in c['examples'][:3])} |")
    v["miss_table"], v["n_miss"] = "\n".join(rows), str(total_miss)
    imp = list(res["feature_importance"].items())[:6]
    v["top_features"] = ", ".join(f"`{k}`" for k, _ in imp)
    t = res["timing"]
    first = [x["first_build_s"] for x in t.values() if x.get("first_build_s")]
    incr = sum(x["incremental_s"] for x in t.values()) / max(1, sum(x["incremental_updates"] for x in t.values()))
    reidx = sum(x["files_reindexed"] for x in t.values()) / max(1, sum(x["incremental_updates"] for x in t.values()))
    django = t.get("django/django", {})
    v |= {
        "events": f"{sum(x['events'] for x in t.values()):,}",
        "verified": f"{sum(x['verified_equal'] for x in t.values())}/{sum(x['verified_total'] for x in t.values())}",
        "incr_ms": f"{1000 * incr:.0f}",
        "reindexed": f"{reidx:.1f}",
        "django_first": f"{django.get('first_build_s', 0):.1f}",
        "django_files": f"{django.get('mean_snapshot_files', 0):,.0f}",
        "feat_ms": f"{1000 * sum(x['features_s'] for x in t.values()) / max(1, sum(x['records'] for x in t.values())):.0f}",
        "replay_min": f"{sum(x['repo_seconds'] for x in t.values()) / 60:.0f}",
        "dense_files": f"{max(x.get('dense_files_embedded', 0) for x in t.values()):,}",  # running total
        "dense_min": f"{sum(x.get('dense_seconds', 0) for x in t.values()) / 60:.0f}",
        "first_build_max": f"{max(first):.1f}",
        "eval_min": f"{res['seconds'] / 60:.0f}",
        "nodense_min": f"{(sum(x['repo_seconds'] for x in t.values()) + res['seconds']) / 60:.0f}",
    }
    hv = hist["variants"]
    h_rows = [
        "| Method | Recall@10 | MRR | Recall@10 (file names masked) | MRR (masked) |",
        "|---|---:|---:|---:|---:|",
    ]
    hl = {
        "bm25_file": "BM25 (file)",
        "hybrid_hand_tuned": "Hand-tuned hybrid",
        "learned_gbt": "Commit-supervised reranker",
    }
    for m, lab in hl.items():
        a, b = hv["as_is"]["pooled"][m], hv["masked"]["pooled"][m]
        h_rows.append(
            f"| {lab} | {a['recall@10']['mean']:.3f} | {a['mrr']['mean']:.3f} | {b['recall@10']['mean']:.3f} | {b['mrr']['mean']:.3f} |"
        )
    v["history_table"], v["history_n"] = "\n".join(h_rows), f"{hist['n_test']:,}"
    gbt = hv["as_is"]["pooled"]["learned_gbt"]["recall@10"]["mean"]
    bm = hv["as_is"]["pooled"]["bm25_file"]["recall@10"]["mean"]
    v["hist_gbt_r10"] = f"{gbt:.3f}"
    v["hist_bm25_r10"] = f"{bm:.3f}"
    return {k: str(x) for k, x in v.items()}


def _fill(template: Path, vals: dict[str, str]) -> str:
    text = template.read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(vals))
    if missing:
        raise KeyError(f"{template.name} placeholders without values: {missing}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], text)


def render(write: bool = False) -> str:
    vals = values()
    readme = _fill(TEMPLATE, vals)
    results = _fill(RESULTS_TEMPLATE, vals)
    if write:
        README.write_text(readme)
        RESULTS_DOC.write_text(results)
    return readme


def render_results() -> str:
    return _fill(RESULTS_TEMPLATE, values())
