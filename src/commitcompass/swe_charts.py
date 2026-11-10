"""Charts for SWE-bench Lite results (results/swebench_*.png)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "bm25_file": "BM25, file",
    "bm25_chunk": "BM25, chunk-max",
    "path_match": "Path / filename match",
    "symbol_match": "Symbol match",
    "history_prior": "History prior",
    "hybrid_hand_tuned": "Hand-tuned hybrid",
    "commit_reranker": "Commit-supervised reranker",
    "commit_reranker_loro": "Commit reranker, leave-one-repo-out",
    "dense_jina_code": "Jina-Code-v2 pool rerank (local)",
    "rrf_bm25_dense": "BM25 ⊕ Jina-Code (RRF)",
    "rrf_reranker_dense": "Commit reranker ⊕ Jina-Code (RRF)",
}
SHOW = ["bm25_file", "hybrid_hand_tuned", "commit_reranker", "dense_jina_code", "rrf_reranker_dense"]
OURS, PUB, LLM = "#2457c5", "#b8bec6", "#e3a33b"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
        }
    )


def leaderboard(res: dict, out: Path) -> None:
    _style()
    agg = res["aggregate"]
    rows: list[tuple[str, float, list[float] | None, str]] = [
        (f"{p['method']} (reported)", p["acc@5"], None, LLM if "LLM" in p["kind"] else PUB) for p in res["published"]
    ]
    for m in SHOW:
        if m in agg:
            a = agg[m]["acc@5"]
            rows.append((f"{LABELS[m]} (ours)", 100 * a["mean"], [100 * a["ci95"][0], 100 * a["ci95"][1]], OURS))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.7)
    for yi, r in zip(y, rows, strict=True):
        if r[2]:
            ax.plot(r[2], [yi, yi], color="#111", lw=1.2)
        x_label = (r[2][1] if r[2] else r[1]) + 0.8  # after the CI whisker, never on it
        ax.text(x_label, float(yi), f"{r[1]:.1f}", va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("file-level Acc@5 on SWE-bench Lite (%), all gold files in top 5")
    ax.set_title("SWE-bench Lite file localization, Acc@5")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=OURS, label="measured here (CPU, no API)"),
            Patch(color=PUB, label="reported: retrieval"),
            Patch(color=LLM, label="reported: LLM pipelines / agents"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.09),
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out / "swebench_leaderboard.png", dpi=170)
    plt.close(fig)


def acc_curves(res: dict, out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ks = [1, 3, 5, 10]
    for m in SHOW:
        if m not in res["aggregate"]:
            continue
        ax.plot(
            ks,
            [100 * res["aggregate"][m][f"acc@{k}"]["mean"] for k in ks],
            marker="o",
            label=LABELS[m],
            lw=2.2 if m.startswith("rrf_reranker") or m == "commit_reranker" else 1.4,
        )
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Acc@k (%)")
    ax.set_title("Acc@k on SWE-bench Lite (300 issues)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "swebench_acc_at_k.png", dpi=170)
    plt.close(fig)


def per_repo(res: dict, out: Path) -> None:
    _style()
    pr = res["per_repo_acc@5"]
    repos = sorted(pr, key=lambda r: -pr[r]["n"])
    methods = [m for m in SHOW if m in next(iter(pr.values()))]
    M = np.array([[pr[r][m] for m in methods] for r in repos])
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(repos) + 1.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(repos)):
        for j in range(len(methods)):
            ax.text(
                j,
                i,
                f"{100 * M[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if M[i, j] > 0.6 else "#222",
            )
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(repos)), [f"{r.split('/')[1]} (n={pr[r]['n']})" for r in repos], fontsize=8)
    ax.set_title("Acc@5 by repository")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "swebench_per_repo.png", dpi=170)
    plt.close(fig)


def build_all(results: Path) -> None:
    res = json.loads((results / "swebench_lite.json").read_text())
    leaderboard(res, results)
    acc_curves(res, results)
    per_repo(res, results)
