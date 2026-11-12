"""Charts for SWE-bench Lite results (results/swebench_*.png)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import chartstyle as cs

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
COLOUR = {
    "rrf_reranker_dense": cs.BLUE,
    "commit_reranker": cs.PURPLE,
    "hybrid_hand_tuned": cs.ORANGE,
    "dense_jina_code": cs.GREEN,
    "bm25_file": cs.NEUTRAL,
}
OURS, PUB, LLM = cs.BLUE, "#d3cec4", cs.ORANGE


def _spread(points: list[list], gap: float) -> list[list]:
    """Push end-of-line labels apart so they never overlap (points are [y, label, ...], sorted high to low)."""
    points.sort(key=lambda p: -p[0])
    for i in range(1, len(points)):
        points[i][0] = min(points[i][0], points[i - 1][0] - gap)
    return points


def _lab(m: str) -> str:
    """Chart label: the chart font has no ⊕, so fused methods read "A + B (RRF)" in figures."""
    return LABELS[m].replace(" ⊕ ", " + ")


def leaderboard(res: dict, out: Path) -> None:
    cs.apply()
    agg = res["aggregate"]
    rows: list[tuple[str, float, list[float] | None, str]] = [
        (f"{p['method']} (reported)", p["acc@5"], None, LLM if "LLM" in p["kind"] else PUB) for p in res["published"]
    ]
    for m in SHOW:
        if m in agg:
            a = agg[m]["acc@5"]
            rows.append((_lab(m), 100 * a["mean"], [100 * a["ci95"][0], 100 * a["ci95"][1]], OURS))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(rows) + 1.2))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
    for yi, r in zip(y, rows, strict=True):
        if r[2]:
            ax.plot(r[2], [yi, yi], color=cs.INK, lw=1)
        x_label = (r[2][1] if r[2] else r[1]) + 0.8  # after the CI whisker, never on it
        ax.text(x_label, float(yi), f"{r[1]:.1f}", va="center", fontsize=8, color=cs.INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Acc@5 on SWE-bench Lite (%): every changed file in the top 5")
    ax.set_title("File localization on SWE-bench Lite")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=OURS, label="Measured here (no paid model), with 95% CI"),
            Patch(color=PUB, label="Reported: retrieval"),
            Patch(color=LLM, label="Reported: LLM pipelines and agents"),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.1),
        ncol=3,
        handlelength=1,
    )
    fig.tight_layout()
    fig.savefig(out / "swebench_leaderboard.png")
    plt.close(fig)


def acc_curves(res: dict, out: Path) -> None:
    cs.apply()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ks = [1, 3, 5, 10]
    ends = []
    for m in SHOW:
        if m not in res["aggregate"]:
            continue
        ys = [100 * res["aggregate"][m][f"acc@{k}"]["mean"] for k in ks]
        ax.plot(ks, ys, marker="o", color=COLOUR[m], lw=2.2 if m == "rrf_reranker_dense" else 1.6)
        ends.append([ys[-1], _lab(m)])
    for yv, lab in _spread(ends, 1.9):
        ax.annotate(lab, (10, yv), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=cs.INK2)
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 16)
    ax.spines["bottom"].set_bounds(0.5, 10.5)
    ax.set_xlabel("Files suggested (k)")
    ax.set_ylabel("Acc@k (%)")
    ax.set_title(f"Accuracy as more files are suggested, {res['n_test']} issues")
    fig.tight_layout()
    fig.savefig(out / "swebench_acc_at_k.png")
    plt.close(fig)


def per_repo(res: dict, out: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cs.apply()
    pr = res["per_repo_acc@5"]
    repos = sorted(pr, key=lambda r: -pr[r]["n"])
    methods = [m for m in SHOW if m in next(iter(pr.values()))]
    M = np.array([[pr[r][m] for m in methods] for r in repos])
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(repos) + 1.6))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("ink_blue", ["#ffffff", cs.BLUE])
    ax.imshow(M, cmap=cmap, vmin=0.25, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(-0.5, len(methods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(repos)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(repos)):
        for j in range(len(methods)):
            ax.text(
                j,
                i,
                f"{100 * M[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if M[i, j] > 0.72 else cs.INK,
            )
    ax.set_xticks(range(len(methods)), [_lab(m) for m in methods], rotation=25, ha="right")
    ax.set_yticks(range(len(repos)), [f"{r.split('/')[1]} ({pr[r]['n']})" for r in repos])
    ax.set_title("Acc@5 by repository (%), issue count in brackets")
    fig.tight_layout()
    fig.savefig(out / "swebench_per_repo.png")
    plt.close(fig)


def build_all(results: Path) -> None:
    res = json.loads((results / "swebench_lite.json").read_text())
    leaderboard(res, results)
    acc_curves(res, results)
    per_repo(res, results)
