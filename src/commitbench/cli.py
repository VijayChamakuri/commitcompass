"""Command line entry point.

commitbench swebench     SWE-bench Lite: fetch, replay 12 repos, dense pass, evaluate, charts
commitbench history      commit-history benchmark (flask, requests, httpx): mine, eval, report
commitbench site         build site/ (leaderboard + per-instance explorer)
commitbench readme       render README.md from results (CI checks it is up to date)
commitbench reproduce    everything above, in order
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def run_swebench(results: Path, dense: bool = True) -> None:
    from .dense import DenseScorer
    from .swe_charts import build_all
    from .swe_eval import evaluate, write
    from .swe_run import CACHE, DATA, replay_all
    from .swebench import load

    replayed = replay_all()
    if dense:
        from .swe_run import add_dense

        add_dense(replayed, DenseScorer(CACHE / "dense"))
    records = [r for recs, _ in replayed.values() for r in recs]
    issue_ts = {
        i.instance_id: datetime.fromisoformat(i.created_at.replace("Z", "+00:00")).timestamp()
        for i in load(DATA / "swebench_lite.jsonl")
    }
    res = evaluate(records, issue_ts)
    res["timing"] = {repo: t for repo, (_, t) in replayed.items()}
    write(res, results)
    build_all(results)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="commitbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "command", choices=["swebench", "history", "site", "readme", "reproduce", "mine", "eval", "report", "all"]
    )
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--no-dense", action="store_true", help="skip the dense (embedding) pass")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    args.data.mkdir(parents=True, exist_ok=True)
    cmd = args.command
    history = cmd in ("history", "all", "reproduce")
    if history or cmd == "mine":
        from .mine import mine_all

        mine_all(args.data)
    if history or cmd == "eval":
        from .evaluate import run

        run(args.data, args.results)
    if history or cmd == "report":
        from .report import write_report

        write_report(args.results, Path("."))
    if cmd in ("swebench", "reproduce"):
        run_swebench(args.results, dense=not args.no_dense)
    if cmd in ("site", "reproduce"):
        from .site import build

        print(build(args.results))
    if cmd in ("readme", "reproduce"):
        from .readme import render

        render(write=True)
    print(f"[commitbench] {cmd} finished in {time.perf_counter() - t0:.1f}s")
