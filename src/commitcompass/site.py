"""Build site/ (GitHub Pages): a plain write-up plus a per-instance explorer, from results files only.

Numbers and tables come from the same `values()` that renders the README, so the two cannot disagree.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

from .swe_charts import LABELS

SITE = Path("site")
REPO_URL = "https://github.com/VijayChamakuri/commitcompass"
EXPLORE = ["commit_reranker", "rrf_reranker_dense", "dense_jina_code", "bm25_file"]


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"&lt;sub&gt;(.+?)&lt;/sub&gt;", r"<span class=dim>\1</span>", s)  # the README's CI brackets
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def md_table(text: str) -> str:
    """Render one of the README's pipe tables as an HTML table (right-aligned where the README says ---:)."""
    lines = [ln.strip().strip("|") for ln in text.strip().splitlines()]
    head, align, body = lines[0], lines[1], lines[2:]
    right = [a.strip().endswith(":") for a in align.split("|")]

    def cells(row: str, tag: str) -> str:
        parts = [c.strip() for c in row.split("|")]
        return "".join(
            f"<{tag}{' class=n' if i < len(right) and right[i] else ''}>{_inline(c)}</{tag}>"
            for i, c in enumerate(parts)
        )

    rows = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in body)
    return f"<div class=scroll><table><thead><tr>{cells(head, 'th')}</tr></thead><tbody>{rows}</tbody></table></div>"


def build(results: Path) -> Path:
    from .readme import values

    inst = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in (Path("data") / "swebench_lite.jsonl").read_text().splitlines()
    }
    rows = [json.loads(line) for line in (results / "swebench_lite_per_instance.jsonl").read_text().splitlines()]
    explore = []
    for r in rows:
        methods = {
            m: {"rank": r["methods"][m]["rank"], "top10": r["methods"][m]["top10"]}
            for m in EXPLORE
            if m in r["methods"]
        }
        explore.append(
            {
                "id": r["instance_id"],
                "repo": r["repo"],
                "gold": r["gold"],
                "n": r["n_candidates"],
                "issue": inst[r["instance_id"]]["problem_statement"][:1600],
                "methods": methods,
            }
        )
    SITE.mkdir(exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "instances.json").write_text(json.dumps(explore))
    for png in results.glob("swebench_*.png"):
        shutil.copy(png, SITE / png.name)

    v = values()
    page = TEMPLATE.replace("{{css}}", CSS).replace("{{repo}}", REPO_URL)
    page = page.replace("{{labels}}", json.dumps({m: LABELS[m] for m in EXPLORE}))
    for k, val in v.items():
        key = "{{" + k + "}}"
        if key in page:
            page = page.replace(key, md_table(val) if k.endswith("_table") or k == "leaderboard" else html.escape(val))
    missing = re.findall(r"\{\{(\w+)\}\}", page)
    if missing:
        raise KeyError(f"unfilled placeholders in site template: {missing}")
    (SITE / "index.html").write_text(page)
    (SITE / ".nojekyll").write_text("")
    return SITE / "index.html"


# Shared with the TrustGate and JudgeGate pages so the three read as one body of work.
CSS = """
:root{--paper:#ffffff;--ink:#1c1b19;--ink2:#47443e;--muted:#77736a;--rule:#d9d3c6;--rule2:#ebe6db;--link:#1a4c8c;--flag:#a3261b;--code:#f2eee3;--hit:#e3efdc}
html{color-scheme:light}*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.62 Charter,"Bitstream Charter","Iowan Old Style","Sitka Text",Cambria,Georgia,serif;-webkit-text-size-adjust:100%}
.page{max-width:700px;margin:0 auto;padding:20px 20px 96px}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 24px;padding-bottom:12px;margin-bottom:56px;border-bottom:1px solid var(--rule);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted)}
.top a{color:var(--ink2);text-decoration:none}.top a:hover{color:var(--link);text-decoration:underline}
.top nav{display:flex;flex-wrap:wrap;gap:4px 18px}.top nav span{color:var(--ink);font-weight:600}
h1{font-size:clamp(28px,5.4vw,38px);line-height:1.18;letter-spacing:-.01em;margin:0 0 14px}
.meta{font-style:italic;color:var(--muted);font-size:16px;margin:0 0 40px}
h2{font-size:24px;line-height:1.3;margin:60px 0 12px}
p{margin:0 0 18px}ol,ul{margin:0 0 18px;padding-left:26px}li{margin:0 0 12px}li::marker{color:var(--muted)}
a{color:var(--link);text-decoration-thickness:1px;text-underline-offset:3px}
code{font:.8em ui-monospace,"SF Mono",Menlo,Consolas,monospace;background:var(--code);padding:2px 4px;border-radius:3px}
.wide{position:relative;left:50%;transform:translateX(-50%);width:min(1000px,calc(100vw - 40px))}
figure{margin:36px 0}figure.narrow img{max-width:620px}figure img{display:block;width:100%;height:auto;margin:0 auto}
figcaption,.cap{font-size:15.5px;line-height:1.5;color:var(--ink2);max-width:700px;margin:12px auto 0}
.cap{margin:0 auto 10px}figcaption b,.cap b{color:var(--ink)}
.tbl{margin:32px 0}.scroll{overflow-x:auto}
.tbl.wide{width:fit-content;min-width:min(700px,calc(100vw - 40px));max-width:min(1000px,calc(100vw - 40px))}.cap{contain:inline-size}
table{border-collapse:collapse;width:100%;font:13.5px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink)}
th{font-weight:600;text-align:left;vertical-align:bottom;padding:8px 10px;border-bottom:1px solid var(--ink)}
td{padding:6px 10px;border-bottom:1px solid var(--rule2);vertical-align:top}tr:last-child td{border-bottom:0}
.n{text-align:right;white-space:nowrap}th.n{white-space:normal;min-width:5em}td:first-child{min-width:11em}td:nth-child(2){white-space:nowrap}.dim{color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:0 0 14px;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink2)}
select,input{font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:5px 8px;border:1px solid #b9b2a3;border-radius:3px;background:#fff;color:var(--ink);margin-left:6px;max-width:100%}
#list{border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
details{border-bottom:1px solid var(--rule2)}details:last-child{border-bottom:0}
summary{cursor:pointer;padding:7px 4px;display:grid;grid-template-columns:17em 10em minmax(0,1fr);gap:2px 16px;list-style:none}summary::-webkit-details-marker{display:none}
summary .id{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:13px}summary .r{color:var(--muted);overflow-wrap:anywhere}
details[open] summary{background:#faf8f3}.body{padding:8px 4px 18px}
pre{white-space:pre-wrap;font:12.5px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace;max-height:280px;overflow:auto;background:#faf8f3;border-left:3px solid var(--rule);padding:10px 12px;margin:0 0 14px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 20px}.cols h4{margin:6px 0 2px;font-size:13px}
.cols ol{font:12px/1.55 ui-monospace,"SF Mono",Menlo,Consolas,monospace;padding-left:24px;margin:0}.cols li{margin:0;word-break:break-all}li.hit{background:var(--hit);font-weight:700}
footer{margin-top:80px;padding-top:14px;border-top:1px solid var(--rule);font-size:15px;color:var(--muted)}
@media (max-width:600px){body{font-size:17px}.top{margin-bottom:36px}h2{margin-top:48px}summary{grid-template-columns:1fr}}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CommitCompass</title>
<meta name="description" content="Which file does a coding agent need to open? A retriever trained only on git history, tested on all 300 SWE-bench Lite issues.">
<style>{{css}}</style></head><body><div class=page>
<header class=top><a href="https://github.com/VijayChamakuri">Vijay Chamakuri</a>
<nav><a href="https://vijaychamakuri.github.io/trustgate/">TrustGate</a><span>CommitCompass</span><a href="https://vijaychamakuri.github.io/judgegate/">JudgeGate</a><a href="https://vijaychamakuri.github.io/sift/">Sift</a></nav></header>
<main>
<h1>Which file does a coding agent need to open?</h1>
<p class=meta>Vijay Chamakuri, November 2026. Code, data and full method on <a href="{{repo}}">GitHub</a>.</p>

<p>When a coding agent gets a bug report, its first job is to find the file to change. If it opens the wrong one, nothing it
writes afterwards will fix the bug. Retrievers for this step are usually trained on labelled data or lean on a large paid model.
I wanted to see how far you get with supervision every repository already has: its git history. Each commit pairs a description
of a change with the files it touched.</p>
<p>CommitCompass trains a small reranker on {{n_train}} of those commit-message and changed-file pairs, all from before each
repository's first benchmark issue, and tests it on all {{n_test}} issues in SWE-bench Lite. It never sees a SWE-bench label, and
everything runs on a laptop CPU.</p>

<h2>What I found</h2>
<ol>
<li><strong>Git history is enough supervision to beat keyword search.</strong> The commit-trained reranker puts the right file
first in {{rr_acc1}}% of issues, against {{bm25_acc1}}% for BM25, a difference of {{d_rr_bm25_acc1}}.</li>
<li><strong>It works on codebases it has never seen.</strong> Trained with each repository's own history held out, it still
reaches {{loro_acc1}}% at Acc@1 and {{loro_acc5}}% at Acc@5.</li>
<li><strong>With a small local code-embedding model added</strong>, the right file is in the top five {{best_acc5}}% of the time
(95% CI {{best_acc5_ci}}). Published systems that call large paid models reach up to {{pub_locagent_acc5}}%. The full comparison
is below.</li>
<li><strong>The learning adds less than the features do.</strong> Against a hand-tuned mix of the same 18 features the difference
is not significant (Acc@1 {{d_rr_hyb_acc1}}). Most of the gain over BM25 comes from path, symbol and history signals. Learning
their weights from commits matches careful hand-tuning without anyone doing the tuning, and carries over to new repositories.</li>
<li><strong>The index keeps up with change cheaply.</strong> Replaying {{events}} repository snapshots, the incremental index
re-read {{reindexed}} files per update on average and matched a full rebuild in {{verified}} spot-checks.</li>
</ol>

<h2>How it compares</h2>
<p>Acc@k counts an issue as found if the file that was actually changed is among the top k suggestions. Measured rows show a 95%
bootstrap interval over the {{n_test}} issues. Reported rows are copied from {{published_source}} and were not re-run here; the LLM
rows use paid APIs and nothing measured here does. My BM25 lands close to the published one ({{bm25_acc5}}% against
{{pub_bm25_acc5}}% at Acc@5), which is a useful check that the setup matches.</p>
<div class="tbl wide"><p class=cap><b>Table 1.</b> File-level localization on SWE-bench Lite, {{n_test}} issues.</p>{{leaderboard}}</div>
<figure class=narrow><img src="swebench_leaderboard.png" alt="Horizontal bar chart of Acc@5 for measured and reported methods. Measured methods sit between BM25 and the LLM agents.">
<figcaption><b>Figure 1.</b> Acc@5 for every method. Blue bars were measured here; grey and orange bars are reported numbers.</figcaption></figure>
<figure class=narrow><img src="swebench_acc_at_k.png" alt="Accuracy as the number of suggested files grows, for each measured method.">
<figcaption><b>Figure 2.</b> Accuracy as the number of suggested files grows.</figcaption></figure>
<div class="tbl wide"><p class=cap><b>Table 2.</b> Acc@5 by repository.</p>{{per_repo_table}}</div>
<div class="tbl wide"><p class=cap><b>Table 3.</b> Why the best method missed at k=5, sorted into rough categories.</p>{{miss_table}}</div>

<h2>Every issue, one by one</h2>
<p>All {{n_test}} issues, with each method's top ten files. The file that was actually changed is highlighted.</p>
<div class=wide><div class=controls><label>Search<input id=q placeholder="issue text, file or id" size=22></label><label>Repository<select id=repo></select></label>
<label>Method<select id=method></select></label><label>Show<select id=f><option value=all>all issues</option><option value=miss>missed at 5</option><option value=hit>found in top 5</option></select></label>
<span id=count class=dim></span></div>
<div id=list></div></div>

<footer><p>Every number on this page comes from <code>results/</code> and is rebuilt by <code>make reproduce</code>; CI fails if
the README and results disagree.</p>
<p>Code: MIT licence. SWE-bench is MIT-licensed, and each evaluated repository keeps its own licence. Embeddings:
jina-embeddings-v2-base-code, run locally.</p></footer>
</main></div>
<script>
const LABELS={{labels}};let DATA=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
fetch("data/instances.json").then(r=>r.json()).then(d=>{DATA=d;
$("repo").innerHTML=["all",...new Set(d.map(x=>x.repo))].map(r=>`<option>${r}</option>`).join("");
$("method").innerHTML=Object.keys(LABELS).filter(m=>d[0].methods[m]).map(m=>`<option value="${m}">${LABELS[m]}</option>`).join("");render()});
["q","repo","method","f"].forEach(id=>$(id).addEventListener("input",render));
function render(){const q=$("q").value.toLowerCase(),repo=$("repo").value,m=$("method").value,f=$("f").value;
const rows=DATA.filter(x=>(repo==="all"||x.repo===repo)&&(!q||x.issue.toLowerCase().includes(q)||x.gold.join(" ").includes(q)||x.id.includes(q))
&&(f==="all"||(f==="miss")===((x.methods[m].rank||1e9)>5)));
$("count").textContent=rows.length+(rows.length===1?" issue":" issues")+(rows.length>150?", first 150 shown":"");
$("list").innerHTML=rows.slice(0,150).map(x=>{const r=x.methods[m].rank;return `<details data-id="${x.id}"><summary><span class=id>${x.id}</span>
<span class=r>${r?"rank "+r+" of "+x.n.toLocaleString("en"):"not in the candidate pool"}</span><span class=r>${esc(x.gold[0])}</span></summary><div class=body></div></details>`}).join("");}
$("list").addEventListener("toggle",e=>{const el=e.target;if(!el.open)return;const x=DATA.find(d=>d.id===el.dataset.id);
el.querySelector(".body").innerHTML=`<pre>${esc(x.issue)}</pre><div class=cols>`+Object.keys(x.methods).map(k=>
`<div><h4>${LABELS[k]}</h4><ol>${x.methods[k].top10.map(p=>`<li class="${x.gold.includes(p)?"hit":""}">${esc(p)}</li>`).join("")}</ol></div>`).join("")+"</div>"},true);
</script></body></html>
"""
