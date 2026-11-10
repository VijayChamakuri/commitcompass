"""Build site/ (GitHub Pages): leaderboard + per-instance explorer, from results files only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .swe_charts import LABELS

SITE = Path("site")
REPO_URL = "https://github.com/VijayChamakuri/commitcompass"
EXPLORE = ["commit_reranker", "rrf_reranker_dense", "dense_jina_code", "bm25_file"]


def build(results: Path) -> Path:
    res = json.loads((results / "swebench_lite.json").read_text())
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
    agg = res["aggregate"]
    ours = sorted(((m, agg[m]) for m in agg if m in LABELS), key=lambda kv: -kv[1]["acc@5"]["mean"])

    def fmt(d: dict) -> str:
        return f"{100 * d['mean']:.1f}<span class=ci> [{100 * d['ci95'][0]:.1f}–{100 * d['ci95'][1]:.1f}]</span>"

    table = "".join(
        f"<tr><td>{LABELS[m]}</td><td>measured</td><td>{fmt(v['acc@1'])}</td><td>{fmt(v['acc@3'])}</td><td>{fmt(v['acc@5'])}</td></tr>"
        for m, v in ours
    )
    table += "".join(
        f"<tr class=pub><td>{p['method']}</td><td>reported ({p['kind']})</td><td>{p['acc@1']:.1f}</td>"
        f"<td>{p['acc@3']:.1f}</td><td>{p['acc@5']:.1f}</td></tr>"
        for p in res["published"]
    )
    best = res["best_method"]
    page = (
        TEMPLATE.replace("{{table}}", table)
        .replace("{{repo}}", REPO_URL)
        .replace("{{source}}", res["published_source"])
    )
    page = page.replace("{{labels}}", json.dumps({m: LABELS[m] for m in EXPLORE})).replace(
        "{{best}}", LABELS.get(best, best)
    )
    page = page.replace("{{best_acc5}}", f"{100 * agg[best]['acc@5']['mean']:.1f}%").replace(
        "{{n}}", str(res["n_test"])
    )
    page = page.replace("{{ntrain}}", f"{res['n_train_commit_queries']:,}")
    (SITE / "index.html").write_text(page)
    (SITE / ".nojekyll").write_text("")
    return SITE / "index.html"


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CommitCompass</title><meta name="description" content="Code localization on SWE-bench Lite with a reranker trained only on git history.">
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e6e3dc;--card:#fff;--accent:#2457c5;--hit:#e5f4ea;--hitfg:#1f7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121212;--fg:#ecebe7;--muted:#a3a19a;--line:#2b2a27;--card:#1a1a19;--accent:#7aa2ff;--hit:#173323;--hitfg:#7fdca8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 16px 80px}h1{font-size:clamp(26px,5vw,40px);margin:0 0 10px;letter-spacing:-.02em}
h2{margin:48px 0 8px}p{max-width:74ch}.lede{font-size:18px;color:var(--muted)}a{color:var(--accent)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--muted);font-weight:600}tr.pub td{color:var(--muted)}.ci{color:var(--muted);font-size:12px}
img{max-width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}input,select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.list{display:grid;grid-template-columns:minmax(0,1fr);gap:8px}.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
.item b{font-family:ui-monospace,monospace;font-size:13px}.tag{font-size:12px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);margin-left:6px;color:var(--muted)}
.detail{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:10px}
pre{white-space:pre-wrap;font-size:12.5px;max-height:260px;overflow:auto;background:var(--bg);padding:10px;border-radius:8px}
ol{padding-left:22px;font-family:ui-monospace,monospace;font-size:12.5px}li.hit{background:var(--hit);color:var(--hitfg);font-weight:600}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
</style></head><body><main>
<nav><b>CommitCompass</b> · <a href="{{repo}}">GitHub</a></nav>
<h1>Which files does a coding agent actually need?</h1>
<p class="lede">File-level localization on all {{n}} SWE-bench Lite issues. The reranker never sees a SWE-bench label: it is trained on
{{ntrain}} commit-message → modified-file pairs mined from each repository's history before its first issue. Best measured: <b>{{best}}</b>, Acc@5 <b>{{best_acc5}}</b>.</p>
<h2>Leaderboard</h2>
<p>Acc@k counts an issue only if its gold file is in the top k (same definition as the source of the reported rows). Measured rows show 95% bootstrap CIs.
Reported rows are copied from {{source}} and were not re-run here.</p>
<div class="tablewrap"><table><thead><tr><th>Method</th><th>Source</th><th>Acc@1</th><th>Acc@3</th><th>Acc@5</th></tr></thead><tbody>{{table}}</tbody></table></div>
<p><img src="swebench_leaderboard.png" alt="Leaderboard chart"></p>
<div class="cols"><img src="swebench_acc_at_k.png" alt="Acc@k"><img src="swebench_per_repo.png" alt="Per-repository Acc@5"></div>
<h2>Explore every instance</h2>
<div class="controls"><input id="q" placeholder="search issue text or file" size="28"><select id="repo"></select><select id="method"></select>
<select id="f"><option value="all">all</option><option value="miss">missed at 5</option><option value="hit">hit at 5</option></select><span id="count" class="ci"></span></div>
<div class="list" id="list"></div>
</main>
<script>
const LABELS={{labels}};let DATA=[];
const $=id=>document.getElementById(id);
fetch('data/instances.json').then(r=>r.json()).then(d=>{DATA=d;const repos=['all',...new Set(d.map(x=>x.repo))];
$('repo').innerHTML=repos.map(r=>`<option>${r}</option>`).join('');$('method').innerHTML=Object.keys(LABELS).filter(m=>d[0].methods[m]).map(m=>`<option value="${m}">${LABELS[m]}</option>`).join('');render();});
['q','repo','method','f'].forEach(id=>$(id).addEventListener('input',render));
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(){const q=$('q').value.toLowerCase(),repo=$('repo').value,m=$('method').value,f=$('f').value;
const rows=DATA.filter(x=>(repo==='all'||x.repo===repo)&&(!q||x.issue.toLowerCase().includes(q)||x.gold.join(' ').includes(q)||x.id.includes(q))
&&(f==='all'||(f==='miss')===((x.methods[m].rank||1e9)>5)));
$('count').textContent=rows.length+' instances';
$('list').innerHTML=rows.slice(0,150).map(x=>{const r=x.methods[m].rank;return `<div class="item" data-id="${x.id}"><b>${x.id}</b>
<span class="tag">gold rank ${r??'–'} / ${x.n}</span><span class="tag">${esc(x.gold[0])}</span><div class="detail" hidden></div></div>`}).join('');
document.querySelectorAll('.item').forEach(el=>el.addEventListener('click',()=>{const x=DATA.find(d=>d.id===el.dataset.id),det=el.querySelector('.detail');
if(!det.hidden){det.hidden=true;return}det.hidden=false;det.innerHTML=`<pre>${esc(x.issue)}</pre><div class="cols">`+Object.keys(x.methods).map(k=>
`<div><b>${LABELS[k]}</b><ol>${x.methods[k].top10.map(p=>`<li class="${x.gold.includes(p)?'hit':''}">${esc(p)}</li>`).join('')}</ol></div>`).join('')+'</div>';}));}
</script></body></html>
"""
