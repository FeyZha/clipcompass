"""Fixed functional acceptance, plus known mixed50 regression; not a blind accuracy claim."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
from evaluation.english_v1 import ROOT, read, save, digest
from evaluation.mixed50_eval import span_scores, verify

DATA = ROOT / "evaluation_data/intent_v2"
CASES = [
    ("I01","介绍梯度下降","learn"), ("I02","梯度下降入门","learn"),
    ("I03","想学梯度下降","learn"), ("I04","我想了解货币和通货膨胀","learn"),
    ("I05","我想了解食品发酵","learn"), ("I06","介绍黑洞","learn"),
    ("I07","黑洞入门","learn"), ("I08","我想学黑洞相关的知识","learn"),
    ("I09","介绍 LoRA 低秩适配","absent"),
    ("I10","我想学习 Kubernetes PodDisruptionBudget","absent"),
    ("I11","这个是怎么工作的？","clarify"),
    ("I12","用一个数值例子解释梯度下降，不要推导","constrained"),
    ("I13","SGD 和 Adam 有什么区别？","constrained"),
    ("I14","不要讲定义，解释 Momentum 如何帮助跳出鞍点","constrained"),
]


def freeze():
    verify()
    cases=[{"id":i,"question":q,"expected":kind} for i,q,kind in CASES]
    # Known frozen regression: reuse exact old questions and labels; no gold edits.
    cases += [{"id":c["id"],"question":c["question"],"expected":c["kind"]}
              for c in read(ROOT / "evaluation_data/mixed50_v1/cases.json") if c["kind"] in ("specific","absent")]
    save(DATA / "cases.json",cases)
    paths=[DATA/"cases.json",ROOT/"library/intent_pipeline_v2.py",ROOT/"evaluation/intent_v2_check.py"]
    save(DATA / "manifest.json",{"scope":"40-case functional acceptance and known regression, not blind validation",
        "criteria":{"learn":"nonempty report and original topic retained (manual review)",
                    "constrained":"specific route preserves original request; evidence may be absent",
                    "latency":"report P95 <= 60s in this local run",
                    "time_regression":"precision and coverage each no more than 5 percentage points below prior run",
                    "negative":"all absent cases rejected, no technical errors"},
        "hashes":{str(p.relative_to(ROOT)):digest(p) for p in paths}})


def check():
    verify()
    for p,h in read(DATA/"manifest.json")["hashes"].items():
        assert digest(ROOT/p)==h,p


def run(base,output):
    check()
    output.mkdir(parents=True,exist_ok=False)
    for case in read(DATA/"cases.json"):
        start=time.perf_counter()
        row=dict(case)
        request=urllib.request.Request(base+"/api/search",json.dumps({"question":case["question"],"collection_id":"mixed50-v1"}).encode(),{"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request,timeout=250) as response:
                row.update(http_status=response.status,response=json.load(response))
        except urllib.error.HTTPError as exc:
            row.update(http_status=exc.code,response=json.load(exc))
        except Exception as exc:
            row.update(http_status=0,error=str(exc),response={})
        row["latency_ms"]=round((time.perf_counter()-start)*1000,1)
        save(output/(case["id"]+".json"),row)
        print(case["id"],row["http_status"],row["response"].get("status"),flush=True)
    check()
    evaluate(output)


def evaluate(output):
    check()
    gold={c["id"]:c for c in read(ROOT/"evaluation_data/mixed50_v1/cases.json")}
    details=[]
    for case in read(DATA/"cases.json"):
        row=read(output/(case["id"]+".json")); r=row["response"]
        expected=case["expected"]
        passed=(r.get("error")=="clarification_required" if expected=="clarify" else
                row["http_status"]==200 and (
                bool(r.get("results")) if expected in ("learn","specific") else
                not r.get("answerable") if expected=="absent" else r.get("plan",{}).get("task_type") in ("specific","compare","procedure")))
        detail={**case,"contract_pass":passed,"latency_ms":row["latency_ms"],"status":r.get("status"),
                "sections":len(r.get("learning_map",{}).get("groups",[]))}
        if case["id"] in gold:
            detail.update(span_scores(r.get("results",[]),gold[case["id"]]["gold_evidence"]))
        details.append(detail)
    scored=[d for d in details if "returned_seconds" in d]
    total=sum(d["returned_seconds"] for d in scored); hit=sum(d["hit_seconds"] for d in scored)
    target=sum(d["gold_seconds"] for d in scored)
    learn=sorted(d["latency_ms"] for d in details if d["expected"]=="learn")
    summary={"count":len(details),"contracts_passed":sum(d["contract_pass"] for d in details),
             "time_precision":hit/total if total else None,"gold_coverage":hit/target,
             "report_p95_ms":learn[-1],"details":details,
             "caveat":"Contract checks do not establish semantic completeness; manually review topics, constraints and report citations."}
    save(output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='details'},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","evaluate"))
    p.add_argument("--base-url",default="http://127.0.0.1:1486")
    p.add_argument("--output",type=lambda x:ROOT/x,default=ROOT/"evaluation_runs/intent_v2")
    args=p.parse_args()
    if args.action=="freeze": freeze()
    elif args.action=="run":
        if not args.base_url.startswith('http://127.0.0.1:'): p.error('Loopback only')
        run(args.base_url,args.output)
    else: evaluate(args.output)
