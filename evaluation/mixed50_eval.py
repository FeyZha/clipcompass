"""Mixed-library scenario evaluation; fixed labels, no post-result tuning."""

import argparse
import datetime as dt
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict

from evaluation.english_v1 import ROOT, read, save, digest, verify_frozen
from evaluation.mixed50_v1 import DATA, INDEX


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("Invalid evidence interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def span_scores(items, evidence):
    predicted, gold = defaultdict(list), defaultdict(list)
    for item in items:
        predicted[item["video_id"]].append((item["recommended_watch_start"], item["recommended_watch_end"]))
    for item in evidence:
        gold[item["video_id"]].append((item["start"], item["end"]))
    returned = sum(duration(spans) for spans in predicted.values())
    target = sum(duration(spans) for spans in gold.values())
    hit = 0
    for video, spans in predicted.items():
        hit += sum(max(0, min(b, d) - max(a, c)) for a, b in union(spans) for c, d in union(gold[video]))
    return {"returned_seconds": returned, "hit_seconds": hit, "gold_seconds": target,
            "time_precision": hit / returned if returned else None,
            "gold_coverage": hit / target if target else None}


def freeze():
    verify_frozen()
    cases = read(DATA / "case_specs.json")
    for case in cases:
        case["gold_evidence"] = []
        for evidence in case.pop("cue_evidence", []):
            cues = read(DATA / "normalized" / f"{evidence['video_id']}.json")["segments"]
            first, last = evidence["first"], evidence["last"]
            assert 0 <= first <= last < len(cues)
            case["gold_evidence"].append({"video_id": evidence["video_id"], "start": cues[first]["start"],
                                          "end": max(c["end"] for c in cues[first:last+1]),
                                          "first_cue": first, "last_cue": last})
        assert case["kind"] != "specific" or case["gold_evidence"]
    assert len({c["id"] for c in cases}) == len(cases)
    save(DATA / "cases.json", cases)
    save(DATA / "questions.json", [{k: c[k] for k in ("id", "question", "kind")} for c in cases])
    paths = [DATA / name for name in ("case_specs.json", "cases.json", "questions.json", "videos.json", "audit_topics.json")]
    paths += list((DATA / "normalized").glob("*.json")) + list(INDEX.glob("*"))
    paths += [ROOT / name for name in ("evaluation/mixed50_v1.py", "evaluation/mixed50_eval.py", "library/guided_pipeline.py",
              "library/precise_pipeline.py", "library/mvp_pipeline.py", "llm/answer_gate_bundle.py", "library/server.py")]
    save(DATA / "manifest.json", {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "scope": "Agent-authored mixed-library scenario test, NOT independent human blind validation",
         "topic_labels_used_for_retrieval": False, "hashes": {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}})
    print(f"Frozen {len(cases)} cases before predictions", flush=True)


def verify():
    verify_frozen()
    for name, expected in read(DATA / "manifest.json")["hashes"].items():
        assert digest(ROOT / name) == expected, name


def run(base_url, output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "config.json", {"base_url": base_url, "manifest_sha256": digest(DATA / "manifest.json")})
    for question in read(DATA / "questions.json"):
        started = time.perf_counter()
        row = dict(question)
        try:
            request = urllib.request.Request(base_url + "/api/search", json.dumps({"question": question["question"],
                "collection_id": "mixed50-v1"}).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=240) as response:
                row["response"], row["http_status"] = json.load(response), response.status
            row["error"] = None
        except urllib.error.HTTPError as exc:
            row.update(http_status=exc.code, response=json.load(exc), error=str(exc))
        except Exception as exc:
            row.update(error=str(exc))
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        save(output / f"{row['id']}.json", row)
        print(f"{row['id']}: {row.get('response', {}).get('answerable')} {row['error']}", flush=True)
    verify()
    evaluate(output)


def evaluate(output):
    verify()
    cases = read(DATA / "cases.json")
    topics = read(DATA / "audit_topics.json")
    details = []
    for case in cases:
        row = read(output / f"{case['id']}.json")
        response = row.get("response", {})
        items = response.get("results", [])
        detail = {"id": case["id"], "kind": case["kind"], "question": case["question"], "error": row["error"],
                  "answerable": response.get("answerable"), "latency_ms": row["latency_ms"], "results": items}
        if case["kind"] in ("specific", "absent"):
            detail["answerability_correct"] = not row["error"] and response.get("answerable") == (case["kind"] == "specific")
            detail.update(span_scores(items, case["gold_evidence"]))
        elif case["kind"] == "clarify":
            detail["clarification_correct"] = response.get("error") == "clarification_required"
        else:
            groups = response.get("learning_map", {}).get("groups", [])
            detail["report_sections"] = len(groups)
            detail["section_questions"] = [g["question"] for g in groups]
            detail["within_topic"] = bool(items) and all(topics[i["video_id"]] in case["allowed_topics"] for i in items)
            detail["report_contract_pass"] = not row["error"] and len(groups) >= 2 and detail["within_topic"]
        # Check provenance/bounds, independent of whether the answer was semantically right.
        detail["source_bounds_valid"] = True
        for item in items:
            cues = read(DATA / "normalized" / f"{item['video_id']}.json")["segments"]
            first, last = item["localization"]["start_cue_id"], item["localization"]["end_cue_id"]
            detail["source_bounds_valid"] &= (item["recommended_watch_start"] == item["playback_start"] == cues[first]["start"]
                 and item["recommended_watch_end"] == max(c["end"] for c in cues[first:last+1]))
        details.append(detail)
    specific = [d for d in details if d["kind"] == "specific"]
    absent = [d for d in details if d["kind"] == "absent"]
    reports = [d for d in details if d["kind"] == "broad"]
    scored = specific + absent
    returned = sum(d["returned_seconds"] for d in scored)
    hit = sum(d["hit_seconds"] for d in scored)
    gold = sum(d["gold_seconds"] for d in specific)
    summary = {"questions": len(cases), "specific_count": len(specific), "absent_count": len(absent),
               "specific_answerability_correct": sum(d["answerability_correct"] for d in specific),
               "absent_correct": sum(d["answerability_correct"] for d in absent),
               "returned_seconds": returned, "hit_seconds": hit, "time_precision": hit / returned if returned else None,
               "gold_coverage": hit / gold if gold else None, "report_count": len(reports),
               "report_contract_pass": sum(d["report_contract_pass"] for d in reports),
               "unexpected_errors": sum(bool(d["error"]) and not d.get("clarification_correct", False) for d in details),
               "mean_latency_ms": statistics.mean(d["latency_ms"] for d in details),
               "caveat": "Gold-interval overlap, not semantic completeness. Broad reports have scope/structure checks only, not accuracy labels.",
               "details": details}
    save(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "evaluate"))
    parser.add_argument("--base-url", default="http://127.0.0.1:1485")
    parser.add_argument("--output", type=lambda value: ROOT / value, default=ROOT / "evaluation_runs/mixed50_v1")
    args = parser.parse_args()
    if not args.base_url.startswith("http://127.0.0.1:"):
        parser.error("Only loopback evaluation is allowed")
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(args.base_url, args.output)
    else:
        evaluate(args.output)
