"""Reused-case regression, NOT a new blind test. Never overwrite historical runs."""

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request

from evaluation.english_v1 import DATA, ROOT, RUN, digest, read, save, verify_frozen


def run(base_url, output):
    verify_frozen()
    output.mkdir(parents=True, exist_ok=False)
    questions = read(DATA / "questions.json") + [
        {"id": "feedback", "question": "如何问同事报告写好没有"}]
    save(output / "config.json", {"scope": "Known-case regression; not blind evaluation", "base_url": base_url,
         "questions": questions, "hashes": {name: digest(ROOT / name) for name in
             ("library/precise_pipeline.py", "library/static/app.js", "mvp.py")}})
    predictions = []
    for question in questions:
        request = urllib.request.Request(base_url + "/api/search",
            json.dumps({"question": question["question"], "collection_id": "english-v1"}).encode(),
            {"Content-Type": "application/json"})
        started = time.perf_counter()
        row = dict(question)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                row["response"] = json.load(response)
            row["error"] = None
        except Exception as exc:
            row["error"] = str(exc)
        row["http_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        predictions.append(row)
        save(output / f"{row['id']}.json", row)
        print(f"{row['id']}: answerable={row.get('response', {}).get('answerable')}, error={row['error']!a}", flush=True)

    # Labels and old results are read only after every live prediction has been saved.
    cases = {c["id"]: c for c in read(DATA / "cases.json")}
    old = {c["id"]: c for c in [json.loads(line) for line in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()]}
    details = []
    for row in predictions:
        response = row.get("response", {})
        item = next(iter(response.get("results", [])), None)
        detail = {"id": row["id"], "question": row["question"], "error": row["error"], "result": item}
        if row["id"] in cases:
            gold = cases[row["id"]]
            detail["answerability_correct"] = not row["error"] and response.get("answerable") == gold["answerable"]
            if item:
                start, end = item["recommended_watch_start"], item["recommended_watch_end"]
                best = max((max(0, min(end, e["end"]) - max(start, e["start"])) for e in gold["gold_evidence"]
                            if e["video_id"] == item["video_id"]), default=0)
                detail["label_overlap_fraction_of_returned_span"] = best / (end - start)
                previous = old[row["id"]]["response"]["results"][0]
                detail["old_duration"] = previous["recommended_watch_end"] - previous["recommended_watch_start"]
                detail["new_duration"] = end - start
        if item:
            cues = read(DATA / "normalized" / f"{item['video_id']}.json")["segments"]
            first, last = item["localization"]["start_cue_id"], item["localization"]["end_cue_id"]
            assert item["recommended_watch_start"] == item["playback_start"] == cues[first]["start"]
            assert item["recommended_watch_end"] == max(c["end"] for c in cues[first:last + 1])
            assert item["context_start"] <= item["playback_start"] < item["recommended_watch_end"] <= item["context_end"]
            detail["selected_captions"] = cues[first:last + 1]
        details.append(detail)
    timed = [d for d in details if "new_duration" in d]
    summary = {"scope": "Regression on known English questions; requires sufficiency review, not accuracy proof",
               "questions": len(details), "errors": sum(bool(d["error"]) for d in details),
               "answerability_correct": sum(d.get("answerability_correct", False) for d in details),
               "mean_old_duration": statistics.mean(d["old_duration"] for d in timed) if timed else None,
               "mean_new_duration": statistics.mean(d["new_duration"] for d in timed) if timed else None,
               "mean_http_ms": statistics.mean(r["http_latency_ms"] for r in predictions), "cases": details}
    verify_frozen()
    save(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_runs/precise_span_v1")
    args = parser.parse_args()
    if not args.base_url.startswith("http://127.0.0.1:"):
        parser.error("Only the local test service is allowed")
    run(args.base_url, args.output)
