"""Append a two-pass gate to frozen candidates; no retrieval or label access in inference."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.freeze_holdout_v1 import digest, read_json, rows, save_new, check_hashes, hashes

RUN = ROOT / "evaluation_runs/answer_gate_v1"
PIPELINE = ROOT / "evaluation/pipeline_v1.json"
FREEZE = RUN / "frozen_gate.json"
PRICING = {"source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
           "checked_date": "2026-09-26", "currency": "CNY", "unit": "per million tokens",
           "off_peak": {"cached": .02, "uncached": 1., "output": 4.},
           "peak": {"cached": .04, "uncached": 2., "output": 8.},
           "peak_hours": "Beijing Monday-Friday [09,12) and [14,18)",
           "note": "usage-based estimate, not an invoice"}
AUDIT_POLICY = (
    "Independent reviewer sees only original question, English query and the selected candidate, "
    "never Gate reasons, labels, Gold or adjacent text. A single candidate must cover all requested "
    "conditions and steps without outside knowledge. Assign sufficient/partial/mention/none/uncertain. "
    "All positive decisions must be reviewed, not just disagreements. Never feed review results back into Gate."
)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, obj):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_new_rows(path, data):
    with path.open("x", encoding="utf-8") as handle:
        for obj in data:
            handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def gate_sources():
    return hashes([ROOT / p for p in ("llm/answer_gate.py", "llm/provider.py", "llm/deepseek.py",
                  "evaluation/run_answer_gate_v1.py", "evaluation/test_answer_gate_v1.py",
                  "evaluation/test_answer_gate_eval.py")])


def verify_parent():
    pipeline = read_json(PIPELINE)
    assert pipeline["enabled_l_rescue"] is False
    check_hashes(pipeline["source_hashes"])
    check_hashes(pipeline["frozen_data_hashes"])


def price(usage, now):
    local = now.astimezone(dt.timezone(dt.timedelta(hours=8)))
    peak = local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)
    rates = PRICING["peak" if peak else "off_peak"]
    cached = int(usage.get("cached_input_tokens") or 0)
    inputs, outputs = int(usage["input_tokens"]), int(usage["output_tokens"])
    assert 0 <= cached <= inputs and outputs >= 0
    return {**usage, "estimated_cost_cny": (cached * rates["cached"] + (inputs - cached) * rates["uncached"]
                                              + outputs * rates["output"]) / 1e6,
            "pricing_period": "peak" if peak else "off_peak", "timestamp_utc": now.isoformat()}


def prepare(split):
    from llm.answer_gate import candidate_payload
    verify_parent()
    directory = RUN / split
    assert not directory.exists(), "Do not overwrite an existing split"
    if split == "holdout":
        frozen = read_json(FREEZE)
        check_hashes(frozen["gate_source_hashes"])
        assert frozen["dev_summary_sha256"] == digest(RUN / "dev/summary.json")
        manifest = read_json(ROOT / "evaluation_data/holdout_v1/manifest.json")
        previous = read_json(ROOT / "evaluation_runs/holdout_v1/config.json")
        assert manifest["pipeline_manifest_sha256"] == previous["pipeline_sha256"] == digest(PIPELINE)
        assert previous["status"] == "complete" and previous["questions_sha256"] == manifest["questions_sha256"]
        assert digest(ROOT / "evaluation_data/holdout_v1/questions.jsonl") == manifest["questions_sha256"]
        source = ROOT / "evaluation_runs/holdout_v1/predictions.jsonl"
        assert digest(source) == previous["predictions_sha256"]
        expected_count = manifest["cases"]
    else:
        source = ROOT / "evaluation_runs/reranker_v1/results.jsonl"
        expected_count = 40
    chunks = {c["chunk_id"]: c for c in rows(ROOT / "evaluation_runs/window_v1/dense_window_m/chunks.jsonl")}
    sanitized = []
    for row in rows(source):
        candidates = row["results"]
        assert len(candidates) == 20 and [c["rank"] for c in candidates] == list(range(1, 21))
        for c in candidates:
            assert all(c[k] == chunks[c["chunk_id"]][k] for k in ("video_id", "start", "end", "text"))
        sanitized.append({"test_id": row["test_id"], "original_question": row["question"],
                          "english_query": row["english_query"], "candidates": candidate_payload(candidates)})
    assert len(sanitized) == len({r["test_id"] for r in sanitized}) == expected_count
    directory.mkdir(parents=True)
    write_new_rows(directory / "inputs.jsonl", sanitized)
    save_new(directory / "config.json", {
        "split": split, "status": "prepared", "questions": expected_count,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "pipeline_sha256": digest(PIPELINE),
        "candidate_source_sha256": digest(source), "inputs_sha256": digest(directory / "inputs.jsonl"),
        "gate_source_hashes": gate_sources(), "cascade": [[1, 5], [6, 20]], "early_exit": "any sufficient in pass 1",
        "schema_error_retries": 1, "retry_policy": "same provider and identical input; explicit logging; API failure stops",
        "new_query_understanding_calls": 0, "retrieval_replayed_without_changes": True,
        "pricing": PRICING, "audit_policy": AUDIT_POLICY,
    })
    print(json.dumps({"prepared": split, "questions": expected_count, "gold_fields_in_inputs": False}))


def input_guard(root, directory):
    protected = [(root / name).resolve() for name in ("evaluation_data", "evaluation_runs")]
    allowed = {(directory / name).resolve() for name in
               ("config.json", "inputs.jsonl", "calls.jsonl", "results.jsonl", "error.json")}
    counts = {"allowed": 0, "denied": 0}
    def guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(folder) for folder in protected):
            if path not in allowed:
                counts["denied"] += 1
                raise PermissionError("Gate inference cannot read Ground Truth or other experiment data")
            counts["allowed"] += 1
    return guard, counts


def run(split):
    from llm.answer_gate import GateResponseError, gate_pass, validate_decision
    directory = RUN / split
    guard, access = input_guard(ROOT, directory)
    sys.addaudithook(guard)
    config = read_json(directory / "config.json")
    assert config["status"] != "complete", "Completed split cannot be rerun"
    check_hashes(config["gate_source_hashes"])
    assert digest(PIPELINE) == config["pipeline_sha256"]
    assert digest(directory / "inputs.jsonl") == config["inputs_sha256"]
    inputs = rows(directory / "inputs.jsonl")
    calls_path, results_path = directory / "calls.jsonl", directory / "results.jsonl"
    calls = rows(calls_path) if calls_path.exists() else []
    assert not any(c.get("error") and c.get("fatal", True) for c in calls), "Fatal error requires explicit review"
    cached = {(c["test_id"], c["pass"]): c for c in calls if not c.get("error")}
    assert len(cached) == sum(not c.get("error") for c in calls)
    completed = rows(results_path) if results_path.exists() else []
    done = {r["test_id"] for r in completed}
    assert len(done) == len(completed) and done <= {r["test_id"] for r in inputs}
    config["status"] = "running"
    write_json(directory / "config.json", config)
    for number, row in enumerate(inputs, 1):
        if row["test_id"] in done:
            continue
        used = []
        for pass_number, candidates in ((1, row["candidates"][:5]), (2, row["candidates"][5:])):
            key = (row["test_id"], pass_number)
            if key not in cached:
                previous_attempts = [c for c in calls if (c['test_id'], c['pass']) == key]
                spent_ms = sum(c['wall_latency_ms'] for c in previous_attempts)
                for attempt in range(len(previous_attempts) + 1, 3):
                    started = time.perf_counter()
                    try:
                        decision, usage = gate_pass(row["original_question"], row["english_query"], candidates)
                        usage = price(usage, dt.datetime.now(dt.timezone.utc))
                    except Exception as exc:
                        usage = getattr(exc, "usage", None)
                        if usage:
                            usage = price(usage, dt.datetime.now(dt.timezone.utc))
                        elapsed = (time.perf_counter() - started) * 1000
                        retry = isinstance(exc, GateResponseError) and attempt == 1
                        failed_call = {"test_id": row["test_id"], "pass": pass_number, "attempt": attempt,
                                       "error": type(exc).__name__, "usage": usage, "fatal": not retry,
                                       "validation_detail": str(exc) if isinstance(exc, GateResponseError) else None,
                                       "invalid_decision": getattr(exc, "decision", None), "wall_latency_ms": elapsed}
                        append(calls_path, failed_call)
                        calls.append(failed_call)
                        spent_ms += elapsed
                        print("Gate structured-output validation failed; recorded error and cost. " +
                              ("Retrying once with identical model and input." if retry else "Stopped without fallback."),
                              file=sys.stderr, flush=True)
                        if retry:
                            continue
                        write_json(directory / "error.json", {"error_type": type(exc).__name__, "completed": len(done),
                                   "policy": "stop; no fallback; error is not no-answer"})
                        raise RuntimeError("Gate API/schema failure; stopped explicitly") from None
                    break
                else:
                    raise RuntimeError("Structured-output attempt budget exhausted")
                elapsed = (time.perf_counter() - started) * 1000
                call = {"test_id": row["test_id"], "pass": pass_number,
                        "attempt": attempt,
                        "candidate_ids": [c["candidate_id"] for c in candidates], "decision": decision,
                        "usage": usage, "wall_latency_ms": elapsed, "pass_total_latency_ms": spent_ms + elapsed, "error": None}
                append(calls_path, call)
                calls.append(call)
                cached[key] = call
            call = cached[key]
            validate_decision(call["decision"], candidates)
            assert call["candidate_ids"] == [c["candidate_id"] for c in candidates]
            used.append(call)
            if call["decision"]["answerable"]:
                break
        decision = used[-1]["decision"]
        result = {"test_id": row["test_id"], "answerable": decision["answerable"],
                  "best_candidate_id": decision["best_candidate_id"], "support": decision["support"],
                  "pass_count": len(used), "gate_latency_ms": sum(c["pass_total_latency_ms"] for c in used),
                  "estimated_cost_cny": sum(c["usage"]["estimated_cost_cny"] for c in calls if c['test_id'] == row['test_id']), "error": None}
        append(results_path, result)
        done.add(row["test_id"])
        print(f"Gate completed {number}/{len(inputs)}; individual outcomes hidden.", flush=True)
    assert len(done) == config["questions"]
    check_hashes(config["gate_source_hashes"])
    assert digest(directory / "inputs.jsonl") == config["inputs_sha256"]
    config.update(status="complete", calls_sha256=digest(calls_path), results_sha256=digest(results_path),
                  access_guard=access, completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(directory / "config.json", config)
    print(json.dumps({"completed": split, "questions": len(done), "api_calls": len(calls), "gold_access": "prohibited"}))


def validate_completed(split):
    directory = RUN / split
    config = read_json(directory / "config.json")
    assert config["status"] == "complete"
    check_hashes(config["gate_source_hashes"])
    for key, file in (("inputs_sha256", "inputs.jsonl"), ("calls_sha256", "calls.jsonl"), ("results_sha256", "results.jsonl")):
        assert digest(directory / file) == config[key]
    return directory, config, rows(directory / "inputs.jsonl"), rows(directory / "results.jsonl"), rows(directory / "calls.jsonl")


def export_audit(split):
    directory, config, inputs, predictions, _ = validate_completed(split)
    source = {r["test_id"]: r for r in inputs}
    samples = []
    for row in predictions:
        if row["answerable"]:
            data = source[row["test_id"]]
            chosen = next(c for c in data["candidates"] if c["candidate_id"] == row["best_candidate_id"])
            samples.append({"test_id": row["test_id"], "original_question": data["original_question"],
                            "english_query": data["english_query"], "selected_candidate": chosen})
    write_new_rows(directory / "audit_inputs.jsonl", samples)
    print(json.dumps({"audit_exported": split, "positive_decisions_to_review": len(samples), "gate_reasons_and_gold_hidden": True}))


def temporal_hit(candidate, evidence):
    return any(candidate["video_id"] == gold["video_id"] and
               min(candidate["end"], gold["end"]) > max(candidate["start"], gold["start"]) for gold in evidence)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(gold, inputs, predictions, audit, calls):
    source = {r["test_id"]: r for r in inputs}
    predicted = {r["test_id"]: r for r in predictions}
    judgments = {r["test_id"]: r for r in audit}
    assert len(source) == len(predicted) == len(gold) and set(source) == set(predicted) == {t["id"] for t in gold}
    positives = {i for i, r in predicted.items() if r["answerable"]}
    assert set(judgments) == positives and len(judgments) == len(audit), "Review every positive exactly once"
    assert all(j["support"] in {"sufficient", "partial", "mention", "none", "uncertain"} for j in audit)
    counts = Counter()
    failures = Counter({label: 0 for label in ("GATE_FALSE_POSITIVE", "GATE_FALSE_NEGATIVE", "GATE_WRONG_EVIDENCE", "RETRIEVAL_CEILING")})
    for test in gold:
        row, data = predicted[test["id"]], source[test["id"]]
        assert data["original_question"] == test["question"] and not row["error"]
        available = bool(test["answerable"] and any(temporal_hit(c, test["gold_evidence"]) for c in data["candidates"]))
        selected = next((c for c in data["candidates"] if c["candidate_id"] == row["best_candidate_id"]), None)
        assert bool(selected) == row["answerable"]
        hit = bool(selected and temporal_hit(selected, test["gold_evidence"]))
        counts["answerable"] += test["answerable"]
        counts["no_answer"] += not test["answerable"]
        counts["predicted_positive"] += row["answerable"]
        counts["true_positive_answerability"] += test["answerable"] and row["answerable"]
        counts["true_negative"] += not test["answerable"] and not row["answerable"]
        counts["gold_in_pool"] += available
        counts["conditional_true"] += available and row["answerable"]
        counts["selected_gold_hit"] += test["answerable"] and hit
        counts["ranking_repaired_by_gate"] += hit and not temporal_hit(data["candidates"][0], test["gold_evidence"])
        counts["gold_available_rejected"] += available and not row["answerable"]
        if row["answerable"]:
            support = judgments[test["id"]]["support"]
            counts["independently_sufficient"] += test["answerable"] and support == "sufficient"
            counts["uncertain_on_answerable"] += test["answerable"] and support == "uncertain"
            counts["positive_evidence_insufficient"] += support in {"partial", "mention", "none"}
            counts["gold_vs_audit_conflict"] += not test["answerable"] and support == "sufficient"
            counts["positive_evidence_uncertain"] += support == "uncertain"
        # A retrieval ceiling is not assigned a Gate FN/wrong-evidence label.
        failures["RETRIEVAL_CEILING"] += test["answerable"] and not available
        failures["GATE_FALSE_POSITIVE"] += not test["answerable"] and row["answerable"]
        failures["GATE_FALSE_NEGATIVE"] += available and not row["answerable"]
        failures["GATE_WRONG_EVIDENCE"] += available and row["answerable"] and (
            not hit or judgments[test['id']]['support'] in {'partial', 'mention', 'none'})
    metrics = {
        "answerable_recall": ratio(counts["true_positive_answerability"], counts["answerable"]),
        "no_answer_accuracy": ratio(counts["true_negative"], counts["no_answer"]),
        "answerable_precision_independent_audit": ratio(counts["independently_sufficient"], counts["predicted_positive"]),
        "answerability_label_precision": ratio(counts["true_positive_answerability"], counts["predicted_positive"]),
        "gold_window_supported_precision": ratio(counts["selected_gold_hit"], counts["predicted_positive"]),
        "conditional_gate_recall": ratio(counts["conditional_true"], counts["gold_in_pool"]),
        "evidence_selection_accuracy": ratio(counts["selected_gold_hit"], counts["true_positive_answerability"]),
        "evidence_selection_end_to_end": ratio(counts["selected_gold_hit"], counts["answerable"]),
    }
    latencies = sorted(r["gate_latency_ms"] for r in predictions)
    performance = {"average_gate_latency_ms": statistics.mean(latencies),
                   "p95_gate_latency_ms": latencies[math.ceil(.95 * len(latencies)) - 1],
                   "pass1_early_exit_questions": sum(r["pass_count"] == 1 for r in predictions),
                   "api_total_estimated_cost_cny": sum(c["usage"]["estimated_cost_cny"] for c in calls),
                   "api_calls": len(calls), "api_errors": sum(bool(c.get("error")) for c in calls)}
    performance["retry_calls"] = len(calls) - sum(r['pass_count'] for r in predictions)
    performance["mean_incremental_cost_cny"] = performance["api_total_estimated_cost_cny"] / len(predictions)
    for number in (1, 2):
        group = [c for c in calls if c["pass"] == number]
        performance[f"pass{number}"] = {"calls": len(group), **{
            f"mean_{key}": statistics.mean(c["usage"][key] for c in group) if group else 0
            for key in ("input_tokens", "output_tokens")}}
    return {"questions": len(gold), "metrics": metrics, "counts": dict(counts), "failures": dict(failures),
            "performance": performance, "audit_support_counts": dict(Counter(j["support"] for j in audit)),
            "precision_audit_uncertainty_bounds": [metrics["answerable_precision_independent_audit"],
                ratio(counts["independently_sufficient"] + counts["uncertain_on_answerable"], counts["predicted_positive"])],
            "metric_notes": {"answerable_precision": "independent AI review of selected text, constrained by frozen answerability; not human-validated truth",
                "conditional_gate_recall": "conditioned on positive Gold-window overlap in Top20, not independently sufficient-candidate coverage",
                "evidence_selection_accuracy": "selected Gold-window hits / answerable questions for which Gate returned true",
                "time_rule": "unchanged same-video positive temporal overlap; this is not semantic sufficiency",
                "wrong_evidence": "Gold window available, but selected window misses Gold or independent audit finds it insufficient",
                "gate_false_negative": "Gold-window-present rejection proxy; not proof the available candidate itself was fully sufficient",
                "api_cost": "Gate only; cached Query Understanding and candidates; usage-based CNY estimate",
                "latency": "sum of live pass durations per question; P95 nearest-rank; no full retrieval replay latency",
                "independent_review": AUDIT_POLICY}}


def evaluate(split):
    verify_parent()
    directory, config, inputs, predictions, calls = validate_completed(split)
    assert not (directory / "summary.json").exists(), "No overwrite or repeat evaluation"
    if split == "holdout":
        frozen = read_json(FREEZE)
        check_hashes(frozen["gate_source_hashes"])
        manifest = read_json(ROOT / "evaluation_data/holdout_v1/manifest.json")
        path = ROOT / "evaluation_data/holdout_v1/private_ground_truth.jsonl"
        assert digest(path) == manifest["private_ground_truth_sha256"]
    else:
        path = ROOT / "evaluation_data/dev_v1/test_cases.jsonl"
        assert digest(path) == read_json(ROOT / "evaluation_data/dev_v1/manifest.json")["sha256"]
    report = summarize(rows(path), inputs, predictions, rows(directory / "sufficiency_audit.jsonl"), calls)
    if split == 'dev':
        old_calls = []
        for previous in sorted(RUN.glob('dev_attempt_*')):
            old_calls.extend(rows(previous / 'calls.jsonl'))
            if (previous / 'diagnostic_call.json').exists():
                old_calls.append(read_json(previous / 'diagnostic_call.json'))
        if old_calls:
            old_cost = sum(c['usage']['estimated_cost_cny'] for c in old_calls)
            report['performance']['prior_development_attempts'] = {'api_calls':len(old_calls), 'estimated_cost_cny':old_cost,
                'note':'prior incomplete development runs and explicit format diagnostics; all known usage included'}
            report['performance']['all_development_api_cost_cny'] = old_cost + report['performance']['api_total_estimated_cost_cny']
    report.update(split=split, checks={"pipeline_v1_unchanged": True, "ground_truth_unchanged": True,
        "private_gold_not_read_by_gate": True, "only_aggregate_output": True},
        audit_sha256=digest(directory / "sufficiency_audit.jsonl"))
    save_new(directory / "summary.json", report)
    with (directory / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write(f"# Answer Gate v1 / {split}\n\nAggregate only. No case identities or evidence disclosed.\n\n```json\n")
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n")
    print(json.dumps(report, ensure_ascii=False))


def freeze():
    from llm.answer_gate import GATE_PROMPT, GATE_SCHEMA
    verify_parent()
    summary = read_json(RUN / "dev/summary.json")
    assert summary["questions"] == 40 and read_json(RUN / 'dev/config.json')['status'] == 'complete'
    assert summary["counts"]["gold_in_pool"] == 30
    save_new(FREEZE, {"id": "answer_gate_v1", "frozen": True,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "provider": "deepseek", "model": "deepseek-flash",
        "pipeline_sha256": digest(PIPELINE), "gate_source_hashes": gate_sources(),
        "prompt": GATE_PROMPT, "schema": GATE_SCHEMA, "temperature": 0.0,
        "cascade": [[1, 5], [6, 20]], "early_exit": "pass1 contains sufficient", "threshold": None,
        "schema_error_retry": "one identical model/input retry, explicitly logged and billed; API failure stops without fallback",
        "dev_summary_sha256": digest(RUN / "dev/summary.json"),
        "blind_test_policy": "exactly one holdout run; aggregate outputs only; no post-holdout prompt/code changes",
        "audit_policy": AUDIT_POLICY, "pricing": PRICING})
    print("Gate prompt, schema, cascade and evaluator frozen; Holdout not yet evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit-export", "evaluate", "freeze"))
    parser.add_argument("split", nargs="?", choices=("dev", "holdout"))
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            freeze()
        else:
            assert args.split
            {"prepare": prepare, "run": run, "audit-export": export_audit, "evaluate": evaluate}[args.action](args.split)
    except Exception as exc:
        print(f"{args.action} stopped: {type(exc).__name__}. No fallback or automatic data/prompt changes.", file=sys.stderr)
        raise SystemExit(1) from None
