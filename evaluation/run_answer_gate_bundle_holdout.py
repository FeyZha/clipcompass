"""Run Answer Gate Bundle v2 exactly once on frozen holdout candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import run_answer_gate_v1 as v1
from evaluation.freeze_holdout_v1 import check_hashes, digest, hashes, read_json, rows, save_new
from llm.answer_gate import candidate_payload
from llm.answer_gate_bundle import BUNDLE_PROMPT, GATE_SCHEMA, attach_bundles, bundle_payload, gate_pass

RUN = ROOT / "evaluation_runs/answer_gate_v2_bundle/holdout_v1"
QUESTIONS = ROOT / "evaluation_data/holdout_v1/questions.jsonl"
PRIVATE_GOLD = ROOT / "evaluation_data/holdout_v1/private_ground_truth.jsonl"
MANIFEST = ROOT / "evaluation_data/holdout_v1/manifest.json"
SOURCE = ROOT / "evaluation_runs/holdout_v1/predictions.jsonl"
SOURCE_CONFIG = ROOT / "evaluation_runs/holdout_v1/config.json"
CHUNKS = ROOT / "evaluation_runs/window_v1/dense_window_m/chunks.jsonl"
PIPELINE = ROOT / "evaluation/pipeline_v1.json"


def source_files():
    return [ROOT / path for path in (
        "evaluation/run_answer_gate_bundle_holdout.py", "evaluation/run_answer_gate_v1.py",
        "llm/answer_gate.py", "llm/answer_gate_bundle.py", "llm/provider.py", "llm/deepseek.py",
        "evaluation/pipeline_v1.json", "evaluation_runs/holdout_v1/predictions.jsonl",
        "evaluation_runs/holdout_v1/config.json", "evaluation_data/holdout_v1/questions.jsonl",
        "evaluation_runs/window_v1/dense_window_m/chunks.jsonl",
        "evaluation_runs/window_v1/dense_window_m/embedding_cache.npz",
    )]


def gate_files():
    return [ROOT / path for path in (
        "evaluation/run_answer_gate_bundle_holdout.py", "evaluation/run_answer_gate_v1.py",
        "llm/answer_gate.py", "llm/answer_gate_bundle.py", "llm/provider.py", "llm/deepseek.py",
    )]


def validate_manifest(include_private=False):
    manifest = read_json(MANIFEST)
    assert manifest["frozen"] is True and manifest["cases"] == 24
    assert digest(QUESTIONS) == manifest["questions_sha256"]
    assert digest(PIPELINE) == manifest["pipeline_manifest_sha256"]
    if include_private:
        assert digest(PRIVATE_GOLD) == manifest["private_ground_truth_sha256"]
    source_config = read_json(SOURCE_CONFIG)
    assert source_config["status"] == "complete" and source_config["enabled_l_rescue"] is False
    assert digest(SOURCE) == source_config["predictions_sha256"]
    return manifest


def prepare():
    validate_manifest()
    v1.verify_parent()
    assert not RUN.exists(), "The final Holdout Gate run already exists"
    chunks = rows(CHUNKS)
    questions = {row["id"]: row["question"] for row in rows(QUESTIONS)}
    prepared = []
    for row in rows(SOURCE):
        assert row["test_id"] in questions and row["question"] == questions[row["test_id"]]
        cores = candidate_payload(row["results"])
        prepared.append({"test_id": row["test_id"], "original_question": row["question"],
                         "english_query": row["english_query"], "candidates": attach_bundles(cores, chunks)})
    assert len(prepared) == len(questions) == 24 and {r["test_id"] for r in prepared} == set(questions)
    assert all(len(row["candidates"]) == 20 for row in prepared)
    RUN.mkdir(parents=True)
    v1.write_new_rows(RUN / "inputs.jsonl", prepared)
    frozen_sources = hashes(source_files())
    save_new(RUN / "config.json", {
        "run_id": "answer_gate_bundle_v2_holdout_v1", "split": "holdout_v1", "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "questions": 24,
        "provider": "deepseek", "model": "deepseek-flash", "temperature": 0.0,
        "prompt": BUNDLE_PROMPT, "schema": GATE_SCHEMA, "pipeline_sha256": digest(PIPELINE),
        "inputs_sha256": digest(RUN / "inputs.jsonl"), "gate_source_hashes": hashes(gate_files()),
        "frozen_source_hashes": frozen_sources, "holdout_questions_sha256": digest(QUESTIONS),
        "candidate_source_sha256": digest(SOURCE), "bundle_rule": "fixed same-video previous/core/next",
        "selection_rule": "original core ID and times; bundle is recommended watch context",
        "cascade": [[1, 5], [6, 20]], "early_exit": "any sufficient in pass 1",
        "enabled_l_rescue": False, "ground_truth_visible_to_inference": False,
        "no_post_holdout_tuning": True, "pricing": v1.PRICING,
    })
    print(json.dumps({"prepared": 24, "gold_access": "prohibited", "candidate_source": "frozen"}))


def repair_preflight():
    config = read_json(RUN / "config.json")
    assert config["status"] == "prepared" and not (RUN / "calls.jsonl").exists()
    assert not (RUN / "results.jsonl").exists() and digest(RUN / "inputs.jsonl") == config["inputs_sha256"]
    config.update(gate_source_hashes=hashes(gate_files()), frozen_source_hashes=hashes(source_files()),
                  preflight_repair={"model_calls_before_repair": 0,
                      "reason": "Separated inference code hashes from frozen evaluation artifact hashes."})
    v1.write_json(RUN / "config.json", config)
    print(json.dumps(config["preflight_repair"]))


def run():
    validate_manifest()
    config = read_json(RUN / "config.json")
    check_hashes(config["frozen_source_hashes"])
    with patch.object(v1, "RUN", RUN.parent), patch("llm.answer_gate.gate_pass", gate_pass):
        v1.run(RUN.name)


def overlap(start, end, video_id, evidence):
    return any(video_id == gold["video_id"] and min(end, gold["end"]) > max(start, gold["start"])
               for gold in evidence)


def aggregate(gold, inputs, predictions, calls):
    source = {row["test_id"]: row for row in inputs}
    predicted = {row["test_id"]: row for row in predictions}
    assert len(gold) == len(source) == len(predicted)
    counts = Counter()
    for test in gold:
        row, result = source[test["id"]], predicted[test["id"]]
        assert row["original_question"] == test["question"] and not result["error"]
        chosen = next((c for c in row["candidates"] if c["candidate_id"] == result["best_candidate_id"]), None)
        assert bool(chosen) == result["answerable"]
        bundle_hit = bool(chosen and overlap(chosen["bundle_start"], chosen["bundle_end"],
                                             chosen["video_id"], test["gold_evidence"]))
        core_hit = bool(chosen and overlap(chosen["core_start"], chosen["core_end"],
                                           chosen["video_id"], test["gold_evidence"]))
        counts["answerable"] += test["answerable"]
        counts["no_answer"] += not test["answerable"]
        counts["positive"] += result["answerable"]
        counts["accepted_answerable"] += test["answerable"] and result["answerable"]
        counts["true_negative"] += not test["answerable"] and not result["answerable"]
        counts["precision_true"] += test["answerable"] and bundle_hit
        counts["selected_core_hit"] += test["answerable"] and core_hit
        counts["false_positive"] += not test["answerable"] and result["answerable"]
        counts["false_negative"] += test["answerable"] and not result["answerable"]
    ratio = lambda n, d: n / d if d else None
    metrics = {
        "answerable_recall": ratio(counts["accepted_answerable"], counts["answerable"]),
        "answerable_precision": ratio(counts["precision_true"], counts["positive"]),
        "no_answer_accuracy": ratio(counts["true_negative"], counts["no_answer"]),
        "evidence_selection_accuracy": ratio(counts["selected_core_hit"], counts["accepted_answerable"]),
    }
    latencies = sorted(row["gate_latency_ms"] for row in predictions)
    cost = sum(call["usage"]["estimated_cost_cny"] for call in calls)
    query_cost = read_json(SOURCE_CONFIG)["api_usage"]["estimated_cost_cny"]
    performance = {
        "gate_api_calls": len(calls), "gate_errors": sum(bool(c.get("error")) for c in calls),
        "pass1_early_exit": sum(row["pass_count"] == 1 for row in predictions),
        "pass2_calls": sum(row["pass_count"] == 2 for row in predictions),
        "average_gate_latency_ms": statistics.mean(latencies),
        "p95_gate_latency_ms": latencies[math.ceil(.95 * len(latencies)) - 1],
        "gate_cost_cny": cost, "query_understanding_cost_cny": query_cost,
        "total_deepseek_cost_cny": cost + query_cost,
        "average_deepseek_cost_cny_per_query": (cost + query_cost) / len(predictions),
        "input_tokens": sum(call["usage"]["input_tokens"] for call in calls),
        "output_tokens": sum(call["usage"]["output_tokens"] for call in calls),
    }
    thresholds = {"answerable_recall": .70, "answerable_precision": .85,
                  "no_answer_accuracy": .80, "evidence_selection_accuracy": .80}
    acceptance = {key: {"actual": metrics[key], "required": value, "met": metrics[key] >= value}
                  for key, value in thresholds.items()}
    acceptance["p95_gate_latency_ms"] = {"actual": performance["p95_gate_latency_ms"], "required_max": 6000,
                                          "met": performance["p95_gate_latency_ms"] <= 6000}
    acceptance["average_deepseek_cost_cny_per_query"] = {
        "actual": performance["average_deepseek_cost_cny_per_query"], "required_max": .02,
        "met": performance["average_deepseek_cost_cny_per_query"] <= .02}
    return {"questions": 24, "answerable": counts["answerable"], "no_answer": counts["no_answer"],
            "metrics": metrics, "counts": dict(counts), "performance": performance,
            "acceptance": acceptance, "all_acceptance_met": all(item["met"] for item in acceptance.values()),
            "metric_definitions": {
                "answerable_precision": "Predicted positives whose selected bundle overlaps frozen sufficient Gold evidence / all predicted positives; strict lower bound for unlisted alternatives.",
                "evidence_selection_accuracy": "Selected original core overlaps Gold / accepted answerable cases; bundle bounds never inflate this metric.",
                "cost": "Query Understanding cost from the frozen candidate run plus this Gate run; estimate, not invoice.",
            }}


def evaluate():
    manifest = validate_manifest(include_private=True)
    config = read_json(RUN / "config.json")
    assert config["status"] == "complete" and not (RUN / "summary.json").exists()
    check_hashes(config["frozen_source_hashes"])
    for key, name in (("inputs_sha256", "inputs.jsonl"), ("calls_sha256", "calls.jsonl"),
                      ("results_sha256", "results.jsonl")):
        assert digest(RUN / name) == config[key]
    report = aggregate(rows(PRIVATE_GOLD), rows(RUN / "inputs.jsonl"), rows(RUN / "results.jsonl"),
                       rows(RUN / "calls.jsonl"))
    report.update(run_id=config["run_id"], evaluated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  frozen_ground_truth_sha256=manifest["private_ground_truth_sha256"],
                  pipeline_unchanged=True, holdout_tuning=False, individual_cases_disclosed=False)
    save_new(RUN / "summary.json", report)
    with (RUN / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("# Answer Gate Bundle v2 — final Holdout\n\nAggregate only; no case details disclosed.\n\n```json\n")
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n")
    print(json.dumps(report, ensure_ascii=False))


def self_check():
    gold = [{"id": "a", "question": "q", "answerable": True,
             "gold_evidence": [{"video_id": "v", "start": 10, "end": 20}]},
            {"id": "n", "question": "n", "answerable": False, "gold_evidence": []}]
    candidate = {"candidate_id": "c", "video_id": "v", "start": 21, "end": 25,
                 "core_start": 21, "core_end": 25, "bundle_start": 9, "bundle_end": 30}
    inputs = [{"test_id": "a", "original_question": "q", "candidates": [candidate]},
              {"test_id": "n", "original_question": "n", "candidates": [candidate]}]
    predictions = [{"test_id": "a", "answerable": True, "best_candidate_id": "c", "error": None,
                    "pass_count": 1, "gate_latency_ms": 10},
                   {"test_id": "n", "answerable": False, "best_candidate_id": None, "error": None,
                    "pass_count": 2, "gate_latency_ms": 20}]
    calls = [{"usage": {"estimated_cost_cny": 0, "input_tokens": 1, "output_tokens": 1}, "error": None}]
    with patch.object(Path, "read_text", return_value=json.dumps({"api_usage": {"estimated_cost_cny": 0}})):
        report = aggregate(gold, inputs, predictions, calls)
    assert report["metrics"] == {"answerable_recall": 1, "answerable_precision": 1,
                                  "no_answer_accuracy": 1, "evidence_selection_accuracy": 0}
    print("holdout aggregate self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-check", "prepare", "repair-preflight", "run", "evaluate"))
    action = parser.parse_args().action
    {"self-check": self_check, "prepare": prepare, "repair-preflight": repair_preflight,
     "run": run, "evaluate": evaluate}[action]()
