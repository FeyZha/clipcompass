"""One frozen blind run; invoke `evaluate` in a separate process for aggregates only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from evaluation.run_dense_v1 import encode, load_jsonl, price_usage, sha256, write_json
from evaluation.run_reranker_v1 import reorder, score_pairs
from evaluation.run_window_v1 import match_detail

HOLDOUT = ROOT / "evaluation_data/holdout_v1"
OUTPUT = ROOT / "evaluation_runs/holdout_v1"
PIPELINE = ROOT / "evaluation/pipeline_v1.json"
WINDOW = ROOT / "evaluation_runs/window_v1/dense_window_m"
METRICS = ("candidate_recall", "recall_at_5", "recall_at_10", "recall_at_20", "top_1",
           "mrr_at_20", "timestamp_hit", "video_recall_at_5")


def make_read_guard(root, questions, output):
    """Deny data/result leakage in the inference process, including accidental helper reads.

    ponytail: a Python audit guard prevents accidental leakage, not hostile native-code access;
    use an OS-isolated worker if the runner ever executes untrusted code.
    """
    data, runs = (root / "evaluation_data").resolve(), (root / "evaluation_runs").resolve()
    allowed = {questions.resolve()}
    allowed.update((root / "evaluation_runs/window_v1/dense_window_m" / name).resolve()
                   for name in ("chunks.jsonl", "embedding_cache.npz", "config.json"))
    allowed.update((output / name).resolve() for name in
                   ("config.json", "query_cache.jsonl", "predictions.jsonl", "error.json"))
    counts = {"protected_opens_allowed": 0, "protected_opens_denied": 0}

    def guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(data) or path.is_relative_to(runs):
            if path not in allowed:
                counts["protected_opens_denied"] += 1
                raise PermissionError("Blind runner attempted to access prohibited evaluation data")
            counts["protected_opens_allowed"] += 1

    return guard, counts


def validate_questions(rows):
    if not rows or any(set(row) != {"id", "question"} for row in rows):
        raise ValueError("Questions must contain exactly id and question, with no labels or evidence")
    if any(not isinstance(row[key], str) or not row[key].strip()
           for row in rows for key in ("id", "question")):
        raise ValueError("Questions contain empty or invalid fields")
    if len({row["id"] for row in rows}) != len(rows) or len({row["question"] for row in rows}) != len(rows):
        raise ValueError("Questions contain duplicate IDs or question text")


def verify_hashes(expected):
    if not expected or any(sha256(ROOT / path) != digest for path, digest in expected.items()):
        raise ValueError("Frozen source/data hash mismatch; refusing to run or score")


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_predictions(rows, questions, chunks):
    by_id = {row["test_id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(by_id).issubset({q["id"] for q in questions}):
        raise ValueError("Duplicate or unknown prediction IDs")
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    for question in questions:
        if question["id"] not in by_id:
            continue
        row = by_id[question["id"]]
        if row["question"] != question["question"] or row.get("error"):
            raise ValueError("Prediction input mismatch or failed inference")
        dense, ranked = row["dense_results"], row["results"]
        if len(dense) != 20 or len(ranked) != 20:
            raise ValueError("The frozen pipeline must return exactly 20 candidates")
        for items in (dense, ranked):
            if len({item["chunk_id"] for item in items}) != 20:
                raise ValueError("Duplicate candidate chunks")
            if [item["rank"] for item in items] != list(range(1, 21)):
                raise ValueError("Invalid result ranks")
            for item in items:
                chunk = chunk_map[item["chunk_id"]]
                if any(item[key] != chunk[key] for key in ("text", "video_id", "start", "end", "segment_ids")):
                    raise ValueError("Candidate does not match the frozen Window M index")
        if {c["chunk_id"] for c in dense} != {c["chunk_id"] for c in ranked}:
            raise ValueError("Reranking changed the candidate set")
        expected = reorder(dense, [next(c["reranker_score"] for c in ranked
                                       if c["chunk_id"] == item["chunk_id"]) for item in dense])
        if ranked != expected:
            raise ValueError("Results do not preserve the frozen reranker-only stable sort")
        if row.get("system_answerable") is not None or row.get("explicit_no_answer") is not False:
            raise ValueError("Ranking-only pipeline must not introduce an Answer Gate")
    return by_id


def run(questions_sha256):
    questions_path = HOLDOUT / "questions.jsonl"
    guard, access_counts = make_read_guard(ROOT, questions_path, OUTPUT)
    sys.addaudithook(guard)
    pipeline_hash = sha256(PIPELINE)
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    if pipeline["enabled_l_rescue"] is not False:
        raise ValueError("Only the frozen pipeline_v1 without L Rescue is permitted")
    verify_hashes(pipeline["source_hashes"])
    if sha256(questions_path) != questions_sha256:
        raise ValueError("Question file does not match its pre-run frozen checksum")
    questions = load_jsonl(questions_path)
    validate_questions(questions)
    if len(questions) < 24:
        raise ValueError("Holdout must contain at least 24 questions")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT / "config.json"
    identity = {"pipeline_sha256": pipeline_hash, "questions_sha256": questions_sha256,
                "runner_sha256": sha256(Path(__file__))}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if any(config.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume would change the frozen run identity")
        if config.get("status") == "complete":
            raise ValueError("Completed holdout runs cannot be overwritten or rerun")
    else:
        if any(OUTPUT.iterdir()):
            raise ValueError("Refusing to reuse an output directory without its run configuration")
        config = {**identity, "run_id": "holdout_v1", "pipeline": "pipeline_v1",
                  "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running",
                  "questions": len(questions), "enabled_l_rescue": False,
                  "source_hashes": pipeline["source_hashes"], "candidate_k": 20,
                  "match_rule": "same video, positive temporal overlap (unchanged)",
                  "no_answer": "ranking-only; returns candidates; no Answer Gate decision",
                  "cost_estimate": "historical price_usage estimate; not a current billing quote"}
        write_json(config_path, config)

    from library.server import understand_query
    from llm.provider import load_project_env
    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    load_project_env()
    if os.getenv("LLM_PROVIDER", "deepseek").strip().lower() != "deepseek" or os.getenv(
            "LLM_MODEL", "deepseek-flash").strip() != "deepseek-flash":
        raise ValueError("The frozen Query Understanding provider/model must not change")
    chunks = load_jsonl(WINDOW / "chunks.jsonl")
    with np.load(WINDOW / "embedding_cache.npz", allow_pickle=False) as cache:
        if str(cache["model_id"]) != "BAAI/bge-small-en-v1.5":
            raise ValueError("Frozen index model mismatch")
        vectors = cache["embeddings"]
    if vectors.shape != (len(chunks), 384) or not np.all(np.isfinite(vectors)):
        raise ValueError("Invalid frozen embedding index")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Cosine retrieval requires the frozen normalized index")
    predictions_path, cache_path = OUTPUT / "predictions.jsonl", OUTPUT / "query_cache.jsonl"
    rows = load_jsonl(predictions_path) if predictions_path.exists() else []
    completed = validate_predictions(rows, questions, chunks)
    cached_rows = load_jsonl(cache_path) if cache_path.exists() else []
    cached = {row["test_id"]: row for row in cached_rows}
    if len(cached) != len(cached_rows) or not set(cached).issubset({q["id"] for q in questions}):
        raise ValueError("Query cache contains duplicate or unknown inputs")
    for question in questions:
        if question["id"] in cached and cached[question["id"]]["question"] != question["question"]:
            raise ValueError("Query cache input mismatch")
    if not set(completed).issubset(cached):
        raise ValueError("Completed predictions have no matching Query Understanding provenance")

    started = time.perf_counter()
    dense_path, reranker_path = ROOT / "models/bge-small-en-v1.5", ROOT / "models/ms-marco-MiniLM-L6-v2"
    dense_tokenizer = AutoTokenizer.from_pretrained(dense_path, local_files_only=True)
    dense_model = AutoModel.from_pretrained(dense_path, local_files_only=True).eval()
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_path, local_files_only=True).eval()
    config.update(model_load_ms=(time.perf_counter() - started) * 1000,
                  versions={"numpy": np.__version__, "torch": torch.__version__,
                            "transformers": __import__("transformers").__version__},
                  torch_cpu_threads=torch.get_num_threads())
    write_json(config_path, config)
    try:
        for number, question in enumerate(questions, 1):
            test_id = question["id"]
            if test_id in completed:
                continue
            if test_id not in cached:
                query, usage = understand_query(question["question"])
                usage = price_usage(usage, dt.datetime.now(dt.timezone.utc))
                cached[test_id] = {"test_id": test_id, "question": question["question"],
                                   "query_understanding": query, "usage": usage}
                append_row(cache_path, cached[test_id])
            query, usage = cached[test_id]["query_understanding"], cached[test_id]["usage"]
            english = query["english_query"]
            started = time.perf_counter()
            query_vector = encode(dense_model, dense_tokenizer, [english], query=True, batch_size=1)[0]
            embedding_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            scores = vectors @ query_vector
            order = np.argsort(-scores, kind="stable")[:20]
            candidates = [{**chunks[int(index)], "rank": rank, "score": round(float(scores[index]), 8)}
                          for rank, index in enumerate(order, 1)]
            retrieval_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            pair_scores = score_pairs(reranker, reranker_tokenizer, english, [c["text"] for c in candidates])
            results = reorder(candidates, pair_scores)
            rerank_ms = (time.perf_counter() - started) * 1000
            row = {"test_id": test_id, "question": question["question"], "english_query": english,
                   "query_understanding": query, "llm_usage": usage,
                   "dense_results": candidates, "results": results,
                   "system_answerable": None, "explicit_no_answer": False,
                   "embedding_latency_ms": embedding_ms, "retrieval_latency_ms": retrieval_ms,
                   "reranker_latency_ms": rerank_ms,
                   "local_latency_ms": embedding_ms + retrieval_ms + rerank_ms,
                   "total_latency_ms": usage["latency_ms"] + embedding_ms + retrieval_ms + rerank_ms,
                   "error": None}
            validate_predictions([row], [question], chunks)
            append_row(predictions_path, row)
            rows.append(row)
            print(f"Completed {number}/{len(questions)} questions; individual results remain sealed.", flush=True)
    except Exception as exc:
        # Do not echo API response bodies, keys, question IDs, or individual failures to stdout.
        write_json(OUTPUT / "error.json", {"error_type": type(exc).__name__, "completed": len(rows),
                   "action": "Stopped without fallback, algorithm changes, or deleting cases"})
        print(f"Blind run stopped: {type(exc).__name__}; no fallback was used.", file=sys.stderr)
        raise SystemExit(1) from None
    verify_hashes(pipeline["source_hashes"])
    if sha256(PIPELINE) != pipeline_hash or sha256(questions_path) != questions_sha256:
        raise ValueError("Frozen pipeline or questions changed during inference")
    validate_predictions(rows, questions, chunks)
    if len(rows) != len(questions):
        raise ValueError("The blind run is incomplete")
    config.update(status="complete", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  predictions_sha256=sha256(predictions_path), query_cache_sha256=sha256(cache_path),
                  completed=len(rows), access_guard=access_counts, frozen_source_hashes_unchanged=True,
                  api_usage={"calls": len(cached), "input_tokens": sum(r["usage"]["input_tokens"] for r in cached.values()),
                             "output_tokens": sum(r["usage"]["output_tokens"] for r in cached.values()),
                             "estimated_cost_cny": sum(r["usage"]["estimated_cost_cny"] for r in cached.values())})
    write_json(config_path, config)
    print(json.dumps({"status": "complete", "questions": len(rows), "gold_access": "prohibited",
                      "new_algorithm_changes": 0}, ensure_ascii=False), flush=True)


def aggregate(gold, predictions, dev_metrics):
    """Only this evaluator consumes labels; return no IDs, questions, evidence, or ranks."""
    by_id = {row["test_id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {test["id"] for test in gold}:
        raise ValueError("Evaluator inputs do not cover exactly the same questions")
    ranks, candidate_hits, video_hits = [], [], []
    failures = Counter({label: 0 for label in ("RETRIEVAL_MISS", "RANKING_ERROR", "TIMESTAMP_ERROR",
                                              "MULTI_VIDEO_CONFUSION", "NO_ANSWER_NOT_REJECTED")})
    no_answer, no_answer_rejected, failed = 0, 0, 0
    for test in gold:
        row = by_id[test["id"]]
        if row["question"] != test["question"]:
            raise ValueError("Evaluator question text mismatch")
        evidence, results = test["gold_evidence"], row["results"]
        if bool(evidence) != bool(test["answerable"]):
            raise ValueError("Invalid answerability/evidence contract")
        if not test["answerable"]:
            no_answer += 1
            no_answer_rejected += bool(row["explicit_no_answer"])
            failures["NO_ANSWER_NOT_REJECTED"] += not row["explicit_no_answer"]
            failed += not row["explicit_no_answer"]
            continue
        rank = next((index for index, item in enumerate(results, 1)
                     if match_detail(item, evidence)["is_gold"]), None)
        candidate_hit = any(match_detail(item, evidence)["is_gold"] for item in row["dense_results"])
        if candidate_hit != (rank is not None):
            raise ValueError("Candidate Recall changed during reranking")
        ranks.append(rank)
        candidate_hits.append(candidate_hit)
        videos = {e["video_id"] for e in evidence}
        video_hits.append(any(item["video_id"] in videos for item in results[:5]))
        failed += rank != 1
        failures["RETRIEVAL_MISS"] += rank is None
        failures["RANKING_ERROR"] += rank is not None and rank != 1
        if rank != 1 and results:
            failures["TIMESTAMP_ERROR"] += results[0]["video_id"] in videos
            failures["MULTI_VIDEO_CONFUSION"] += test["video_relevance"].get(results[0]["video_id"]) in {"partial", "mention"}
    count = len(ranks)
    if not count:
        raise ValueError("No answerable cases in evaluator input")
    metrics = {"candidate_recall": sum(candidate_hits) / count,
               **{f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / count for k in (5, 10, 20)},
               "top_1": sum(rank == 1 for rank in ranks) / count,
               "mrr_at_20": sum(1 / rank if rank else 0 for rank in ranks) / count,
               "timestamp_hit": sum(rank == 1 for rank in ranks) / count,
               "video_recall_at_5": sum(video_hits) / count}
    return {"total_cases": len(gold), "answerable_cases": count,
            "type_distribution": dict(Counter(test["type"] for test in gold)),
            "metrics": metrics, "dev_metrics": {key: dev_metrics[key] for key in METRICS},
            "delta_vs_dev": {key: metrics[key] - dev_metrics[key] for key in METRICS},
            "no_answer_behavior": {"cases": no_answer, "explicit_no_answer": no_answer_rejected,
                                   "returned_candidates": no_answer - no_answer_rejected,
                                   "gate_evaluated": False, "semantic_no_answer_accuracy": None},
            "failure_count": failed, "overall_failed_cases": failed,
            "answerable_top1_failures": sum(rank != 1 for rank in ranks),
            "failure_category_counts": dict(failures), "failure_categories_may_overlap": True,
            "not_assessed": ["SEGMENTATION_ERROR", "ANSWER_GATE_FALSE_NEGATIVE", "SEMANTIC_ANSWER_GATE"],
            "limits": ["Same four videos; this is new-question/region holdout, not unseen-video generalization.",
                       "Timestamp Hit retains the frozen positive-overlap rule, not semantic-sufficiency grading.",
                       "No Answer Gate was introduced or evaluated; non-rejection is recorded as behavior only."]}


def evaluate():
    if (OUTPUT / "summary.json").exists() or (OUTPUT / "summary.md").exists():
        raise ValueError("Refusing to overwrite a completed aggregate holdout evaluation")
    manifest = json.loads((HOLDOUT / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    if manifest.get("frozen") is not True or config.get("status") != "complete":
        raise ValueError("Only frozen questions and completed runs may be evaluated")
    if manifest["pipeline_manifest_sha256"] != sha256(PIPELINE):
        raise ValueError("Pipeline changed since holdout sealing")
    if manifest["verification_report_sha256"] != sha256(HOLDOUT / "verification_report.json"):
        raise ValueError("Verification record changed since holdout sealing")
    verify_hashes(manifest["evaluation_code_sha256"])
    for key, path in (("questions_sha256", HOLDOUT / "questions.jsonl"),
                      ("private_ground_truth_sha256", HOLDOUT / "private_ground_truth.jsonl")):
        if sha256(path) != manifest[key]:
            raise ValueError("Frozen holdout data mismatch")
    if config["questions_sha256"] != manifest["questions_sha256"] or config["pipeline_sha256"] != sha256(PIPELINE):
        raise ValueError("The run identity does not match the frozen evaluator inputs")
    if config["runner_sha256"] != sha256(Path(__file__)) or config["predictions_sha256"] != sha256(OUTPUT / "predictions.jsonl"):
        raise ValueError("Runner or predictions changed since the blind run")
    if config["query_cache_sha256"] != sha256(OUTPUT / "query_cache.jsonl"):
        raise ValueError("Query provenance changed since the blind run")
    verify_hashes(pipeline["source_hashes"])
    if pipeline.get("frozen_data_hashes"):
        verify_hashes(pipeline["frozen_data_hashes"])
    questions = load_jsonl(HOLDOUT / "questions.jsonl")
    validate_questions(questions)
    gold = load_jsonl(HOLDOUT / "private_ground_truth.jsonl")
    if [{"id": row["id"], "question": row["question"]} for row in gold] != questions:
        raise ValueError("Public questions and private labels are not aligned")
    rows = load_jsonl(OUTPUT / "predictions.jsonl")
    validate_predictions(rows, questions, load_jsonl(WINDOW / "chunks.jsonl"))
    report = aggregate(gold, rows, pipeline["dev_metrics"])
    report.update(checks={"source_hashes_unchanged": True, "frozen_data_hashes_unchanged": True,
                          "questions_ground_truth_aligned": True, "all_candidate_sets_unchanged": True,
                          "private_data_guard_denied_opens": config["access_guard"]["protected_opens_denied"],
                          "private_ground_truth_not_read_by_runner": True,
                          "only_aggregate_metrics_published": True},
                  api_usage=config["api_usage"],
                  artifact_hashes={"questions": manifest["questions_sha256"],
                                   "private_ground_truth": manifest["private_ground_truth_sha256"],
                                   "predictions": config["predictions_sha256"],
                                   "pipeline": config["pipeline_sha256"]})
    write_json(OUTPUT / "summary.json", report)
    lines = ["# pipeline_v1 × holdout_v1", "", "Aggregate-only, one frozen run; no per-question output.", "",
             "| Metric | dev_v1 | holdout_v1 | Delta |", "|---|---:|---:|---:|"]
    for key in METRICS:
        lines.append(f"| {key} | {report['dev_metrics'][key]:.6f} | {report['metrics'][key]:.6f} | {report['delta_vs_dev'][key]:+.6f} |")
    lines += ["", "## Counts", "", "```json", json.dumps({key: report[key] for key in
              ("total_cases", "answerable_cases", "type_distribution", "no_answer_behavior", "failure_count", "failure_category_counts")},
              ensure_ascii=False, indent=2), "```", "", *report["limits"]]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("--questions-sha256", required=True)
    commands.add_parser("evaluate")
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args.questions_sha256)
        else:
            evaluate()
    except Exception as exc:
        # Exceptions from data-bearing helpers must not leak a case through a traceback.
        print(f"{args.command} stopped: {type(exc).__name__}. Frozen-input or execution check failed; no aggregate published.",
              file=sys.stderr)
        raise SystemExit(1) from None
