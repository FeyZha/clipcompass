"""Offline cross-encoder experiment on exactly the frozen Window M raw Top20."""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from evaluation.run_dense_v1 import encode, load_jsonl, sha256, write_json, write_jsonl
from evaluation.run_window_v1 import match_detail

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_PATH = ROOT / "models/ms-marco-MiniLM-L6-v2"
DENSE_PATH = ROOT / "models/bge-small-en-v1.5"
WINDOW_PATH = ROOT / "evaluation_runs/window_v1/dense_window_m"
OUTPUT = ROOT / "evaluation_runs/reranker_v1"
BOUNDARY = {"T022", "T024", "T036", "T040"}


def memory_bytes():
    """Windows process working set and lifetime peak, not system-wide RAM use."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "pagefile", "peak_pagefile", "private",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"rss": counters.rss, "peak_rss": counters.peak_rss, "private": counters.private}


def freeze_hashes():
    paths = set()
    for directory in ("evaluation_data", "evaluation_runs/baseline_v1", "evaluation_runs/query_understanding_v1",
                      "evaluation_runs/dense_v1", "evaluation_runs/window_v1", "library", "llm"):
        paths.update(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update(ROOT / name for name in ("demo.py", "evaluation/run_dense_v1.py", "evaluation/run_window_v1.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def score_pairs(model, tokenizer, question, texts):
    """No evidence labels, video IDs or Dense scores enter model inference."""
    features = tokenizer([question] * len(texts), texts, padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
    with torch.inference_mode():
        scores = model(**features).logits.squeeze(-1).cpu().tolist()
    if len(scores) != len(texts) or not all(math.isfinite(score) for score in scores):
        raise ValueError("Cross-encoder returned invalid scores")
    return scores


def reorder(candidates, scores):
    if len(candidates) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Invalid reranker scores")
    # Stable ties retain the original Dense order; there is no score fusion.
    return [{**candidates[i], "rank": rank, "dense_rank": candidates[i]["rank"],
             "dense_score": candidates[i]["score"], "reranker_score": scores[i]}
            for rank, i in enumerate(sorted(range(len(scores)), key=lambda i: -scores[i]), 1)]


def first_gold(candidates):
    return next((item["rank"] for item in candidates if item["is_gold"]), None)


def classifications(before, after, answerable):
    if not answerable:
        return ["NO_ANSWER_UNASSESSED"]
    if before is None:
        assert after is None
        return ["RETRIEVAL_CEILING"]
    assert after is not None
    labels = []
    if before > 5 and after <= 5:
        labels.append("RANKING_RESCUE_TOP5")
    if before > 1 and after == 1:
        labels.append("RANKING_RESCUE_TOP1")
    labels.append("RANKING_DAMAGE" if after > before else "RANK_IMPROVED" if after < before else "RANK_UNCHANGED")
    return labels


def metrics(rows, result_key, rank_key):
    rows = [row for row in rows if row["gold_answerable"]]
    ranks = [row[rank_key] for row in rows]
    total = len(rows)
    return {
        "answerable_cases": total,
        **{f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / total for k in (5, 10, 20)},
        "top_1": sum(rank == 1 for rank in ranks) / total,
        "mrr_at_20": sum(1 / rank if rank else 0 for rank in ranks) / total,
        "timestamp_hit": sum(row[result_key][0]["is_gold"] for row in rows) / total,
        "video_recall_at_5": sum(any(item["video_id"] in {g["video_id"] for g in row["gold_evidence"]}
                                    for item in row[result_key][:5]) for row in rows) / total,
    }


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite an existing experiment: {OUTPUT}")
    frozen = freeze_hashes()
    tests = load_jsonl(ROOT / "evaluation_data/test_cases/test_cases.jsonl")
    queries = {row["test_id"]: row for row in load_jsonl(ROOT / "evaluation_runs/dense_v1/results.jsonl")}
    dense = {row["test_id"]: row for row in load_jsonl(WINDOW_PATH / "results.jsonl")}
    chunks = load_jsonl(WINDOW_PATH / "chunks.jsonl")
    chunk_map = {item["chunk_id"]: item for item in chunks}
    window_config = json.loads((WINDOW_PATH / "config.json").read_text(encoding="utf-8"))
    window_freeze = json.loads((WINDOW_PATH.parent / "config.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "evaluation_data/test_cases/test_cases.jsonl") == window_freeze["ground_truth_sha256"]
    assert len(tests) == len(dense) == len(queries) == 40
    for test in tests:
        source = dense[test["id"]]
        assert test["question"] == source["question"] == queries[test["id"]]["question"]
        assert source["gold_evidence"] == test["gold_evidence"]
        assert len(source["raw_top20"]) == len({c["chunk_id"] for c in source["raw_top20"]}) == 20
        for candidate in source["raw_top20"]:
            chunk = chunk_map[candidate["chunk_id"]]
            assert all(candidate[key] == chunk[key] for key in ("text", "video_id", "start", "end", "segment_ids"))
            assert candidate["is_gold"] == match_detail(candidate, test["gold_evidence"])["is_gold"]

    memory_before = memory_bytes()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).eval()
    load_ms = (time.perf_counter() - started) * 1000
    memory_after_reranker = memory_bytes()
    dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_PATH, local_files_only=True)
    dense_model = AutoModel.from_pretrained(DENSE_PATH, local_files_only=True).eval()
    with np.load(WINDOW_PATH / "embedding_cache.npz", allow_pickle=False) as cache:
        corpus_vectors = cache["embeddings"]

    OUTPUT.mkdir()
    config = {
        "run_id": "reranker_v1", "source_hashes": frozen,
        "runner_sha256": sha256(Path(__file__)), "model": MODEL_ID, "revision": MODEL_REVISION,
        "model_files_sha256": {p.name: sha256(p) for p in sorted(MODEL_PATH.iterdir()) if p.is_file()},
        "model_disk_bytes": sum(p.stat().st_size for p in MODEL_PATH.iterdir() if p.is_file()),
        "candidate_source": "window_v1/dense_window_m/results.jsonl:raw_top20",
        "query_source": "dense_v1/results.jsonl:dense_query", "query_source_is_cached": True,
        "score": "raw sequence-classification logit; descending stable sort; no Dense score fusion",
        "max_length": 512, "batch_size": 20, "device": "cpu", "dtype": "float32",
        "gold_match_rule": "unchanged Window M rule: same video, positive temporal overlap",
        "no_answer": "all six scored, no answer decision or accuracy assigned",
        "new_api_calls": 0, "new_api_cost_cny": 0,
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "transformers": __import__("transformers").__version__, "python": sys.version},
        "platform": platform.platform(), "torch_cpu_threads": torch.get_num_threads(),
    }
    write_json(OUTPUT / "config.json", config)
    rows = []
    for test in tests:
        source = dense[test["id"]]
        candidates = source["raw_top20"]
        english = queries[test["id"]]["dense_query"]
        started = time.perf_counter()
        query_vector = encode(dense_model, dense_tokenizer, [english], query=True, batch_size=1)[0]
        embedding_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        dense_scores = corpus_vectors @ query_vector
        order = np.argsort(-dense_scores, kind="stable")[:20]
        retrieval_ms = (time.perf_counter() - started) * 1000
        assert [chunks[int(i)]["chunk_id"] for i in order] == [c["chunk_id"] for c in candidates], "Dense Top20 drift"
        pair_lengths = tokenizer([english] * 20, [c["text"] for c in candidates], truncation=False)["input_ids"]
        started = time.perf_counter()
        scores = score_pairs(reranker, tokenizer, english, [c["text"] for c in candidates])
        results = reorder(candidates, scores)
        rerank_ms = (time.perf_counter() - started) * 1000
        before, after = first_gold(candidates), first_gold(results)
        assert (before is None) == (after is None)
        assert {c["chunk_id"] for c in candidates} == {c["chunk_id"] for c in results}
        row = {
            "test_id": test["id"], "question": test["question"], "english_query": english,
            "evaluation_bucket": "boundary" if test["id"] in BOUNDARY else "core",
            "gold_answerable": test["answerable"], "gold_evidence": test["gold_evidence"],
            "gold_rank_before": before, "gold_rank_after": after,
            "dense_full_index_gold_rank": source["first_gold_raw_rank"],
            "labels": classifications(before, after, test["answerable"]),
            "dense_results": candidates, "results": results,
            "candidate_set_unchanged": True, "live_dense_matches_frozen": True,
            "embedding_latency_ms": embedding_ms, "retrieval_latency_ms": retrieval_ms,
            "rerank_latency_ms": rerank_ms, "local_chain_latency_ms": embedding_ms + retrieval_ms + rerank_ms,
            "max_pair_tokens": max(map(len, pair_lengths)),
            "truncated_pairs": sum(len(ids) > 512 for ids in pair_lengths),
            "memory_bytes": memory_bytes(), "error": None,
        }
        rows.append(row)
        with (OUTPUT / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{test['id']} gold {before} -> {after}; rerank {rerank_ms:.1f} ms", flush=True)

    before_metrics = metrics(rows, "dense_results", "gold_rank_before")
    after_metrics = metrics(rows, "results", "gold_rank_after")
    assert before_metrics["recall_at_20"] == after_metrics["recall_at_20"] == 30 / 34
    for key, source_key in (("recall_at_5", "recall_at_5"), ("recall_at_10", "recall_at_10"),
                            ("recall_at_20", "raw_recall_at_20"), ("top_1", "top_1_accuracy"),
                            ("mrr_at_20", "mrr_at_20"), ("timestamp_hit", "timestamp_hit"),
                            ("video_recall_at_5", "video_recall_at_5")):
        assert math.isclose(before_metrics[key], window_config["metrics"][source_key])
    assert freeze_hashes() == frozen, "Frozen files changed during experiment"
    compact = [{k: row[k] for k in ("test_id", "question", "gold_rank_before", "gold_rank_after", "labels")} for row in rows]
    rescues = sorted([r for r in compact if "RANKING_RESCUE_TOP5" in r["labels"] or "RANKING_RESCUE_TOP1" in r["labels"]],
                     key=lambda r: (r["gold_rank_after"], -r["gold_rank_before"], r["test_id"]))
    improvements = sorted([r for r in compact if "RANK_IMPROVED" in r["labels"]],
                          key=lambda r: (r["gold_rank_after"], -r["gold_rank_before"], r["test_id"]))
    damages = sorted([r for r in compact if "RANKING_DAMAGE" in r["labels"]],
                     key=lambda r: (-(1 / r["gold_rank_before"] - 1 / r["gold_rank_after"]), r["test_id"]))
    report = {
        "dense_m": before_metrics, "reranked": after_metrics,
        "conditional_on_gold_in_top20": {"cases": 30,
            "top1_after": sum(r["gold_rank_after"] == 1 for r in rows),
            "top5_after": sum(r["gold_rank_after"] is not None and r["gold_rank_after"] <= 5 for r in rows)},
        "rescue_top5": sum("RANKING_RESCUE_TOP5" in r["labels"] for r in rows),
        "rescue_top1": sum("RANKING_RESCUE_TOP1" in r["labels"] for r in rows),
        "rank_improved": len(improvements), "rank_damaged": len(damages),
        "rescue_cases": rescues, "all_improved_cases": improvements, "damage_cases": damages,
        "retrieval_ceiling": [r for r in compact if "RETRIEVAL_CEILING" in r["labels"]],
        "rank_comparison": compact,
        "performance": {
            "model_load_ms": load_ms,
            **{key: statistics.mean(r[key] for r in rows) for key in
               ("embedding_latency_ms", "retrieval_latency_ms", "rerank_latency_ms", "local_chain_latency_ms")},
            "rerank_p95_ms": float(np.percentile([r["rerank_latency_ms"] for r in rows], 95)),
            "dense_only_same_run_latency_ms": statistics.mean(r["embedding_latency_ms"] + r["retrieval_latency_ms"] for r in rows),
            "frozen_dense_m_latency_ms": window_config["average_local_query_ms"],
            "memory_before_model_bytes": memory_before, "memory_after_reranker_load_bytes": memory_after_reranker,
            "memory_final_bytes": memory_bytes(),
            "memory_peak_working_set_bytes": max(r["memory_bytes"]["peak_rss"] for r in rows),
            "model_disk_bytes": config["model_disk_bytes"],
        },
        "checks": {"questions": len(rows), "candidate_pairs": sum(len(r["results"]) for r in rows),
                   "source_hashes_unchanged": True, "all_candidate_sets_unchanged": True,
                   "all_live_dense_matches_frozen": True, "recall20_unchanged": True,
                   "truncated_pairs": sum(r["truncated_pairs"] for r in rows)},
        "buckets": {name: {"dense_m": metrics(group, "dense_results", "gold_rank_before"),
                           "reranked": metrics(group, "results", "gold_rank_after")}
                    for name in ("core", "boundary") if (group := [r for r in rows if r["evaluation_bucket"] == name])},
    }
    write_json(OUTPUT / "summary.json", report)
    summary = ["# Dense M vs local Cross-Encoder", "", "| Metric | Dense M | Reranked |", "|---|---:|---:|"]
    for key in before_metrics:
        summary.append(f"| {key} | {before_metrics[key]:.6f} | {after_metrics[key]:.6f} |")
    summary += ["", "Gold rank is the first overlapping frozen evidence window (not semantic sufficiency).",
                "Recall@20 and all 40 candidate sets are exactly unchanged. New API calls/cost: 0 / CNY 0.",
                "No-answer cases are scored but no answer decision is made.", "", "## Rank changes", "",
                "| Test | Before | After | Labels |", "|---|---:|---:|---|"]
    summary += [f"| {r['test_id']} | {r['gold_rank_before']} | {r['gold_rank_after']} | {', '.join(r['labels'])} |" for r in compact]
    summary += ["", "## Performance", "", "```json", json.dumps(report["performance"], indent=2), "```",
                "", f"Strict rescue cases: {len(rescues)}; all rank improvements: {len(improvements)}; damages: {len(damages)}.",
                "Remaining retrieval ceiling: " + ", ".join(r["test_id"] for r in report["retrieval_ceiling"])]
    (OUTPUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"before": before_metrics, "after": after_metrics, "rescues": len(rescues),
                      "damages": len(damages), "checks": report["checks"]}), flush=True)


if __name__ == "__main__":
    main()
