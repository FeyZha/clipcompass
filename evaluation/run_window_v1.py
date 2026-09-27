"""Controlled S/M/L word-window experiment over the frozen dense-v1 setup."""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_dense_v1 import (
    DATA_ROOT,
    MODEL_ID,
    MODEL_PATH,
    QUERY_INSTRUCTION,
    ROOT,
    encode,
    load_corpus,
    load_jsonl,
    sha256,
    write_json,
    write_jsonl,
)


RUN_ROOT = ROOT / "evaluation_runs" / "window_v1"
TEST_CASES_PATH = DATA_ROOT / "test_cases" / "test_cases.jsonl"
DENSE_V1_ROOT = ROOT / "evaluation_runs" / "dense_v1"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
SENTENCE_END_RE = re.compile(r'''[.!?](?:["')\]]+)?$''')
REGION_OVERLAP_COEFFICIENT = 0.20
SPECS = {
    "s": {"min_words": 60, "target_words": 80, "max_words": 100, "overlap_words": 25},
    "m": {"min_words": 120, "target_words": 150, "max_words": 180, "overlap_words": 40},
    "l": {"min_words": 220, "target_words": 260, "max_words": 300, "overlap_words": 60},
}
ORIGINAL_DENSE_MISSES = {
    "T001", "T004", "T006", "T008", "T011", "T012", "T016",
    "T018", "T019", "T022", "T023", "T028", "T030", "T031",
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_units(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use punctuation when present; cap punctuation-free runs at an original cue boundary."""
    units: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    words = 0
    for segment in segments:
        current.append(segment)
        words += word_count(segment["text"])
        if SENTENCE_END_RE.search(segment["text"].strip()) or words >= 40:
            units.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "segment_ids": [item["segment_id"] for item in current],
                "word_count": words,
                "text": " ".join(item["text"].strip() for item in current if item["text"].strip()),
            })
            current = []
            words = 0
    if current:
        units.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "segment_ids": [item["segment_id"] for item in current],
            "word_count": words,
            "text": " ".join(item["text"].strip() for item in current if item["text"].strip()),
        })
    return units


def build_windows(corpus: list[dict[str, Any]], name: str, spec: dict[str, int]) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_order: list[str] = []
    for segment in corpus:
        if segment["video_id"] not in by_video:
            video_order.append(segment["video_id"])
        by_video[segment["video_id"]].append(segment)

    chunks: list[dict[str, Any]] = []
    for video_id in video_order:
        units = sentence_units(by_video[video_id])
        start = 0
        number = 1
        while start < len(units):
            end = start
            words = 0
            while end < len(units):
                next_words = units[end]["word_count"]
                if words >= spec["min_words"] and words + next_words > spec["max_words"]:
                    break
                words += next_words
                end += 1
                if words >= spec["target_words"]:
                    break
            if end == start:
                end += 1
                words = units[start]["word_count"]
            selected = units[start:end]
            chunks.append({
                "chunk_id": f"{name}:{video_id}:{number:04d}",
                "video_id": video_id,
                "start": selected[0]["start"],
                "end": selected[-1]["end"],
                "segment_ids": [segment_id for unit in selected for segment_id in unit["segment_ids"]],
                "word_count": words,
                "text": " ".join(unit["text"] for unit in selected),
            })
            number += 1
            if end == len(units):
                break
            overlap = 0
            next_start = end
            while next_start > start and overlap < spec["overlap_words"]:
                next_start -= 1
                overlap += units[next_start]["word_count"]
            start = max(start + 1, next_start)
    return chunks


def match_detail(chunk: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    best = {"is_gold": False, "overlap_seconds": 0.0, "gold_coverage_ratio": 0.0}
    for gold in evidence:
        if chunk["video_id"] != gold["video_id"]:
            continue
        overlap = max(0.0, min(chunk["end"], gold["end"]) - max(chunk["start"], gold["start"]))
        gold_duration = max(0.0, gold["end"] - gold["start"])
        coverage = overlap / gold_duration if gold_duration else 0.0
        if overlap > best["overlap_seconds"]:
            best = {
                "is_gold": overlap > 0,
                "overlap_seconds": round(overlap, 3),
                "gold_coverage_ratio": round(coverage, 6),
            }
    return best


def region_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["video_id"] != right["video_id"]:
        return 0.0
    overlap = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    shorter = min(left["end"] - left["start"], right["end"] - right["start"])
    return overlap / shorter if shorter > 0 else 0.0


def unique_region_indices(order: np.ndarray, chunks: list[dict[str, Any]], limit: int = 20) -> list[int]:
    selected: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if any(region_overlap(chunks[index], chunks[kept]) >= REGION_OVERLAP_COEFFICIENT for kept in selected):
            continue
        selected.append(index)
        if len(selected) == limit:
            break
    return selected


def result_row(rank: int, index: int, score: float, chunks: list[dict[str, Any]], test: dict[str, Any]) -> dict[str, Any]:
    chunk = chunks[index]
    detail = match_detail(chunk, test["gold_evidence"])
    return {
        "rank": rank,
        "chunk_id": chunk["chunk_id"],
        "video_id": chunk["video_id"],
        "start": chunk["start"],
        "end": chunk["end"],
        "segment_ids": chunk["segment_ids"],
        "word_count": chunk["word_count"],
        "score": round(float(score), 8),
        "text": chunk["text"],
        "window_duration_seconds": round(chunk["end"] - chunk["start"], 3),
        **detail,
    }


def evaluate_variant(
    name: str,
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    tests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    retrieval_times: list[float] = []
    for query_vector, test in zip(query_vectors, tests):
        started = time.perf_counter()
        scores = vectors @ query_vector
        order = np.argsort(-scores, kind="stable")
        raw_indices = [int(index) for index in order[:20]]
        unique_indices = unique_region_indices(order, chunks)
        retrieval_times.append((time.perf_counter() - started) * 1000)
        raw = [result_row(rank, index, scores[index], chunks, test) for rank, index in enumerate(raw_indices, 1)]
        unique = [result_row(rank, index, scores[index], chunks, test) for rank, index in enumerate(unique_indices, 1)]
        first_raw_rank = next(
            (rank for rank, index in enumerate(order, 1) if match_detail(chunks[int(index)], test["gold_evidence"])["is_gold"]),
            None,
        )
        first_unique_rank = next((item["rank"] for item in unique if item["is_gold"]), None)
        gold_videos = {item["video_id"] for item in test["gold_evidence"]}
        rows.append({
            "test_id": test["id"],
            "question": test["question"],
            "gold_answerable": test["answerable"],
            "gold_evidence": test["gold_evidence"],
            "raw_top20": raw,
            "unique_region_top20": unique,
            "first_gold_raw_rank": first_raw_rank,
            "first_gold_unique_rank": first_unique_rank,
            "video_recall_at_5": bool(test["answerable"] and any(item["video_id"] in gold_videos for item in raw[:5])),
            "timestamp_hit": bool(test["answerable"] and raw[0]["is_gold"]),
        })
    return rows, float(np.mean(retrieval_times))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["gold_answerable"]]
    total = len(answerable)
    top1 = [row["raw_top20"][0] for row in answerable]
    successful_top1 = [item for item in top1 if item["is_gold"]]
    return {
        "answerable_cases": total,
        "recall_at_5": sum(row["first_gold_raw_rank"] is not None and row["first_gold_raw_rank"] <= 5 for row in answerable) / total,
        "recall_at_10": sum(row["first_gold_raw_rank"] is not None and row["first_gold_raw_rank"] <= 10 for row in answerable) / total,
        "raw_recall_at_20": sum(row["first_gold_raw_rank"] is not None and row["first_gold_raw_rank"] <= 20 for row in answerable) / total,
        "unique_recall_at_20": sum(row["first_gold_unique_rank"] is not None for row in answerable) / total,
        "video_recall_at_5": sum(row["video_recall_at_5"] for row in answerable) / total,
        "top_1_accuracy": sum(row["first_gold_raw_rank"] == 1 for row in answerable) / total,
        "mrr_at_20": sum(1 / row["first_gold_raw_rank"] if row["first_gold_raw_rank"] and row["first_gold_raw_rank"] <= 20 else 0 for row in answerable) / total,
        "timestamp_hit": sum(row["timestamp_hit"] for row in answerable) / total,
        "top20_miss_count": sum(row["first_gold_raw_rank"] is None or row["first_gold_raw_rank"] > 20 for row in answerable),
        "average_top1_duration_seconds": float(np.mean([item["window_duration_seconds"] for item in top1])),
        "average_top1_word_count": float(np.mean([item["word_count"] for item in top1])),
        "average_successful_top1_gold_coverage": float(np.mean([item["gold_coverage_ratio"] for item in successful_top1])) if successful_top1 else 0.0,
    }


def current_video_recall(dense_rows: list[dict[str, Any]]) -> float:
    answerable = [row for row in dense_rows if row["gold_answerable"]]
    hits = 0
    for row in answerable:
        videos = {item["video_id"] for item in row["gold_evidence"]}
        hits += any(item["video_id"] in videos for item in row["results"][:5])
    return hits / len(answerable)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tests = load_jsonl(TEST_CASES_PATH)
    dense_v1_rows = load_jsonl(DENSE_V1_ROOT / "results.jsonl")
    cached_queries = {row["test_id"]: row for row in dense_v1_rows}
    if {test["id"] for test in tests} != set(cached_queries):
        raise SystemExit("Dense-v1 Query Understanding cache does not match the frozen tests.")
    english_queries = [cached_queries[test["id"]]["dense_query"] for test in tests]
    corpus, corpus_hashes = load_corpus()

    model_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    model_load_ms = (time.perf_counter() - model_started) * 1000

    query_vectors: list[np.ndarray] = []
    query_times: list[float] = []
    for query in english_queries:
        started = time.perf_counter()
        query_vectors.append(encode(model, tokenizer, [query], query=True, batch_size=1)[0])
        query_times.append((time.perf_counter() - started) * 1000)
    query_matrix = np.stack(query_vectors)
    np.savez_compressed(RUN_ROOT / "query_embeddings.npz", embeddings=query_matrix)
    average_query_ms = float(np.mean(query_times))

    reports: dict[str, dict[str, Any]] = {}
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for name, spec in SPECS.items():
        variant_root = RUN_ROOT / f"dense_window_{name}"
        variant_root.mkdir(parents=True, exist_ok=True)
        chunks = build_windows(corpus, name, spec)
        write_jsonl(variant_root / "chunks.jsonl", chunks)
        index_started = time.perf_counter()
        vectors = encode(model, tokenizer, [chunk["text"] for chunk in chunks])
        cache_path = variant_root / "embedding_cache.npz"
        np.savez_compressed(cache_path, embeddings=vectors, model_id=np.array(MODEL_ID))
        index_ms = (time.perf_counter() - index_started) * 1000
        rows, retrieval_ms = evaluate_variant(name, chunks, vectors, query_matrix, tests)
        write_jsonl(variant_root / "results.jsonl", rows)
        result_metrics = metrics(rows)
        report = {
            "name": name,
            "window": spec,
            "chunk_count": len(chunks),
            "minimum_chunk_words": min(chunk["word_count"] for chunk in chunks),
            "maximum_chunk_words": max(chunk["word_count"] for chunk in chunks),
            "average_chunk_words": float(np.mean([chunk["word_count"] for chunk in chunks])),
            "index_size_bytes": cache_path.stat().st_size,
            "index_build_ms": index_ms,
            "average_query_embedding_ms": average_query_ms,
            "average_retrieval_ms": retrieval_ms,
            "average_local_query_ms": average_query_ms + retrieval_ms,
            "metrics": result_metrics,
        }
        write_json(variant_root / "config.json", report)
        reports[name] = report
        rows_by_variant[name] = rows

    recovery = []
    for test in tests:
        if test["id"] not in ORIGINAL_DENSE_MISSES:
            continue
        item = {"test_id": test["id"], "question": test["question"], "variants": {}}
        for name in SPECS:
            row = next(row for row in rows_by_variant[name] if row["test_id"] == test["id"])
            item["variants"][name] = {
                "raw_gold_rank": row["first_gold_raw_rank"],
                "unique_region_gold_rank": row["first_gold_unique_rank"],
                "raw_top20_recovered": row["first_gold_raw_rank"] is not None and row["first_gold_raw_rank"] <= 20,
            }
        recovery.append(item)
    write_json(RUN_ROOT / "dense_miss_recovery.json", recovery)

    deepseek_usage = [row["llm_usage"] for row in dense_v1_rows]
    config = {
        "run_id": "window_v1",
        "ground_truth_sha256": sha256(TEST_CASES_PATH),
        "normalized_sha256": corpus_hashes,
        "query_understanding_source": "evaluation_runs/dense_v1/results.jsonl",
        "deepseek_new_calls": 0,
        "deepseek_cached_calls": len(deepseek_usage),
        "deepseek_cached_cost_cny": sum(item["estimated_cost_cny"] for item in deepseek_usage),
        "model": MODEL_ID,
        "model_local_only": True,
        "query_instruction": QUERY_INSTRUCTION,
        "similarity": "cosine via dot product of L2-normalized vectors",
        "reranker": None,
        "business_threshold": None,
        "unique_region_rule": f"same video and temporal overlap / shorter duration >= {REGION_OVERLAP_COEFFICIENT}",
        "sentence_boundary_rule": "punctuation when available; otherwise split only at original subtitle boundary after 40 words",
        "model_load_ms": model_load_ms,
        "average_query_embedding_ms": average_query_ms,
        "variants": reports,
    }
    write_json(RUN_ROOT / "config.json", config)

    current = {
        "recall_at_5": 0.471,
        "recall_at_10": 0.529,
        "raw_recall_at_20": 0.588,
        "unique_recall_at_20": 0.588,
        "video_recall_at_5": current_video_recall(dense_v1_rows),
        "top_1_accuracy": 0.235,
        "mrr_at_20": 0.331,
        "timestamp_hit": 0.235,
        "chunk_count": 5308,
        "index_size_bytes": 7573937,
        "average_local_query_ms": 9.820 + 0.574,
    }
    table_rows = [("Current segment", current)] + [
        (f"Window {name.upper()}", {**reports[name]["metrics"], **reports[name]}) for name in SPECS
    ]
    summary = [
        "# Controlled Window Experiment",
        "",
        "| Index | Recall@5 | Recall@10 | Raw Recall@20 | Unique Recall@20 | Video Recall@5 | Top-1 | MRR@20 | Timestamp Hit | Chunks | Index bytes | Local latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in table_rows:
        summary.append(
            f"| {label} | {pct(item['recall_at_5'])} | {pct(item['recall_at_10'])} | "
            f"{pct(item['raw_recall_at_20'])} | {pct(item['unique_recall_at_20'])} | "
            f"{pct(item['video_recall_at_5'])} | {pct(item['top_1_accuracy'])} | {item['mrr_at_20']:.3f} | "
            f"{pct(item['timestamp_hit'])} | {item['chunk_count']} | {item['index_size_bytes']} | "
            f"{item['average_local_query_ms']:.3f} ms |"
        )
    summary.extend(["", "## Original 14 Dense Misses", ""])
    for item in recovery:
        ranks = ", ".join(
            f"{name.upper()}={item['variants'][name]['raw_gold_rank']}"
            for name in SPECS
        )
        summary.append(f"- `{item['test_id']}`: {ranks}")
    summary.extend([
        "",
        f"New DeepSeek calls: 0; reused cached cost: CNY {config['deepseek_cached_cost_cny']:.8f}.",
        "No Ground Truth, Query Understanding, model, similarity, reranker, Answer Gate, or frontend changes were made.",
    ])
    (RUN_ROOT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"reports": reports, "recovery": recovery}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
