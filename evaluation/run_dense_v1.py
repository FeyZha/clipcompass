"""Run lexical forensics and the frozen dense-v1 retrieval evaluation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.server import normalize, retrieval_query, search_tokens, understand_query


MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_PATH = ROOT / "models" / "bge-small-en-v1.5"
DATA_ROOT = ROOT / "evaluation_data"
RUN_ROOT = ROOT / "evaluation_runs" / "dense_v1"
TEST_CASES_PATH = DATA_ROOT / "test_cases" / "test_cases.jsonl"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BOUNDARY_IDS = {"T022", "T024", "T036", "T040"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, str]]:
    metadata = json.loads((DATA_ROOT / "raw" / "videos.json").read_text(encoding="utf-8"))
    corpus: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for video in metadata:
        video_id = video["video_id"]
        path = DATA_ROOT / "normalized" / f"{video_id}.json"
        hashes[video_id] = sha256(path)
        normalized = json.loads(path.read_text(encoding="utf-8"))
        for segment_id, segment in enumerate(normalized["segments"], 1):
            corpus.append({
                "video_id": video_id,
                "segment_id": segment_id,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment["text"]),
            })
    return corpus, hashes


def overlaps(segment: dict[str, Any], evidence: dict[str, Any]) -> bool:
    if segment["video_id"] != evidence["video_id"]:
        return False
    intersection = max(0.0, min(segment["end"], evidence["end"]) - max(segment["start"], evidence["start"]))
    shorter = min(segment["end"] - segment["start"], evidence["end"] - evidence["start"])
    return shorter > 0 and intersection / shorter >= 0.5


def is_gold(segment: dict[str, Any], test: dict[str, Any]) -> bool:
    return any(overlaps(segment, evidence) for evidence in test["gold_evidence"])


def encode(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    query: bool = False,
    batch_size: int = 128,
) -> np.ndarray:
    if query:
        texts = [QUERY_INSTRUCTION + text for text in texts]
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            vector = model(**inputs).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            vectors.append(vector.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(vectors)


def lexical_forensic(corpus: list[dict[str, Any]], test: dict[str, Any], query: str) -> dict[str, Any]:
    query_normalized = normalize(query)
    query_tokens = search_tokens(query)
    ranked: list[dict[str, Any]] = []
    gold_scores: list[float] = []
    gold_passes_threshold = False
    for corpus_index, segment in enumerate(corpus):
        transcript_tokens = search_tokens(segment["text"])
        matched = query_tokens & transcript_tokens
        coverage = len(matched) / len(query_tokens) if query_tokens else 0.0
        exact = bool(query_normalized and query_normalized in normalize(segment["text"]))
        raw_score = coverage * 100 + (30 if exact else 0)
        entry = {
            "corpus_index": corpus_index,
            "video_id": segment["video_id"],
            "segment_id": segment["segment_id"],
            "start": segment["start"],
            "end": segment["end"],
            "raw_score": round(raw_score, 6),
            "coverage": round(coverage, 6),
            "matched_terms": sorted(matched),
            "passes_threshold": exact or coverage >= 0.75,
            "text": segment["text"],
            "is_gold": is_gold(segment, test),
        }
        if entry["is_gold"]:
            gold_scores.append(raw_score)
            gold_passes_threshold = gold_passes_threshold or entry["passes_threshold"]
        if matched or exact:
            ranked.append(entry)
    ranked.sort(key=lambda item: (-item["raw_score"], item["start"], item["corpus_index"]))
    for rank, entry in enumerate(ranked, 1):
        entry["raw_rank"] = rank
    top20 = ranked[:20]
    gold_ranks = [entry["raw_rank"] for entry in ranked if entry["is_gold"]]
    gold_in_top20 = any(entry["is_gold"] for entry in top20)
    return {
        "test_id": test["id"],
        "question": test["question"],
        "answerable": test["answerable"],
        "retrieval_query": query,
        "query_token_count": len(query_tokens),
        "top20_raw_candidates": top20,
        "gold_in_top20": gold_in_top20,
        "first_gold_raw_rank": min(gold_ranks, default=None),
        "gold_raw_score": round(max(gold_scores, default=0.0), 6),
        "gold_passes_threshold": gold_passes_threshold,
        "threshold_only_filtered": bool(test["answerable"] and gold_in_top20 and not gold_passes_threshold),
        "no_raw_top20_recall": bool(test["answerable"] and not gold_in_top20),
        "no_lexical_overlap": bool(test["answerable"] and max(gold_scores, default=0.0) == 0),
    }


def price_usage(usage: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    beijing = now.astimezone(dt.timezone(dt.timedelta(hours=8)))
    peak = beijing.weekday() < 5 and ((9 <= beijing.hour < 12) or (14 <= beijing.hour < 18))
    rates = {"cached": 0.04, "uncached": 2.0, "output": 8.0} if peak else {
        "cached": 0.02, "uncached": 1.0, "output": 4.0
    }
    cached = int(usage.get("cached_input_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    usage["estimated_cost_cny"] = round(
        (cached * rates["cached"] + (input_tokens - cached) * rates["uncached"] + output_tokens * rates["output"])
        / 1_000_000,
        8,
    )
    usage["pricing_period"] = "peak" if peak else "off_peak"
    return usage


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["gold_answerable"]]
    total = len(answerable)
    no_answer = len(rows) - total
    return {
        "answerable_cases": total,
        "no_answer_cases": no_answer,
        "recall_at_5": sum(row["first_gold_rank"] is not None and row["first_gold_rank"] <= 5 for row in answerable) / total,
        "recall_at_10": sum(row["first_gold_rank"] is not None and row["first_gold_rank"] <= 10 for row in answerable) / total,
        "recall_at_20": sum(row["first_gold_rank"] is not None for row in answerable) / total,
        "top_1_accuracy": sum(row["first_gold_rank"] == 1 for row in answerable) / total,
        "mrr_at_20": sum(1 / row["first_gold_rank"] if row["first_gold_rank"] else 0 for row in answerable) / total,
        "timestamp_hit": sum(row["timestamp_hit"] for row in answerable) / total,
    }


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    if not MODEL_PATH.is_dir():
        raise SystemExit(f"Local model missing: {MODEL_PATH}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tests = load_jsonl(TEST_CASES_PATH)
    corpus, corpus_hashes = load_corpus()

    model_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    model_load_ms = round((time.perf_counter() - model_started) * 1000, 3)

    corpus_fingerprint = hashlib.sha256(
        json.dumps(corpus_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = RUN_ROOT / "embedding_cache.npz"
    cache_reused = False
    index_started = time.perf_counter()
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["corpus_fingerprint"]) != corpus_fingerprint or str(cached["model_id"]) != MODEL_ID:
            raise SystemExit("Embedding cache does not match the frozen corpus or model.")
        corpus_vectors = cached["embeddings"]
        cache_reused = True
    else:
        corpus_vectors = encode(model, tokenizer, [segment["text"] for segment in corpus])
        np.savez_compressed(
            cache_path,
            embeddings=corpus_vectors,
            corpus_fingerprint=np.array(corpus_fingerprint),
            model_id=np.array(MODEL_ID),
        )
    index_ms = round((time.perf_counter() - index_started) * 1000, 3)

    lexical_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    for test in tests:
        query_understanding, usage = understand_query(test["question"])
        usage = price_usage(usage, dt.datetime.now(dt.timezone.utc))
        english_query = query_understanding["english_query"]

        lexical_rows.append(lexical_forensic(corpus, test, retrieval_query(query_understanding)))

        embedding_started = time.perf_counter()
        query_vector = encode(model, tokenizer, [english_query], query=True, batch_size=1)[0]
        embedding_ms = round((time.perf_counter() - embedding_started) * 1000, 3)
        retrieval_started = time.perf_counter()
        scores = corpus_vectors @ query_vector
        top_indices = np.argsort(-scores, kind="stable")[:20]
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
        results = []
        for rank, corpus_index in enumerate(top_indices, 1):
            segment = corpus[int(corpus_index)]
            results.append({
                "rank": rank,
                "video_id": segment["video_id"],
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "score": round(float(scores[corpus_index]), 8),
                "text": segment["text"],
                "is_gold": is_gold(segment, test),
            })
        first_gold_rank = next((result["rank"] for result in results if result["is_gold"]), None)
        gold_videos = {evidence["video_id"] for evidence in test["gold_evidence"]}
        top1_correct_video_wrong_time = bool(
            test["answerable"] and results[0]["video_id"] in gold_videos and not results[0]["is_gold"]
        )
        failure_labels: list[str] = []
        if test["answerable"]:
            if first_gold_rank is None:
                failure_labels.append("DENSE_RETRIEVAL_MISS")
            elif first_gold_rank > 5:
                failure_labels.append("CANDIDATE_RANKING_WEAK")
            elif first_gold_rank > 1:
                failure_labels.append("RANKING_ERROR")
            if top1_correct_video_wrong_time:
                failure_labels.append("TIMESTAMP_ERROR")
            if first_gold_rank is None and any(result["video_id"] in gold_videos for result in results):
                failure_labels.append("POSSIBLE_SEGMENTATION_ERROR")
        dense_rows.append({
            "test_id": test["id"],
            "question": test["question"],
            "evaluation_bucket": "boundary" if test["id"] in BOUNDARY_IDS else "core",
            "query_understanding": query_understanding,
            "dense_query": english_query,
            "llm_usage": usage,
            "results": results,
            "gold_answerable": test["answerable"],
            "gold_evidence": test["gold_evidence"],
            "first_gold_rank": first_gold_rank,
            "timestamp_hit": first_gold_rank == 1,
            "top1_correct_video_wrong_timestamp": top1_correct_video_wrong_time,
            "failure_labels": failure_labels,
            "embedding_latency_ms": embedding_ms,
            "retrieval_latency_ms": retrieval_ms,
            "total_latency_ms": round(usage["latency_ms"] + embedding_ms + retrieval_ms, 3),
        })

    write_jsonl(RUN_ROOT / "lexical_forensic.jsonl", lexical_rows)
    write_jsonl(RUN_ROOT / "results.jsonl", dense_rows)
    full_metrics = metrics(dense_rows)
    core_metrics = metrics([row for row in dense_rows if row["evaluation_bucket"] == "core"])
    boundary_metrics = metrics([row for row in dense_rows if row["evaluation_bucket"] == "boundary"])
    forensic_answerable = [row for row in lexical_rows if row["answerable"]]
    failure_counts = {
        label: sum(label in row["failure_labels"] for row in dense_rows)
        for label in (
            "DENSE_RETRIEVAL_MISS",
            "CANDIDATE_RANKING_WEAK",
            "RANKING_ERROR",
            "TIMESTAMP_ERROR",
            "POSSIBLE_SEGMENTATION_ERROR",
        )
    }
    llm_usage = [row["llm_usage"] for row in dense_rows]
    config = {
        "run_id": "dense_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ground_truth_sha256": sha256(TEST_CASES_PATH),
        "normalized_sha256": corpus_hashes,
        "query_understanding": "DeepSeek deepseek-flash; question only; dense input uses english_query only",
        "model": MODEL_ID,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_local_only": True,
        "query_instruction": QUERY_INSTRUCTION,
        "pooling": "CLS",
        "normalized_embeddings": True,
        "similarity": "cosine via dot product of L2-normalized vectors",
        "retrieval_unit": "one frozen normalized subtitle segment; unchanged",
        "top_k": [5, 10, 20],
        "business_threshold": None,
        "corpus_segments": len(corpus),
        "model_load_ms": model_load_ms,
        "index_build_or_load_ms": index_ms,
        "index_cache_reused": cache_reused,
        "index_size_bytes": cache_path.stat().st_size,
        "python_packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    write_json(RUN_ROOT / "config.json", config)

    severe = sorted(
        [row for row in dense_rows if row["gold_answerable"] and row["failure_labels"]],
        key=lambda row: (
            0 if "DENSE_RETRIEVAL_MISS" in row["failure_labels"] else 1,
            -(row["first_gold_rank"] or 999),
            row["test_id"],
        ),
    )[:10]
    summary = [
        "# Dense Retrieval v1",
        "",
        "## Lexical forensic",
        "",
        f"- Gold in raw lexical Top 20: {sum(row['gold_in_top20'] for row in forensic_answerable)}/34",
        f"- Threshold-only filtered: {sum(row['threshold_only_filtered'] for row in forensic_answerable)}/34",
        f"- Still absent from raw Top 20 with threshold removed: {sum(row['no_raw_top20_recall'] for row in forensic_answerable)}/34",
        f"- Gold with zero lexical overlap: {sum(row['no_lexical_overlap'] for row in forensic_answerable)}/34",
        "",
        "## Dense metrics",
        "",
        "| Set | Recall@5 | Recall@10 | Recall@20 | Top-1 | MRR@20 | Timestamp Hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in (("Core", core_metrics), ("Boundary", boundary_metrics), ("Full", full_metrics)):
        summary.append(
            f"| {name} | {percentage(item['recall_at_5'])} | {percentage(item['recall_at_10'])} | "
            f"{percentage(item['recall_at_20'])} | {percentage(item['top_1_accuracy'])} | "
            f"{item['mrr_at_20']:.3f} | {percentage(item['timestamp_hit'])} |"
        )
    summary.extend([
        "",
        "## Failure counts",
        "",
        *[f"- {label}: {count}" for label, count in failure_counts.items()],
        "",
        "## Performance and cost",
        "",
        f"- Model load: {model_load_ms:.3f} ms",
        f"- Full corpus index: {index_ms:.3f} ms (cache reused: {str(cache_reused).lower()})",
        f"- Index size: {cache_path.stat().st_size} bytes",
        f"- Average query embedding: {np.mean([row['embedding_latency_ms'] for row in dense_rows]):.3f} ms",
        f"- Average retrieval: {np.mean([row['retrieval_latency_ms'] for row in dense_rows]):.3f} ms",
        f"- Average DeepSeek Query Understanding: {np.mean([usage['latency_ms'] for usage in llm_usage]):.3f} ms",
        f"- Average total per question: {np.mean([row['total_latency_ms'] for row in dense_rows]):.3f} ms",
        f"- DeepSeek calls: {len(llm_usage)}",
        f"- DeepSeek input tokens: {sum(usage['input_tokens'] for usage in llm_usage)}",
        f"- DeepSeek output tokens: {sum(usage['output_tokens'] for usage in llm_usage)}",
        f"- DeepSeek estimated cost: CNY {sum(usage['estimated_cost_cny'] for usage in llm_usage):.8f}",
        "- Local embedding API cost: CNY 0 (local compute)",
        "",
        "## Ten representative failures",
        "",
        *[
            f"- `{row['test_id']}` — {', '.join(row['failure_labels'])}; first gold rank: {row['first_gold_rank']} — {row['question']}"
            for row in severe
        ],
        "",
        "No segmentation, reranker, Answer Gate, frontend, or Ground Truth changes were made.",
    ])
    (RUN_ROOT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": full_metrics, "failures": failure_counts, "run_root": str(RUN_ROOT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
