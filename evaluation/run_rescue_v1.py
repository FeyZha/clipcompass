"""Offline L-region rescue into frozen M windows, with exactly two strengths."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_reranker_v1 import (
    AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BOUNDARY,
    DENSE_PATH, MODEL_ID, MODEL_PATH, MODEL_REVISION, encode, freeze_hashes,
    load_jsonl, match_detail, memory_bytes, np, score_pairs, sha256, torch, write_json,
)

WINDOWS = ROOT / "evaluation_runs/window_v1"
BASELINE = ROOT / "evaluation_runs/reranker_v1"
OUTPUT = ROOT / "evaluation_runs/rescue_v1"
VARIANTS = {"m_only": 0, "rescue_10": 10, "rescue_20": 20}
FOCUS = ("T001", "T006", "T018", "T019")
WATCH = ("T016", "T029", "T030", "T034")


def region_mapping(m_chunks, l_chunks):
    """Inclusive midpoint containment, same video; no evidence-dependent input."""
    return {large["chunk_id"]: [small["chunk_id"] for small in m_chunks
            if small["video_id"] == large["video_id"]
            and large["start"] <= (small["start"] + small["end"]) / 2 <= large["end"]]
            for large in l_chunks}


def candidate_pool(m_top20_ids, l_ids, mapping):
    ids = list(m_top20_ids)
    provenance = {chunk_id: {"m_top20": True, "l_regions": []} for chunk_id in ids}
    for region_id in l_ids:
        for chunk_id in mapping[region_id]:
            if chunk_id not in provenance:
                ids.append(chunk_id)
                provenance[chunk_id] = {"m_top20": False, "l_regions": []}
            provenance[chunk_id]["l_regions"].append(region_id)
    return ids, provenance


def snapshot():
    hashes = freeze_hashes()
    extra = list(BASELINE.rglob("*")) + [ROOT / "evaluation/run_reranker_v1.py"]
    for path in extra:
        if path.is_file():
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    for directory in (MODEL_PATH, DENSE_PATH):
        for path in directory.iterdir():
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def summarize(rows):
    answered = [r for r in rows if r["gold_answerable"]]
    n = len(answered)
    ranks = [r["gold_pool_rank"] for r in answered]
    return {
        "answerable_cases": n,
        "candidate_recall": sum(rank is not None for rank in ranks) / n,
        **{f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / n for k in (5, 10, 20)},
        "top_1": sum(rank == 1 for rank in ranks) / n,
        "mrr_at_20": sum(1 / rank if rank is not None and rank <= 20 else 0 for rank in ranks) / n,
        "timestamp_hit": sum(r["results"][0]["is_gold"] for r in answered) / n,
        "video_recall_at_5": sum(any(c["video_id"] in {g["video_id"] for g in r["gold_evidence"]}
                                     for c in r["results"][:5]) for r in answered) / n,
        "mean_top1_duration_seconds": statistics.mean(r["results"][0]["end"] - r["results"][0]["start"] for r in answered),
        "mean_top1_duration_all40_seconds": statistics.mean(r["results"][0]["end"] - r["results"][0]["start"] for r in rows),
        "candidate_count_mean": statistics.mean(r["candidate_count"] for r in rows),
        "candidate_count_min": min(r["candidate_count"] for r in rows),
        "candidate_count_max": max(r["candidate_count"] for r in rows),
        "reranker_pairs": sum(r["candidate_count"] for r in rows),
        "rerank_latency_ms": statistics.mean(r["rerank_latency_ms"] for r in rows),
        "rerank_p95_ms": float(np.percentile([r["rerank_latency_ms"] for r in rows], 95)),
        "local_chain_latency_ms": statistics.mean(r["local_chain_latency_ms"] for r in rows),
        "memory_sampled_rss_max_bytes": max(r["memory_bytes"]["rss"] for r in rows),
        "candidate_misses": [r["test_id"] for r in answered if r["gold_pool_rank"] is None],
        "top20_misses": [r["test_id"] for r in answered if r["gold_pool_rank"] is None or r["gold_pool_rank"] > 20],
    }


def changes(base, row):
    if not row["gold_answerable"]:
        return {"labels": ["NO_ANSWER_UNASSESSED"], "top1_gain": False, "top1_damage": False}
    before, after = base["gold_pool_rank"], row["gold_pool_rank"]
    gain = after is not None and (before is None or after < before)
    damage = before is not None and (after is None or after > before)
    return {"labels": ["RESCUE_GAIN"] if gain else ["RESCUE_DAMAGE"] if damage else ["UNCHANGED"],
            "top1_gain": before != 1 and after == 1, "top1_damage": before == 1 and after != 1,
            "top5_gain": (before is None or before > 5) and after is not None and after <= 5,
            "top5_damage": before is not None and before <= 5 and (after is None or after > 5),
            "candidate_gain": before is None and after is not None}


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT}")
    frozen = snapshot()
    tests = load_jsonl(ROOT / "evaluation_data/test_cases/test_cases.jsonl")
    queries = {r["test_id"]: r for r in load_jsonl(ROOT / "evaluation_runs/dense_v1/results.jsonl")}
    baseline_rows = {r["test_id"]: r for r in load_jsonl(BASELINE / "results.jsonl")}
    baseline_config = json.loads((BASELINE / "config.json").read_text(encoding="utf-8"))
    window_config = json.loads((WINDOWS / "config.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "evaluation_data/test_cases/test_cases.jsonl") == window_config["ground_truth_sha256"]
    assert len(tests) == len(queries) == len(baseline_rows) == 40
    for file, expected in baseline_config["model_files_sha256"].items():
        assert sha256(MODEL_PATH / file) == expected
    chunks, vectors, previous = {}, {}, {}
    for size in ("m", "l"):
        directory = WINDOWS / f"dense_window_{size}"
        chunks[size] = load_jsonl(directory / "chunks.jsonl")
        previous[size] = {r["test_id"]: r for r in load_jsonl(directory / "results.jsonl")}
        with np.load(directory / "embedding_cache.npz", allow_pickle=False) as cache:
            vectors[size] = cache["embeddings"]
        assert len(chunks[size]) == len(vectors[size])
    mapping = region_mapping(chunks["m"], chunks["l"])
    m_by_id = {c["chunk_id"]: c for c in chunks["m"]}
    started = time.perf_counter()
    rerank_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).eval()
    dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_PATH, local_files_only=True)
    dense_model = AutoModel.from_pretrained(DENSE_PATH, local_files_only=True).eval()
    load_ms = (time.perf_counter() - started) * 1000
    OUTPUT.mkdir()
    for name in VARIANTS:
        (OUTPUT / name).mkdir()
    config = {
        "run_id": "rescue_v1", "frozen_sources_sha256": frozen,
        "runner_sha256": sha256(Path(__file__)), "variants": VARIANTS,
        "mapping": "same video AND L.start <= (M.start+M.end)/2 <= L.end; deduplicate by M chunk_id",
        "pool_order": "original M dense Top20, then L dense rank and original M index order; stable ties",
        "final_unit": "unchanged M chunks only", "candidate_source": "frozen raw dense Top20, not unique-region results",
        "reranker": MODEL_ID, "reranker_revision": MODEL_REVISION,
        "reranker_batch_size": 20, "max_pair_tokens": 512, "score_fusion": False,
        "embedding": "BAAI/bge-small-en-v1.5", "query_source": "cached dense_v1 english_query",
        "metric_rule": "frozen Window M same-video positive temporal overlap; MRR truncated at final Top20",
        "new_api_calls": 0, "new_api_cost_cny": 0, "model_load_total_ms": load_ms,
        "cpu_threads": torch.get_num_threads(), "memory_before_queries_bytes": memory_bytes(),
        "memory_note": "RSS sampled after each variant; lifetime peak is shared process, not isolated per variant",
    }
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "l_to_m_mapping.json", mapping)
    by_variant = {name: [] for name in VARIANTS}
    for test in tests:
        test_id = test["id"]
        english = queries[test_id]["dense_query"]
        assert queries[test_id]["question"] == test["question"] == baseline_rows[test_id]["question"]
        started = time.perf_counter()
        qvec = encode(dense_model, dense_tokenizer, [english], query=True, batch_size=1)[0]
        embedding_ms = (time.perf_counter() - started) * 1000
        orders, retrieval_ms = {}, {}
        for size in ("m", "l"):
            started = time.perf_counter()
            scores = vectors[size] @ qvec
            orders[size] = np.argsort(-scores, kind="stable")[:20]
            retrieval_ms[size] = (time.perf_counter() - started) * 1000
            actual_ids = [chunks[size][int(i)]["chunk_id"] for i in orders[size]]
            expected = previous[size][test_id]
            assert actual_ids == [c["chunk_id"] for c in expected["raw_top20"]], f"{size} candidate drift"
            assert expected["gold_evidence"] == test["gold_evidence"]
        m_ids = [chunks["m"][int(i)]["chunk_id"] for i in orders["m"]]
        l_ids = [chunks["l"][int(i)]["chunk_id"] for i in orders["l"]]
        for name, rescue_k in VARIANTS.items():
            started = time.perf_counter()
            pool_ids, provenance = candidate_pool(m_ids, l_ids[:rescue_k], mapping)
            pool = [m_by_id[i] for i in pool_ids]
            mapping_ms = (time.perf_counter() - started) * 1000
            assert set(m_ids).issubset(pool_ids) and len(pool_ids) == len(set(pool_ids))
            lengths = rerank_tokenizer([english] * len(pool), [c["text"] for c in pool], truncation=False)["input_ids"]
            started = time.perf_counter()
            scores = []
            for offset in range(0, len(pool), 20):
                scores.extend(score_pairs(reranker, rerank_tokenizer, english, [c["text"] for c in pool[offset:offset + 20]]))
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            rerank_ms = (time.perf_counter() - started) * 1000
            ranked = [{**pool[i], "rank": rank, "reranker_score": scores[i], "provenance": provenance[pool_ids[i]],
                       **match_detail(pool[i], test["gold_evidence"])} for rank, i in enumerate(order, 1)]
            gold_rank = next((c["rank"] for c in ranked if c["is_gold"]), None)
            if name == "m_only":
                assert [c["chunk_id"] for c in ranked] == [c["chunk_id"] for c in baseline_rows[test_id]["results"]], "baseline ranking drift"
            else:
                first_scores = {c["chunk_id"]: c["reranker_score"] for c in by_variant["m_only"][-1]["ranked_candidates"]}
                assert all(abs(c["reranker_score"] - first_scores[c["chunk_id"]]) < 1e-6 for c in ranked if c["chunk_id"] in first_scores)
            row = {
                "test_id": test_id, "question": test["question"], "english_query": english,
                "evaluation_bucket": "boundary" if test_id in BOUNDARY else "core",
                "gold_answerable": test["answerable"], "gold_evidence": test["gold_evidence"],
                "candidate_count_before": 20, "candidate_count": len(pool), "candidate_ids": pool_ids,
                "l_candidates": previous["l"][test_id]["raw_top20"][:rescue_k],
                "m_original_gold_rank": previous["m"][test_id]["first_gold_raw_rank"],
                "l_original_gold_rank": previous["l"][test_id]["first_gold_raw_rank"],
                "gold_pool_rank": gold_rank, "gold_final_top20_rank": gold_rank if gold_rank is not None and gold_rank <= 20 else None,
                "ranked_candidates": ranked, "results": ranked[:20],
                "embedding_latency_ms": embedding_ms, "m_dense_latency_ms": retrieval_ms["m"],
                "l_dense_latency_ms": retrieval_ms["l"] if rescue_k else 0,
                "mapping_latency_ms": mapping_ms, "rerank_latency_ms": rerank_ms,
                "local_chain_latency_ms": embedding_ms + retrieval_ms["m"] + (retrieval_ms["l"] if rescue_k else 0) + mapping_ms + rerank_ms,
                "memory_bytes": memory_bytes(), "truncated_pairs": sum(len(ids) > 512 for ids in lengths), "error": None,
            }
            base = row if name == "m_only" else by_variant["m_only"][-1]
            row["comparison_to_m_only"] = {"gold_rank_before": base["gold_pool_rank"], **changes(base, row)}
            by_variant[name].append(row)
            with (OUTPUT / name / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(test_id + " " + " ".join(f"{name}: {rows[-1]['candidate_count']} candidates, gold={rows[-1]['gold_pool_rank']}" for name, rows in by_variant.items()), flush=True)

    assert snapshot() == frozen, "Frozen source changes"
    reports = {name: summarize(rows) for name, rows in by_variant.items()}
    original = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))["reranked"]
    assert all(math.isclose(reports["m_only"][key], value) for key, value in original.items())
    comparisons = {}
    for name, rows in by_variant.items():
        if name == "m_only":
            continue
        compact = [{"test_id": r["test_id"], "question": r["question"], "gold_rank_after": r["gold_pool_rank"],
                    **r["comparison_to_m_only"]} for r in rows]
        comparisons[name] = {
            "rank_gains": [r for r in compact if "RESCUE_GAIN" in r["labels"]],
            "rank_damages": [r for r in compact if "RESCUE_DAMAGE" in r["labels"]],
            "top1_gains": [r["test_id"] for r in compact if r["top1_gain"]],
            "top1_damages": [r["test_id"] for r in compact if r["top1_damage"]],
            "original_top1_checks": [r for r in compact if r["gold_rank_before"] == 1],
            "previous_damage_checks": [r for r in compact if r["test_id"] in WATCH],
        }
    focus = {test_id: {name: {key: row[key] for key in ("m_original_gold_rank", "l_original_gold_rank", "gold_pool_rank", "gold_final_top20_rank", "candidate_count")}
                       for name, rows in by_variant.items() if (row := next(r for r in rows if r["test_id"] == test_id))} for test_id in FOCUS}
    summary = {"metrics": reports, "changes": comparisons, "focus_cases": focus,
               "memory_process_lifetime_peak_bytes": memory_bytes()["peak_rss"],
               "checks": {"frozen_sources_unchanged": True, "all_outputs_original_m_windows": True,
                          "live_dense_matches_frozen": True, "m_baseline_ranks_identical": True,
                          "m_original_scores_identical_in_rescue": True,
                          "truncated_pairs": sum(r["truncated_pairs"] for rows in by_variant.values() for r in rows)},
               "new_api_calls": 0, "new_api_cost_cny": 0,
               "buckets": {name: {bucket: summarize([r for r in rows if r["evaluation_bucket"] == bucket]) for bucket in ("core", "boundary")}
                           for name, rows in by_variant.items()}}
    write_json(OUTPUT / "summary.json", summary)
    lines = ["# M-only vs L-region Rescue", "", "| Metric | M only | Rescue-10 | Rescue-20 |", "|---|---:|---:|---:|"]
    for key in reports["m_only"]:
        if isinstance(reports["m_only"][key], (float, int)):
            lines.append("| " + key + " | " + " | ".join(f"{reports[name][key]:.6f}" for name in VARIANTS) + " |")
    lines += ["", "## Four retrieval ceiling cases", "", "```json", json.dumps(focus, indent=2), "```",
              "", "## Rescue gains and damages", "", "```json", json.dumps(comparisons, ensure_ascii=False, indent=2), "```",
              "", "All candidates and results are unchanged original M windows. L contributes only region-to-M mapping.",
              "Positive time-overlap is the frozen hit rule; it does not prove semantic sufficiency. MRR is truncated at Top20.",
              "All local latency measurements include one query embedding; DeepSeek is cached, new API calls/cost=0.",
              "Memory per variant is sampled process RSS, not independent model memory; lifetime peak includes all variants.",
              "No-answer cases are ranked, without a new Answer Gate decision."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": reports, "checks": summary["checks"]}), flush=True)


if __name__ == "__main__":
    main()
