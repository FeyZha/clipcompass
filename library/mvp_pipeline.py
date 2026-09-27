"""Frozen pipeline_mvp_v1 runtime for the four-video Demo Collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from evaluation.run_dense_v1 import encode
from evaluation.run_reranker_v1 import reorder, score_pairs
from llm.answer_gate import GateResponseError
from llm.answer_gate_bundle import attach_bundles, gate_pass


ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "evaluation_runs/window_v1/dense_window_m"
DENSE_MODEL = ROOT / "models/bge-small-en-v1.5"
RERANKER_MODEL = ROOT / "models/ms-marco-MiniLM-L6-v2"
METADATA = ROOT / "evaluation_data/raw/videos.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MvpPipeline:
    """Small-corpus exact cosine retrieval; no vector database or external answer fallback."""

    def __init__(self, query_understander: Callable[[str], tuple[dict[str, Any], dict[str, Any]]],
                 *, window: Path = WINDOW, metadata: Path = METADATA) -> None:
        started = time.perf_counter()
        self.query_understander = query_understander
        self.chunks = load_jsonl(window / "chunks.jsonl")
        self.metadata = {row["video_id"]: row for row in json.loads(metadata.read_text(encoding="utf-8"))}
        with np.load(window / "embedding_cache.npz", allow_pickle=False) as cache:
            assert str(cache["model_id"]) == "BAAI/bge-small-en-v1.5"
            self.vectors = cache["embeddings"]
        assert self.vectors.shape == (len(self.chunks), 384)
        self.dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, local_files_only=True)
        self.dense_model = AutoModel.from_pretrained(DENSE_MODEL, local_files_only=True).eval()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, local_files_only=True).eval()
        # ponytail: one local demo user; replace with a worker queue only if concurrent traffic becomes real.
        self.lock = threading.Lock()
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def _rank(self, english_query: str, allowed_video_ids: set[str]) -> list[dict[str, Any]]:
        if not allowed_video_ids:
            return []
        query_vector = encode(self.dense_model, self.dense_tokenizer, [english_query], query=True, batch_size=1)[0]
        scores = self.vectors @ query_vector
        allowed = [index for index, chunk in enumerate(self.chunks) if chunk["video_id"] in allowed_video_ids]
        order = sorted(allowed, key=lambda index: (-float(scores[index]), index))[:20]
        dense = [{**self.chunks[index], "rank": rank, "score": round(float(scores[index]), 8)}
                 for rank, index in enumerate(order, 1)]
        if not dense:
            return []
        ranked = reorder(dense, score_pairs(self.reranker, self.reranker_tokenizer, english_query,
                                            [candidate["text"] for candidate in dense]))
        return attach_bundles(ranked, self.chunks)

    @staticmethod
    def _gate(question: str, english_query: str, bundles: list[dict[str, Any]]):
        for batch in (bundles[:5], bundles[5:20]):
            if not batch:
                continue
            for attempt in (1, 2):
                try:
                    decision, usage = gate_pass(question, english_query, batch)
                    break
                except GateResponseError:
                    if attempt == 2:
                        raise
            if decision["answerable"]:
                return decision, usage
        return decision, usage

    def search(self, question: str, allowed_video_ids: set[str]) -> dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            query, query_usage = self.query_understander(question)
            ranked = self._rank(query["english_query"], allowed_video_ids)
            if not ranked:
                return self.no_answer(question, started, query_usage)
            decision, gate_usage = self._gate(question, query["english_query"], ranked)
        if not decision["answerable"]:
            return self.no_answer(question, started, query_usage, gate_usage)
        chosen = next(candidate for candidate in ranked if candidate["candidate_id"] == decision["best_candidate_id"])
        reason = next(label["reason"] for label in decision["candidates"]
                      if label["candidate_id"] == chosen["candidate_id"])
        video = self.metadata[chosen["video_id"]]
        result = {
            "video_id": chosen["video_id"], "platform_video_id": chosen["video_id"],
            "video_title": video["title"], "creator": video["channel"], "source_url": video["url"],
            "core_start": chosen["core_start"], "core_end": chosen["core_end"],
            "recommended_watch_start": chosen["bundle_start"],
            "recommended_watch_end": chosen["bundle_end"],
            "answerable": True, "reason": reason, "segment_id": chosen["candidate_id"],
        }
        return {"ok": True, "question": question, "answerable": True, "results": [result], "message": "",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model": "deepseek-flash", "external_answer_fallback": False}

    @staticmethod
    def no_answer(question, started, query_usage, gate_usage=None):
        return {"ok": True, "question": question, "answerable": False, "results": [],
                "message": "当前收藏的视频中，没有找到足够回答这个问题的内容。",
                "follow_up": "无需继续逐个视频查找。",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model": "deepseek-flash", "external_answer_fallback": False}
