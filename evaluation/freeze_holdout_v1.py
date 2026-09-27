"""Register the existing development baseline; seal a verified, unseen holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "evaluation_data/test_cases/test_cases.jsonl"
HOLDOUT = ROOT / "evaluation_data/holdout_v1"
PIPELINE = ROOT / "evaluation/pipeline_v1.json"
DEV_SHA = "8e4664f03da74dd21c616dc76ab797464b8ad677c91886cf75881cf8218b77e8"
TYPES = {"precise_retrieval": 6, "semantic_paraphrase": 6,
         "multi_video_competition": 4, "cross_segment": 4, "no_answer": 4}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def files_under(directory):
    return [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def check_hashes(expected):
    assert all(digest(ROOT / name) == value for name, value in expected.items()), "Frozen file drift"


def register():
    assert not PIPELINE.exists(), "Refusing to overwrite a registered pipeline"
    assert digest(DEV) == DEV_SHA, "Original development cases changed"
    old = read_json(ROOT / "evaluation_runs/reranker_v1/summary.json")["reranked"]
    dev_metrics = {"candidate_recall": old["recall_at_20"], **{key: old[key] for key in (
        "recall_at_5", "recall_at_10", "recall_at_20", "top_1", "mrr_at_20",
        "timestamp_hit", "video_recall_at_5")}}
    runtime_files = [ROOT / name for name in (
        "evaluation/run_dense_v1.py", "evaluation/run_window_v1.py", "evaluation/run_reranker_v1.py",
        "evaluation_runs/window_v1/dense_window_m/chunks.jsonl",
        "evaluation_runs/window_v1/dense_window_m/embedding_cache.npz",
        "evaluation_runs/window_v1/dense_window_m/config.json")]
    for folder in ("library", "llm", "models/bge-small-en-v1.5", "models/ms-marco-MiniLM-L6-v2"):
        runtime_files.extend(files_under(folder))
    # Model download bookkeeping is not an inference dependency.
    runtime_files = [p for p in runtime_files if ".cache" not in p.parts]
    frozen_data = []
    for folder in ("evaluation_data/raw", "evaluation_data/normalized", "evaluation_data/test_cases",
                   "evaluation_runs/baseline_v1", "evaluation_runs/query_understanding_v1",
                   "evaluation_runs/dense_v1", "evaluation_runs/window_v1",
                   "evaluation_runs/reranker_v1", "evaluation_runs/rescue_v1"):
        frozen_data.extend(files_under(folder))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "id": "pipeline_v1", "registered_at": now, "scope": "default evaluation pipeline only; product server unchanged",
        "enabled_l_rescue": False, "query_understanding": {
            "provider": "deepseek", "model": "deepseek-flash", "implementation": "library.server.understand_query",
            "input": "question only", "retrieval_field": "english_query", "temperature": 0.0,
            "prompt_and_schema": "unchanged; locked by library/server.py hash"},
        "embedding": {"model": "BAAI/bge-small-en-v1.5", "implementation": "evaluation.run_dense_v1.encode",
                      "window": "M", "cosine_similarity": True, "dense_top_k": 20, "threshold": None},
        "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                     "implementation": "evaluation.run_reranker_v1.score_pairs/reorder",
                     "max_length": 512, "batch_size": 20, "sorting": "raw score descending, stable", "score_fusion": False},
        "answer_gate": "not part of frozen ranking experiment; no new gate or rejection threshold",
        "match_rule": "same video and positive temporal overlap; inherited Window M rule, not semantic sufficiency",
        "timestamp_hit": "Top1 overlap with any sufficient gold evidence; identical to prior experiment",
        "mrr": "truncated at 20", "dev_dataset": "dev_v1", "dev_metrics": dev_metrics,
        "source_hashes": hashes(runtime_files), "frozen_data_hashes": hashes(frozen_data),
        "holdout_policy": "one frozen run; aggregate reporting only; never tune using per-case holdout results",
    }
    dev_dir = ROOT / "evaluation_data/dev_v1"
    dev_dir.mkdir(exist_ok=True)
    copied = dev_dir / "test_cases.jsonl"
    assert not copied.exists(), "Refusing to overwrite dev registration"
    shutil.copyfile(DEV, copied)
    assert digest(copied) == DEV_SHA
    save_new(dev_dir / "manifest.json", {
        "id": "dev_v1", "role": "development and regression only; not a generalization benchmark",
        "source": DEV.relative_to(ROOT).as_posix(), "sha256": DEV_SHA, "cases": len(rows(DEV)),
        "registered_at": now, "copy_byte_identical": True,
        "used_for_selection": ["Query Understanding", "embedding", "Window S/M/L", "reranker", "L Rescue"],
        "pipeline": "pipeline_v1", "metrics": dev_metrics,
    })
    save_new(PIPELINE, manifest)
    rescue_files = [ROOT / "evaluation/run_rescue_v1.py", ROOT / "evaluation/test_rescue_v1.py"]
    rescue_files += [ROOT / "evaluation_runs/window_v1/dense_window_l" / name for name in
                     ("chunks.jsonl", "embedding_cache.npz", "config.json")]
    save_new(ROOT / "evaluation/pipeline_v1_rescue.json", {
        "id": "pipeline_v1_rescue", "parent": "pipeline_v1", "parent_manifest_sha256": digest(PIPELINE),
        "default_enabled": False, "status": "retained experimental branch, excluded from holdout run",
        "variant": "rescue_20", "entrypoint": "evaluation/run_rescue_v1.py",
        "existing_results": "evaluation_runs/rescue_v1/rescue_20", "source_hashes": hashes(rescue_files),
    })
    print(json.dumps({"registered": ["dev_v1", "pipeline_v1", "pipeline_v1_rescue"],
                      "dev_byte_identical": True, "l_rescue_default": False}))


def overlap_ratio(a, b):
    if a["video_id"] != b["video_id"]:
        return 0.0
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return overlap / min(a["end"] - a["start"], b["end"] - b["start"])


def seal():
    assert not (HOLDOUT / "manifest.json").exists(), "Already sealed; do not regenerate holdout"
    pipeline = read_json(PIPELINE)
    check_hashes(pipeline["source_hashes"])
    check_hashes(pipeline["frozen_data_hashes"])
    assert digest(DEV) == DEV_SHA == digest(ROOT / "evaluation_data/dev_v1/test_cases.jsonl")
    tests, questions = rows(HOLDOUT / "private_ground_truth.jsonl"), rows(HOLDOUT / "questions.jsonl")
    report = read_json(HOLDOUT / "verification_report.json")
    assert report.get("approved") is True, "Independent verifier must explicitly approve before sealing"
    assert len(tests) == len(questions) == 24
    assert dict(Counter(t["type"] for t in tests)) == TYPES
    assert len({t["id"] for t in tests}) == 24
    assert all(set(q) == {"id", "question"} for q in questions), "Questions must not reveal types or gold"
    assert questions == [{"id": t["id"], "question": t["question"]} for t in tests]
    dev = rows(DEV)
    assert not ({t["question"] for t in tests} & {t["question"] for t in dev})
    corpus = {p.stem: read_json(p)["segments"] for p in (ROOT / "evaluation_data/normalized").glob("*.json")}
    gold_counts, ratios = Counter(), []
    for test in tests:
        assert isinstance(test["answerable"], bool)
        assert test["answerable"] == (test["type"] != "no_answer") == bool(test["gold_evidence"])
        assert set(test["video_relevance"]) == set(corpus)
        assert set(test["video_relevance"].values()) <= {"sufficient", "partial", "mention", "none"}
        assert {g["video_id"] for g in test["gold_evidence"]} == {
            video for video, relevance in test["video_relevance"].items() if relevance == "sufficient"}
        gold_counts.update({g["video_id"] for g in test["gold_evidence"]})
        for gold in test["gold_evidence"]:
            assert gold["start"] < gold["end"]
            ids = gold["segment_ids"]
            assert ids == list(range(min(ids), max(ids) + 1)), "Evidence must identify contiguous source segments"
            source = corpus[gold["video_id"]]
            assert 1 <= min(ids) <= max(ids) <= len(source)
            assert abs(gold["start"] - source[ids[0] - 1]["start"]) < 0.001
            assert abs(gold["end"] - source[ids[-1] - 1]["end"]) < 0.001
            if test["type"] == "cross_segment":
                assert len(ids) > 1
            ratios.extend(overlap_ratio(gold, prior) for old in dev for prior in old["gold_evidence"])
    assert max(ratios, default=0) <= 0.20 + 1e-10, "Dev evidence overlap exceeds preregistered limit"
    manifest = {
        "id": "holdout_v1", "frozen": True, "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "questions_sha256": digest(HOLDOUT / "questions.jsonl"),
        "private_ground_truth_sha256": digest(HOLDOUT / "private_ground_truth.jsonl"),
        "verification_report_sha256": digest(HOLDOUT / "verification_report.json"),
        "pipeline_manifest_sha256": digest(PIPELINE), "dev_sha256": DEV_SHA,
        "evaluation_code_sha256": hashes([Path(__file__), ROOT / "evaluation/run_holdout_v1.py",
                                            ROOT / "evaluation/test_holdout_v1.py"]),
        "cases": len(tests), "answerable_cases": 20, "type_counts": dict(Counter(t["type"] for t in tests)),
        "gold_cases_by_video": dict(gold_counts), "overlap_rule": "intersection / min(new gold duration, dev gold duration) <= 0.20",
        "maximum_dev_gold_overlap_ratio": max(ratios, default=0),
        "overlap_violations": sum(r > 0.20 + 1e-10 for r in ratios),
        "semantic_intent_review": "independent source-only verifier; no system results used",
        "visibility": "runtime: questions only; evaluator: private ground truth; reporting: aggregate only",
        "limitation": "new knowledge intents within the same four videos, not unseen-video generalization",
    }
    save_new(HOLDOUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "seal", "check"))
    action = parser.parse_args().action
    if action == "register":
        register()
    elif action == "seal":
        seal()
    else:
        check_hashes(read_json(PIPELINE)["source_hashes"])
        check_hashes(read_json(PIPELINE)["frozen_data_hashes"])
        assert digest(DEV) == DEV_SHA == digest(ROOT / "evaluation_data/dev_v1/test_cases.jsonl")
        sealed = read_json(HOLDOUT / "manifest.json")
        check_hashes(sealed["evaluation_code_sha256"])
        assert digest(PIPELINE) == sealed["pipeline_manifest_sha256"]
        assert digest(HOLDOUT / "questions.jsonl") == sealed["questions_sha256"]
        assert digest(HOLDOUT / "private_ground_truth.jsonl") == sealed["private_ground_truth_sha256"]
        assert digest(HOLDOUT / "verification_report.json") == sealed["verification_report_sha256"]
        print("Frozen original data, pipeline, dev and holdout hashes: unchanged")
