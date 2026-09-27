"""Independent English collection: frozen inputs, unchanged MVP, real HTTP evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import html
import json
import math
import re
from pathlib import Path
import sys
import threading
import time
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "evaluation_data/english_v1"
INDEX = DATA / "index"
RUN = ROOT / "evaluation_runs/english_v1"
STATE = ROOT / "runtime/english_v1"
SOURCES = (
    ("_Mv7fBqauvc", "en-GB", "manual"), ("xdodZEttvSY", "en-GB", "manual"),
    ("79zqFG7zdnA", "en-GB", "manual"), ("pvoqkQHb3lo", "en-GB", "manual"),
    ("FVmVP9CCRcU", "en", "manual"), ("7XlQWzdhsPA", "en-orig", "auto"),
    ("8CiA9BCRPBk", "en", "manual"), ("I9CZ-nj9joM", "en", "manual"),
)
STAMP = r"\d{2}:\d{2}:\d{2}\.\d{3}"
TIMING = re.compile(rf"^({STAMP}) --> ({STAMP})")
INLINE = re.compile(rf"<{STAMP}>")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise FileExistsError(f"Refusing to overwrite: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def seconds(stamp):
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(text, duration):
    """Preserve manual cues; remove YouTube rolling copies, not genuine repetitions."""
    rolling = bool(INLINE.search(text))
    segments = []
    previous_display = []
    for block in re.split(r"\n\n+", text.replace("\r\n", "\n")):
        lines = block.splitlines()
        found = next(((i, TIMING.match(line)) for i, line in enumerate(lines) if TIMING.match(line)), None)
        if not found:
            continue
        index, match = found
        start, end = map(seconds, match.groups())
        body = lines[index + 1:]
        displayed = [html.unescape(re.sub(r"<[^>]*>", "", line)).strip() for line in body if line.strip()]
        if rolling:
            tagged = [line for line in body if INLINE.search(line)]
            if tagged:
                body = tagged
            else:
                overlap = max((k for k in range(1, min(len(displayed), len(previous_display)) + 1)
                               if previous_display[-k:] == displayed[:k]), default=0)
                body = displayed[overlap:]
            previous_display = displayed
        words = html.unescape(re.sub(r"<[^>]*>", "", " ".join(body)))
        words = " ".join(words.split())
        if not words or re.fullmatch(r"\[(?:Music|Applause|writing)\]", words, re.I):
            continue
        # YouTube's final caption can outlast the video; bound playback to its real duration.
        if not (0 <= start < duration and start < end <= duration + 5):
            raise ValueError(f"Invalid caption interval: {start}, {end}, {duration}")
        segments.append({"start": start, "end": min(end, duration), "text": words})
    segments.sort(key=lambda row: (row["start"], row["end"]))
    if not segments:
        raise ValueError("No usable timestamped captions")
    return segments


def prepare():
    from evaluation.run_window_v1 import SPECS, build_windows
    metadata, corpus = [], []
    for video_id, language, kind in SOURCES:
        info = read(DATA / "raw" / f"{video_id}.info.json")
        caption = DATA / "raw" / f"{video_id}.{language}.vtt"
        assert info["id"] == video_id and info["duration"] > 0
        segments = parse_vtt(caption.read_text(encoding="utf-8"), info["duration"])
        entry = {"video_id": video_id, "title": info["title"], "channel": info["channel"],
                 "url": f"https://www.youtube.com/watch?v={video_id}", "duration": info["duration"],
                 "subtitle_language": language, "subtitle_type": kind,
                 "subtitle_source": f"youtube:{language}", "subtitle_sha256": digest(caption)}
        metadata.append(entry)
        save(DATA / "normalized" / f"{video_id}.json", {**entry, "segments": segments})
        corpus.extend({**segment, "video_id": video_id, "segment_id": i}
                      for i, segment in enumerate(segments, 1))
    save(DATA / "videos.json", metadata)
    chunks = build_windows(corpus, "m", SPECS["m"])
    INDEX.mkdir(parents=True, exist_ok=True)
    chunks_path = INDEX / "chunks.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
    if chunks_path.exists():
        assert chunks_path.read_text(encoding="utf-8") == content, "Index input conflict"
    else:
        with chunks_path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(json.dumps({"videos": len(metadata), "minutes": sum(v["duration"] for v in metadata) / 60,
                      "segments": len(corpus), "windows": len(chunks)}, ensure_ascii=False))


def build():
    from library.mvp_pipeline import DENSE_MODEL, AutoModel, AutoTokenizer, encode, np, load_jsonl
    if (INDEX / "embedding_cache.npz").exists():
        raise FileExistsError("Index already exists")
    chunks = load_jsonl(INDEX / "chunks.jsonl")
    model = AutoModel.from_pretrained(DENSE_MODEL, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, local_files_only=True)
    vectors = encode(model, tokenizer, [row["text"] for row in chunks], batch_size=16)
    with (INDEX / "embedding_cache.npz").open("xb") as handle:
        np.savez_compressed(handle, embeddings=vectors, model_id=np.array("BAAI/bge-small-en-v1.5"),
                            chunks_sha256=np.array(digest(INDEX / "chunks.jsonl")))
    print(f"Built {len(vectors)} local vectors", flush=True)


def library():
    metadata = read(DATA / "videos.json")
    return {"schema_version": 2, "title": "英语学习 · 8 视频收藏集",
            "collections": [{"collection_id": "english-v1", "title": "英语学习 · 8 视频收藏集",
                             "video_ids": [v["video_id"] for v in metadata]}],
            "videos": [{"video_id": v["video_id"], "platform": "youtube",
                        "platform_video_id": v["video_id"], "source_url": v["url"],
                        "title": v["title"], "creator": v["channel"], "duration": v["duration"],
                        "subtitle_type": v["subtitle_type"], "segments": []} for v in metadata],
            "demo_questions": [
                {"label": "语法区别", "question": "for three hours 和 since five o'clock 的时间表达有什么区别？"},
                {"label": "实际表达", "question": "想礼貌地问同事报告写好没有，为什么可以用 I was wondering？"},
                {"label": "库内未讲", "question": "英语的 /θ/ 和 /ð/ 发音时舌头应该放在哪里？"},
            ]}


def check():
    from library.mvp_pipeline import load_jsonl, np
    videos = read(DATA / "videos.json")
    assert len(videos) == len({v["video_id"] for v in videos}) == 8
    chunks = load_jsonl(INDEX / "chunks.jsonl")
    assert {c["video_id"] for c in chunks} == {v["video_id"] for v in videos}
    with np.load(INDEX / "embedding_cache.npz", allow_pickle=False) as cache:
        assert cache["embeddings"].shape == (len(chunks), 384)
        assert np.isfinite(cache["embeddings"]).all()
        assert str(cache["chunks_sha256"]) == digest(INDEX / "chunks.jsonl")
        assert np.allclose(np.linalg.norm(cache["embeddings"], axis=1), 1, atol=1e-5)
    for v in videos:
        path = DATA / "raw" / f"{v['video_id']}.{v['subtitle_language']}.vtt"
        assert digest(path) == v["subtitle_sha256"]
    print("English collection: 8 videos, captions and index verified", flush=True)


def freeze():
    check()
    cases = read(DATA / "cases.json")
    assert len(cases) == len({c["id"] for c in cases}) == 24
    assert sum(c["answerable"] for c in cases) == 16
    for c in cases:
        assert bool(c["gold_evidence"]) == c["answerable"]
        for evidence in c["gold_evidence"]:
            segments = read(DATA / "normalized" / f"{evidence['video_id']}.json")["segments"]
            assert any(s["start"] < evidence["end"] and s["end"] > evidence["start"] for s in segments)
    save(DATA / "questions.json", [{"id": c["id"], "question": c["question"]} for c in cases])
    paths = [DATA / "cases.json", DATA / "questions.json", DATA / "videos.json",
             INDEX / "chunks.jsonl", INDEX / "embedding_cache.npz", ROOT / "library/mvp_pipeline.py",
             ROOT / "library/server.py", ROOT / "llm/answer_gate.py", ROOT / "llm/answer_gate_bundle.py",
             ROOT / "llm/provider.py", ROOT / "llm/deepseek.py", Path(__file__)]
    save(DATA / "manifest.json", {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "English domain transfer test; agent-authored labels, not independent human validation",
        "questions": 24, "answerable": 16, "no_answer": 8, "tuning": False,
        "gold_visible_to_inference": False,
        "hashes": {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}})
    print("24 questions and evidence labels frozen before inference", flush=True)


def verify_frozen():
    for name, expected in read(DATA / "manifest.json")["hashes"].items():
        assert digest(ROOT / name) == expected, f"Frozen input changed: {name}"


def run():
    from library import server, mvp_pipeline as runtime
    verify_frozen()
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "config.json", {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "manifest_sha256": digest(DATA / "manifest.json"), "transport": "HTTP"})
    questions = read(DATA / "questions.json")  # No gold file is read until every prediction is saved.
    pipeline = runtime.MvpPipeline(server.understand_query, window=INDEX, metadata=DATA / "videos.json")
    save(STATE / "library.json", library())
    app = server.AppServer(("127.0.0.1", 0), server.LibraryStore(STATE), server.STATIC_ROOT, pipeline)
    worker = threading.Thread(target=app.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{app.server_address[1]}"
    trace = {}
    original_query, original_reorder, original_gate = pipeline.query_understander, runtime.reorder, runtime.gate_pass

    def query(question):
        value, usage = original_query(question)
        trace.update(query=value, query_usage=usage)
        return value, usage

    def reorder(candidates, scores):
        result = original_reorder(candidates, scores)
        trace.update(dense=candidates, reranked=result)
        return result

    def gate(question, english, candidates):
        call = {"candidate_ids": [c["candidate_id"] for c in candidates]}
        trace.setdefault("gate_calls", []).append(call)
        decision, usage = original_gate(question, english, candidates)
        call.update(decision=decision, usage=usage)
        return decision, usage

    pipeline.query_understander = query
    try:
        with patch.object(runtime, "reorder", reorder), patch.object(runtime, "gate_pass", gate):
            for question in questions:
                trace = {"id": question["id"], "question": question["question"]}
                body = json.dumps({"question": question["question"], "collection_id": "english-v1"}).encode()
                request = urllib.request.Request(base + "/api/search", body, {"Content-Type": "application/json"})
                started = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=360) as response:
                        trace["response"] = json.load(response)
                    trace["error"] = None
                except Exception as exc:
                    trace["error"] = f"{type(exc).__name__}: {exc}"
                trace["http_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                with (RUN / "results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                print(f"{question['id']}: answerable={trace.get('response', {}).get('answerable')}, error={trace['error']}", flush=True)
                if trace["error"]:
                    raise RuntimeError("Inference failed; partial results preserved, do not silently retry a frozen run")
    finally:
        app.shutdown()
        app.server_close()
        worker.join()
    verify_frozen()
    evaluate()


def coverage(candidate, evidence, bundle=False):
    if candidate["video_id"] != evidence["video_id"]:
        return 0
    start = candidate["recommended_watch_start"] if bundle else candidate.get("start", candidate.get("core_start"))
    end = candidate["recommended_watch_end"] if bundle else candidate.get("end", candidate.get("core_end"))
    return max(0, min(end, evidence["end"]) - max(start, evidence["start"])) / (evidence["end"] - evidence["start"])


def evaluate():
    from library.mvp_pipeline import load_jsonl, np
    verify_frozen()
    predictions = load_jsonl(RUN / "results.jsonl")
    cases = {c["id"]: c for c in read(DATA / "cases.json")}
    assert len(predictions) == len(cases) == 24
    counts, details, latencies, usages = Counter(), [], [], []
    for row in predictions:
        gold = cases[row["id"]]
        response = row.get("response", {})
        item = next(iter(response.get("results", [])), None)
        accepted = bool(response.get("answerable"))
        strict = bool(item and any(coverage(item, e, True) >= .9 for e in gold["gold_evidence"]))
        counts["answerable" if gold["answerable"] else "no_answer"] += 1
        counts["accepted"] += accepted
        counts["strict_correct"] += strict
        counts["correct_rejection"] += not accepted and not gold["answerable"] and not row["error"]
        counts["false_accept"] += accepted and not gold["answerable"]
        counts["false_reject"] += not accepted and gold["answerable"]
        counts["core_overlaps_gold"] += bool(item and any(coverage(item, e) > 0 for e in gold["gold_evidence"]))
        for stage in ("dense", "reranked"):
            for k in (1, 5, 20):
                counts[f"{stage}_hit_at_{k}"] += gold["answerable"] and any(
                    coverage(c, e) > 0 for c in row.get(stage, [])[:k] for e in gold["gold_evidence"])
        if item:
            assert 0 <= item["recommended_watch_start"] <= item["core_start"] < item["core_end"] <= item["recommended_watch_end"]
            assert item["source_url"] == f"https://www.youtube.com/watch?v={item['video_id']}"
        assert response.get("external_answer_fallback") is False
        usages.extend([row["query_usage"], *[c["usage"] for c in row.get("gate_calls", []) if "usage" in c]])
        latencies.append(row["http_latency_ms"])
        details.append({"id": row["id"], "question": row["question"], "expected_answerable": gold["answerable"],
                        "accepted": accepted, "strict_evidence_match": strict, "result": item})
    report = {"videos": 8, "questions": len(predictions), "counts": dict(counts),
        "answerable_recall": counts["strict_correct"] / counts["answerable"],
        "answerable_precision": counts["strict_correct"] / counts["accepted"] if counts["accepted"] else None,
        "no_answer_accuracy": counts["correct_rejection"] / counts["no_answer"],
        "stage_recall": {key: value / counts["answerable"] for key, value in counts.items() if "_hit_at_" in key},
        "mean_end_to_end_ms": sum(latencies) / len(latencies), "p95_end_to_end_ms": float(np.percentile(latencies, 95)),
        "api_calls": len(usages), "input_tokens": sum(u["input_tokens"] for u in usages),
        "output_tokens": sum(u["output_tokens"] for u in usages),
        "metric_rule": "Accepted bundle covers >=90% of one pre-labelled sufficient interval; alternate valid evidence may be undercounted.",
        "limitations": "Agent-authored small English-grammar corpus; not independent human blind validation. No algorithm tuning.",
        "source_integrity_verified": True, "cases": details}
    save(RUN / "summary.json", report)
    lines = ["# 英语学习八视频跨主题测试", "", f"24 题：16 个可回答，8 个库内无答案。",
        f"- 严格答案召回率：{report['answerable_recall']:.1%}",
        f"- 严格答案精确率：{report['answerable_precision']:.1%}",
        f"- 无答案判断正确率：{report['no_answer_accuracy']:.1%}",
        f"- 平均完整请求耗时：{report['mean_end_to_end_ms']/1000:.2f} 秒；P95：{report['p95_end_to_end_ms']/1000:.2f} 秒。",
        "", "标准答案在运行前冻结；模型只接收问题及召回字幕，不读取标准答案。", "",
        "这是代理根据字幕标注的小样本跨主题测试，不是独立人工盲测。未根据测试结果调参。",
        "时间段严格匹配可能低估未预先列出的等价答案。", "", "## 视频来源", ""]
    lines += [f"- [{v['title']}]({v['url']}) — {v['channel']}，{v['duration']/60:.1f} 分钟，{v['subtitle_type']} 字幕"
              for v in read(DATA / "videos.json")]
    with (RUN / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False), flush=True)


def serve(open_browser=True):
    from library import server
    from library.mvp_pipeline import MvpPipeline
    check()
    save(STATE / "library.json", library())
    pipeline = MvpPipeline(server.understand_query, window=INDEX, metadata=DATA / "videos.json")
    return server.run_server(STATE, pipeline=pipeline, open_browser=open_browser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "check", "freeze", "run", "evaluate"))
    globals()[parser.parse_args().action]()
