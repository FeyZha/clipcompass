"""Explicitly requested mixed-topic collection. No fetch occurs at startup/check."""

import argparse
import json
import math
import random
import sys
import time

from evaluation.english_v1 import ROOT, read, save, digest, parse_vtt

DATA = ROOT / "evaluation_data/mixed50_v1"
INDEX = DATA / "index"
STATE = ROOT / "runtime/mixed50_v1"
TOPICS = {
    "brain_sleep": "TED-Ed sleep memory brain",
    "food_science": "TED-Ed food cooking fermentation chocolate bread",
    "history": "TED-Ed ancient civilization Rome Egypt history",
    "economics": "TED-Ed economics money inflation banks trade",
    "space_physics": "TED-Ed space gravity black holes stars",
    "environment": "TED-Ed climate ocean forests ecosystem",
}
EXISTING = (
    ("mcp", ROOT / "evaluation_data/raw/videos.json", ROOT / "evaluation_data/normalized", ROOT / "evaluation_runs/window_v1/dense_window_m"),
    ("english", ROOT / "evaluation_data/english_v1/videos.json", ROOT / "evaluation_data/english_v1/normalized", ROOT / "evaluation_data/english_v1/index"),
    ("deep_learning", ROOT / "evaluation_data/lee2021_v1/videos.json", ROOT / "evaluation_data/lee2021_v1/normalized", ROOT / "evaluation_data/lee2021_v1/index"),
)


def fetch():
    """Fetch public metadata and one English caption track, never audio/video."""
    sys.path.insert(0, str(ROOT / "work/caption-tools"))
    import yt_dlp
    for topic, query in TOPICS.items():
        complete = DATA / "sources" / f"{topic}.json"
        if complete.exists():
            continue
        discovery = DATA / "discovery" / f"{topic}.json"
        if not discovery.exists():
            with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True,
                                     "socket_timeout": 20}) as downloader:
                entries = downloader.extract_info(f"ytsearch10:{query}", download=False)["entries"]
            save(discovery, [{k: row.get(k) for k in ("id", "title", "channel")} for row in entries])
        used = {v["video_id"] for path in (DATA / "sources").glob("*.json") for v in read(path)}
        chosen = []
        for row in read(discovery):
            if row["id"] in used or row["channel"] not in ("TED-Ed", "TED"):
                continue
            video_id = row["id"]
            cached = DATA / "fetched" / f"{video_id}.json"
            if cached.exists():
                entry = read(cached)
            else:
                options = {"quiet": True, "skip_download": True, "writesubtitles": True,
                           "subtitlesformat": "vtt", "outtmpl": str(DATA / "raw/%(id)s.%(ext)s"),
                           "socket_timeout": 20, "retries": 1, "noplaylist": True}
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    if not info.get("duration") or info["duration"] > 1800:
                        continue
                    tracks = info.get("subtitles", {})
                    language = next((lang for lang in ("en", "en-US", "en-GB") if lang in tracks), None)
                    if not language:
                        print(f"No publisher English captions: {video_id}", flush=True)
                        continue
                    downloader.params["subtitleslangs"] = [language]
                    downloader.process_ie_result(info, download=True)
                caption = DATA / "raw" / f"{video_id}.{language}.vtt"
                cues = parse_vtt(caption.read_text(encoding="utf-8"), info["duration"])
                entry = {"video_id": video_id, "title": info["title"], "channel": info["channel"],
                         "url": f"https://www.youtube.com/watch?v={video_id}", "duration": info["duration"],
                         "subtitle_language": language, "subtitle_type": "manual",
                         "subtitle_source": f"youtube:{language}", "subtitle_sha256": digest(caption)}
                save(DATA / "normalized" / f"{video_id}.json", {**entry, "segments": cues})
                save(cached, entry)
                time.sleep(2)
            chosen.append(entry)
            used.add(video_id)
            print(f"{topic}: {len(chosen)}/5 {entry['title']}", flush=True)
            if len(chosen) == 5:
                break
        if len(chosen) != 5:
            raise RuntimeError(f"Only {len(chosen)} eligible videos for {topic}; review sources before continuing")
        save(complete, chosen)


def build():
    from evaluation.lee2021_v1 import save_lines
    from evaluation.run_window_v1 import build_windows, SPECS
    from library.mvp_pipeline import load_jsonl, np, encode, AutoModel, AutoTokenizer, DENSE_MODEL
    videos, sources, chunks, vectors, labels = [], [], [], [], {}
    for topic, metadata, normalized, index in EXISTING:
        for video in read(metadata):
            videos.append(video)
            labels[video["video_id"]] = topic
            save(DATA / "normalized" / f"{video['video_id']}.json", read(normalized / f"{video['video_id']}.json"))
        current = load_jsonl(index / "chunks.jsonl")
        chunks.extend(current)
        sources.extend(load_jsonl(index / "source_chunks.jsonl") if (index / "source_chunks.jsonl").exists() else current)
        with np.load(index / "embedding_cache.npz", allow_pickle=False) as cache:
            assert str(cache["model_id"]) == "BAAI/bge-small-en-v1.5"
            assert cache["embeddings"].shape == (len(current), 384)
            vectors.append(cache["embeddings"].copy())
    corpus = []
    for topic in TOPICS:
        rows = read(DATA / "sources" / f"{topic}.json")
        assert len(rows) == 5
        for video in rows:
            videos.append(video)
            labels[video["video_id"]] = topic
            cues = read(DATA / "normalized" / f"{video['video_id']}.json")["segments"]
            corpus.extend({**cue, "video_id": video["video_id"], "segment_id": i} for i, cue in enumerate(cues, 1))
    assert len(videos) == len({v["video_id"] for v in videos}) == 50
    added = build_windows(corpus, "m", SPECS["m"])
    chunks.extend(added)
    sources.extend(added)
    random.Random(20260927).shuffle(videos)
    save(DATA / "videos.json", videos)
    # Audit labels are never passed to the index, runtime library or query model.
    save(DATA / "audit_topics.json", labels)
    save_lines(INDEX / "chunks.jsonl", chunks)
    save_lines(INDEX / "source_chunks.jsonl", sources)
    cache_path = INDEX / "embedding_cache.npz"
    if not cache_path.exists():
        model = AutoModel.from_pretrained(DENSE_MODEL, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, local_files_only=True)
        vectors.append(encode(model, tokenizer, [c["text"] for c in added], batch_size=16))
        with cache_path.open("xb") as handle:
            np.savez_compressed(handle, embeddings=np.concatenate(vectors),
                                model_id=np.array("BAAI/bge-small-en-v1.5"),
                                chunks_sha256=np.array(digest(INDEX / "chunks.jsonl")))
    check()


def check():
    from library.mvp_pipeline import load_jsonl, np
    videos = read(DATA / "videos.json")
    assert len(videos) == len({v["video_id"] for v in videos}) == 50
    source, retrieval = load_jsonl(INDEX / "source_chunks.jsonl"), load_jsonl(INDEX / "chunks.jsonl")
    assert len(source) == len(retrieval) and len({c["chunk_id"] for c in source}) == len(source)
    for a, b in zip(source, retrieval):
        assert (a["chunk_id"], a["video_id"], a["start"], a["end"]) == (b["chunk_id"], b["video_id"], b["start"], b["end"])
        assert a["text"].strip() and b["text"].strip()
        assert "topic" not in b
    assert {c["video_id"] for c in source} == {v["video_id"] for v in videos}
    for video in videos:
        cues = read(DATA / "normalized" / f"{video['video_id']}.json")["segments"]
        # Frozen MCP captions include a final music cue ending 2.229s past platform duration.
        # Preserve that source verbatim; no such tolerance applies to newly fetched captions.
        caption_limit = 6253.229 if video["video_id"] == "kQmXtrmQ5Zg" else video["duration"]
        assert cues and all(math.isfinite(c["start"]) and math.isfinite(c["end"]) and 0 <= c["start"] < c["end"] <= caption_limit for c in cues)
        own = [c for c in source if c["video_id"] == video["video_id"]]
        assert all(any(c["start"] <= cue["start"] < cue["end"] <= c["end"] for c in own) for cue in cues)
    for path in (DATA / "fetched").glob("*.json"):
        video = read(path)
        assert digest(DATA / "raw" / f"{video['video_id']}.{video['subtitle_language']}.vtt") == video["subtitle_sha256"]
    with np.load(INDEX / "embedding_cache.npz", allow_pickle=False) as cache:
        assert str(cache["chunks_sha256"]) == digest(INDEX / "chunks.jsonl")
        assert cache["embeddings"].shape == (len(source), 384)
        assert np.isfinite(cache["embeddings"]).all()
        assert np.allclose(np.linalg.norm(cache["embeddings"], axis=1), 1, atol=1e-5)
    print(f"Mixed collection: 50 videos, {len(source)} windows, {sum(v['duration'] for v in videos)/60:.2f} minutes", flush=True)


def library():
    videos = read(DATA / "videos.json")
    title = "随手收藏 · 50 个视频 · 多主题混合"
    return {"schema_version": 2, "title": title,
            "collections": [{"collection_id": "mixed50-v1", "title": title, "video_ids": [v["video_id"] for v in videos]}],
            "videos": [{"video_id": v["video_id"], "platform": "youtube", "platform_video_id": v["video_id"],
                        "source_url": v["url"], "title": v["title"], "creator": v["channel"], "duration": v["duration"],
                        "subtitle_type": v["subtitle_type"], "segments": []} for v in videos],
            "demo_questions": [{"label": "学习报告", "question": "我想了解睡眠和记忆的关系"},
                               {"label": "混合收藏", "question": "我需要学习梯度下降相关的知识"},
                               {"label": "范围边界", "question": "我想学习 LoRA 低秩适配"}]}


from evaluation.lee2021_v1 import LeePipeline
from library.precise_pipeline import PrecisePipeline
from library.mvp_pipeline import load_jsonl


class MixedPipeline(LeePipeline):
    def __init__(self, query_understander):
        PrecisePipeline.__init__(self, query_understander, window=INDEX,
                                 metadata=DATA / "videos.json", normalized=DATA / "normalized")
        self.source_chunks = load_jsonl(INDEX / "source_chunks.jsonl")
        self.source_by_id = {c["chunk_id"]: c for c in self.source_chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build", "check"))
    globals()[parser.parse_args().action]()
