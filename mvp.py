"""Initialize and run the real four-video MVP Demo Collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "library"))
from library import server  # noqa: E402
from library.guided_pipeline import GuidedPipeline  # noqa: E402


DATA_ROOT = ROOT / ".mvp-demo"
METADATA = ROOT / "evaluation_data/raw/videos.json"
PRESETS = [
    {"label": "容易命中", "question": "MCP 的出现源于 Claude Desktop 和 IDE 之间什么重复操作？"},
    {"label": "跨片段", "question": "一次 MCP 连接如何从服务器声明能力，走到客户端发请求并得到结构化结果？"},
    {"label": "明确无答案", "question": "MCP 规范是否要求所有服务器都必须使用 OAuth 2.1？"},
]


def demo_library():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    return {
        "schema_version": 2, "title": "MCP Demo Collection",
        "collections": [{"collection_id": "demo-mcp", "title": "MCP Demo Collection",
                         "video_ids": [row["video_id"] for row in metadata]}],
        "videos": [{"video_id": row["video_id"], "platform": "youtube",
                    "platform_video_id": row["video_id"], "source_url": row["url"],
                    "title": row["title"], "creator": row["channel"], "duration": row["duration"],
                    "subtitle_type": row["subtitle_type"], "segments": []} for row in metadata],
        "demo_questions": PRESETS,
    }


def initialize():
    path = DATA_ROOT / "library.json"
    if not path.exists():
        server.atomic_write_json(path, demo_library())
    else:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("demo_questions") != PRESETS:
            current["demo_questions"] = PRESETS
            server.atomic_write_json(path, current)
    return path


def check():
    library = json.loads(initialize().read_text(encoding="utf-8"))
    assert library["schema_version"] == 2
    assert len(library["videos"]) == len(library["collections"][0]["video_ids"]) == 4
    assert [row["label"] for row in library["demo_questions"]] == ["容易命中", "跨片段", "明确无答案"]
    assert all(video["source_url"].startswith("https://www.youtube.com/watch?v=") for video in library["videos"])
    assert all((ROOT / path).is_file() for path in (
        "models/bge-small-en-v1.5/model.safetensors",
        "models/bge-small-en-v1.5/tokenizer.json",
        "models/ms-marco-MiniLM-L6-v2/model.safetensors",
        "models/ms-marco-MiniLM-L6-v2/tokenizer.json",
        "evaluation_runs/window_v1/dense_window_m/embedding_cache.npz",
    )), "MVP local model or index artifact is missing"
    print("MVP Demo Collection check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="Initialize Demo Collection and exit")
    parser.add_argument("--check", action="store_true", help="Validate Demo Collection without model/API calls")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: auto)")
    parser.add_argument("--collection", choices=("mcp", "english", "lee2021", "mixed50"), default="mcp")
    parser.add_argument("--intent-v2", action="store_true", help="Use v2 intent planning for mixed50 (v1 remains default)")
    parser.add_argument("--chapters", action="store_true", help="Use the independent eight-video content-chapter pilot")
    args = parser.parse_args()
    if args.intent_v2 and args.collection != "mixed50":
        parser.error("--intent-v2 currently requires --collection mixed50")
    if args.chapters and (args.collection != "mixed50" or args.intent_v2):
        parser.error("--chapters requires --collection mixed50 and cannot be combined with --intent-v2")
    if args.chapters:
        from evaluation.chapter_pilot import check as check_chapters, library as chapter_library
        from evaluation.mixed50_v1 import MixedPipeline, save
        from library.chapter_pipeline import ChapterPipeline
        from library.ingestion import Ingestion
        state = ROOT / 'runtime/chapter_pilot_v1'
        check_chapters()
        if args.check or args.init:
            return 0
        save(state/'library.json', chapter_library())
        pipeline = ChapterPipeline(MixedPipeline(server.understand_query), state/'traces')
        pipeline = Ingestion(state/'live', pipeline)
        return server.run_server(state, pipeline=pipeline,
                                 port=args.port,open_browser=not args.no_open)
    if args.collection == "mixed50":
        from evaluation.mixed50_v1 import check as check_mixed, STATE, library, save, MixedPipeline
        check_mixed()
        if args.check or args.init:
            return 0
        save(STATE / "library.json", library())
        pipeline = MixedPipeline(server.understand_query)
        if args.intent_v2:
            from library.intent_pipeline_v2 import IntentPipeline
            pipeline = IntentPipeline(pipeline, STATE / "intent_v2")
        else:
            pipeline = GuidedPipeline(pipeline)
        return server.run_server(STATE, pipeline=pipeline,
                                 port=args.port, open_browser=not args.no_open)
    if args.collection == "lee2021":
        from evaluation.lee2021_v1 import check as check_lee, STATE, library, save, LeePipeline
        check_lee()
        if args.check or args.init:
            return 0
        save(STATE / "library.json", library())
        return server.run_server(STATE, pipeline=GuidedPipeline(LeePipeline(server.understand_query)),
                                 port=args.port, open_browser=not args.no_open)
    if args.collection == "english":
        from evaluation.english_v1 import check as check_english, DATA, INDEX, STATE, library, save
        if args.check or args.init:
            check_english()
            return 0
        from library.precise_pipeline import PrecisePipeline
        check_english()
        save(STATE / "library.json", library())
        pipeline = PrecisePipeline(server.understand_query, window=INDEX, metadata=DATA / "videos.json",
                                   normalized=DATA / "normalized")
        return server.run_server(STATE, pipeline=GuidedPipeline(pipeline), port=args.port, open_browser=not args.no_open)
    path = initialize()
    if args.init:
        print(path)
        return 0
    if args.check:
        check()
        return 0
    return server.run_server(DATA_ROOT, port=args.port, open_browser=not args.no_open, mvp=True)


if __name__ == "__main__":
    raise SystemExit(main())
