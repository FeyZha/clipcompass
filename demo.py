"""Create and run an isolated synthetic demo of the saved-video knowledge search."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "library"
if str(LIBRARY) not in sys.path:
    sys.path.insert(0, str(LIBRARY))

import server  # noqa: E402


DEMO_ROOT = ROOT / ".demo"
DEMO_LIBRARY = {
    "schema_version": 1,
    "title": "我的视频知识库",
    "collections": [
        {
            "collection_id": "saved",
            "title": "已收藏",
            "video_ids": ["youtube:demo-pi-agent", "youtube:demo-pi-tutorial"],
        }
    ],
    "videos": [
        {
            "video_id": "youtube:demo-pi-agent",
            "platform": "youtube",
            "platform_video_id": "demo-pi-agent",
            "source_url": "https://www.youtube.com/watch?v=demo-pi-agent",
            "title": "Pi Agent：从工具调用到运行时架构",
            "creator": "合成演示作者",
            "view_count": 18000,
            "creator_followers": 4200,
            "saved_at": "2026-09-26T00:00:00+00:00",
            "segments": [
                {
                    "segment_id": "segment-001",
                    "start_seconds": 45,
                    "end_seconds": 170,
                    "title": "基础使用方式",
                    "summary": "介绍如何启动 Pi Agent 和提交第一个任务。",
                    "transcript": "这一段演示 Pi Agent 的安装、启动和基础使用。",
                },
                {
                    "segment_id": "segment-002",
                    "start_seconds": 310,
                    "end_seconds": 620,
                    "title": "Agent 运行时架构",
                    "summary": "解释模型、工具注册、上下文循环与任务执行器之间的关系。",
                    "transcript": "Pi Agent 架构由模型调用层、工具注册表、上下文循环和任务执行器组成。运行时会把工具结果重新写入上下文，再决定下一步操作。",
                },
            ],
        },
        {
            "video_id": "youtube:demo-pi-tutorial",
            "platform": "youtube",
            "platform_video_id": "demo-pi-tutorial",
            "source_url": "https://www.youtube.com/watch?v=demo-pi-tutorial",
            "title": "十分钟上手 Pi Agent",
            "creator": "合成热门作者",
            "view_count": 320000,
            "creator_followers": 90000,
            "saved_at": "2026-09-26T00:00:00+00:00",
            "segments": [
                {
                    "segment_id": "segment-001",
                    "start_seconds": 20,
                    "end_seconds": 540,
                    "title": "安装与常用操作",
                    "summary": "完整演示安装、登录和日常使用步骤。",
                    "transcript": "介绍 Pi Agent 的安装、登录、命令输入和常用操作。",
                }
            ],
        },
    ],
}


def prepare() -> Path:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    path = DEMO_ROOT / "library.json"
    if not path.exists():
        server.atomic_write_json(path, DEMO_LIBRARY)
    return path


def check() -> int:
    path = prepare()
    library = json.loads(path.read_text(encoding="utf-8"))
    results = server.search_library(library, "Pi Agent 架构", "saved")
    assert results and results[0]["segment_title"] == "Agent 运行时架构"
    assert not server.search_library(library, "量子计算纠错", "saved")
    print("demo check passed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="个人收藏视频知识检索演示")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.reset and DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    prepare()
    if args.check:
        return check()
    return server.run_server(DEMO_ROOT, open_browser=not args.no_open)


if __name__ == "__main__":
    raise SystemExit(main())
