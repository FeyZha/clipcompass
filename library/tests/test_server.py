import json
import sys
import tempfile
import unittest
from pathlib import Path


LIBRARY_DIR = Path(__file__).resolve().parents[1]
if str(LIBRARY_DIR) not in sys.path:
    sys.path.insert(0, str(LIBRARY_DIR))

import server


class VideoKnowledgeTests(unittest.TestCase):
    def library(self):
        return {
            "collections": [{"collection_id": "saved", "video_ids": ["deep", "popular"]}],
            "videos": [
                {
                    "video_id": "deep",
                    "platform_video_id": "abcdef12345",
                    "source_url": "https://www.youtube.com/watch?v=abcdef12345",
                    "title": "运行时原理",
                    "creator": "A",
                    "view_count": 100,
                    "creator_followers": 20,
                    "segments": [{
                        "segment_id": "architecture",
                        "start_seconds": 90,
                        "end_seconds": 180,
                        "title": "Pi Agent 运行时架构",
                        "summary": "解释模型、工具注册与上下文循环。",
                        "transcript": "Pi Agent 的架构由模型层、工具注册表和上下文循环组成。",
                    }],
                },
                {
                    "video_id": "popular",
                    "platform_video_id": "zyxwvu98765",
                    "source_url": "https://www.youtube.com/watch?v=zyxwvu98765",
                    "title": "热门入门教程",
                    "creator": "B",
                    "view_count": 9_000_000,
                    "creator_followers": 1_000_000,
                    "segments": [{
                        "segment_id": "install",
                        "start_seconds": 10,
                        "end_seconds": 80,
                        "title": "Pi Agent 安装",
                        "summary": "演示下载和登录步骤。",
                        "transcript": "这里介绍 Pi Agent 的安装与基础使用。",
                    }],
                },
            ],
        }

    def test_relevance_beats_popularity(self):
        results = server.search_library(self.library(), "Pi Agent 的运行时架构是什么", "saved")
        self.assertEqual(["architecture"], [item["segment_id"] for item in results])

    def test_no_answer_is_honest(self):
        self.assertEqual([], server.search_library(self.library(), "量子纠错算法", "saved"))

    def test_popularity_only_breaks_relevance_ties(self):
        library = self.library()
        library["videos"][1]["segments"][0] = dict(library["videos"][0]["segments"][0], segment_id="popular-architecture")
        results = server.search_library(library, "Pi Agent 运行时架构", "saved")
        self.assertEqual("popular", results[0]["video_id"])

    def test_youtube_url_variants(self):
        self.assertEqual("abcdef12345", server.parse_youtube_id("https://youtu.be/abcdef12345?t=12"))
        self.assertEqual("abcdef12345", server.parse_youtube_id("https://www.youtube.com/watch?v=abcdef12345"))
        with self.assertRaises(server.AppError):
            server.parse_youtube_id("https://example.com/video")

    def test_store_adds_video_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = server.LibraryStore(Path(directory))
            store.add_video({
                "source_url": "https://www.youtube.com/watch?v=abcdef12345",
                "title": "测试视频",
                "segment_text": "00:10-01:00 架构 | 解释系统架构",
            })
            saved = json.loads((Path(directory) / "library.json").read_text(encoding="utf-8"))
            self.assertEqual("youtube:abcdef12345", saved["videos"][0]["video_id"])
            self.assertEqual(["youtube:abcdef12345"], saved["collections"][0]["video_ids"])

    def test_server_rejects_public_bind(self):
        with self.assertRaises(server.AppError):
            server.require_loopback_host("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
