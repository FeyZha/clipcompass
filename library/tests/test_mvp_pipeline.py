import threading
import unittest

from library.mvp_pipeline import MvpPipeline


class MvpPipelineContractTests(unittest.TestCase):
    def pipeline(self, answerable=True):
        pipeline = MvpPipeline.__new__(MvpPipeline)
        pipeline.lock = threading.Lock()
        pipeline.query_understander = lambda question: ({"english_query": "MCP tools"}, {"model": "deepseek-flash"})
        pipeline.metadata = {"video": {"title": "A real video", "channel": "Creator",
                                       "url": "https://www.youtube.com/watch?v=video"}}
        bundle = {"candidate_id": "chunk", "rank": 1, "video_id": "video", "start": 20, "end": 40,
                  "text": "core", "core_start": 20, "core_end": 40, "core_text": "core",
                  "bundle_start": 10, "bundle_end": 55, "context_before": "before", "context_after": "after"}
        pipeline._rank = lambda query, allowed: [bundle]
        labels = [{"candidate_id": "chunk", "support": "sufficient", "reason": "Explains the requested relationship."}]
        decision = {"candidates": labels, "answerable": answerable,
                    "best_candidate_id": "chunk" if answerable else None,
                    "support": "sufficient" if answerable else "none"}
        pipeline._gate = lambda question, query, bundles: (decision, {"model": "deepseek-flash"})
        return pipeline

    def test_answer_uses_core_for_jump_and_bundle_for_watch_range(self):
        result = self.pipeline().search("问题", {"video"})
        item = result["results"][0]
        self.assertTrue(result["answerable"])
        self.assertEqual((20, 40), (item["core_start"], item["core_end"]))
        self.assertEqual((10, 55), (item["recommended_watch_start"], item["recommended_watch_end"]))
        self.assertNotIn("transcript", item)

    def test_no_answer_has_no_external_fallback(self):
        result = self.pipeline(False).search("问题", {"video"})
        self.assertFalse(result["answerable"])
        self.assertEqual([], result["results"])
        self.assertFalse(result["external_answer_fallback"])
        self.assertIn("无需继续", result["follow_up"])


if __name__ == "__main__":
    unittest.main()
