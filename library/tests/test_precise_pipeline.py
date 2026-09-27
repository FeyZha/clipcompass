import copy
import unittest
from unittest.mock import patch

from library.mvp_pipeline import MvpPipeline
from library.precise_pipeline import PrecisePipeline, locate_span
from llm.provider import GenerationResult, LLMError


class PreciseSpanTests(unittest.TestCase):
    def test_caption_bounds_and_playback_contract(self):
        cues = [{"start": i * 10, "end": i * 10 + 9, "text": "Caption."} for i in range(5)]
        item = {"video_id": "v", "recommended_watch_start": 10, "recommended_watch_end": 39,
                "core_start": 10, "core_end": 29}
        valid = {"sufficient": True, "start_cue_id": 2, "end_cue_id": 3, "reason": "完整解释"}

        def generation(value):
            return GenerationResult(value, "deepseek-flash", 1, 1, 0, 1)

        with patch("library.precise_pipeline.generate_json", return_value=generation(valid)) as generate:
            answer, _ = locate_span("问题", item, cues)
            self.assertEqual((20, 39, 20), (answer["recommended_watch_start"], answer["recommended_watch_end"], answer["playback_start"]))
            self.assertEqual((10, 39), (answer["context_start"], answer["context_end"]))
            self.assertEqual(10, item["recommended_watch_start"])
            pipeline = PrecisePipeline.__new__(PrecisePipeline)
            pipeline.captions = {"v": cues}
            original = {"answerable": True, "results": [item]}
            with patch.object(MvpPipeline, "search", return_value=copy.deepcopy(original)):
                self.assertEqual(answer, pipeline.search("问题", {"v"})["results"][0])
            generate.reset_mock()
            with patch.object(MvpPipeline, "search", return_value={"answerable": False, "results": []}):
                self.assertFalse(pipeline.search("库内未讲", {"v"})["answerable"])
            generate.assert_not_called()

        for invalid in ({"start_cue_id": 0}, {"end_cue_id": 4}, {"start_cue_id": 3, "end_cue_id": 2},
                        {"start_cue_id": True}, {"start_cue_id": 2.0}, {"sufficient": False},
                        {"end_cue_id": None}, {"reason": ""}):
            with self.subTest(invalid=invalid), patch("library.precise_pipeline.generate_json", return_value=generation({**valid, **invalid})):
                with self.assertRaises(LLMError):
                    locate_span("问题", item, cues)
        with patch("library.precise_pipeline.generate_json") as generate:
            with self.assertRaises(LLMError):
                locate_span("问题", item, [])
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
