import unittest
import threading
from unittest.mock import Mock, patch

from library.guided_pipeline import GuidedPipeline, prepare_question, plan_directions
from library.server import AppError
from llm.provider import GenerationResult, LLMError


class GuidedQueryTests(unittest.TestCase):
    def test_intent_routing_and_validation(self):
        inner = Mock()
        inner.search.side_effect = lambda question, ids: {"answerable": True, "results": [], "question": question}
        guided = GuidedPipeline(inner)
        original = "我需要学习梯度下降相关的知识"
        def generation(mode, text):
            return GenerationResult({"mode": mode, "text": text}, "deepseek-flash", 1, 1, 0, 1)

        with patch("library.guided_pipeline.generate_json", return_value=generation("topic", "梯度下降的基本原理是什么？")), patch.object(guided, "explore", return_value={"answerable": True, "results": []}) as explore:
            result = guided.search(original, {"v"})
            explore.assert_called_once_with(original, {"v"}, "梯度下降的基本原理是什么？")
            inner.search.assert_not_called()
            self.assertEqual(result["question"], original)
            self.assertEqual(result["intent"]["mode"], "topic")
        inner.reset_mock()
        with patch("library.guided_pipeline.generate_json", return_value=generation("question", "错误的改写")):
            concrete = "不要定义，比较 A 和 B，并解释各自适用条件"
            guided.search(concrete, {"v"})
            inner.search.assert_called_once_with(concrete, {"v"})
        inner.reset_mock()
        with patch("library.guided_pipeline.generate_json", return_value=generation("clarify", "你指的是哪个概念？")):
            with self.assertRaises(AppError) as raised:
                guided.search("这个怎么用", {"v"})
            self.assertEqual((raised.exception.code, raised.exception.status), ("clarification_required", 422))
            inner.search.assert_not_called()
        for mode, text in [(None, "abc"), ([], "abc"), ("topic", ""), ("topic", None), ("topic", "a" * 2001)]:
            with self.subTest(mode=mode, text=text), patch("library.guided_pipeline.generate_json", return_value=generation(mode, text)):
                with self.assertRaises(LLMError):
                    prepare_question(original)

    def test_map_checks_every_direction_and_does_not_invent_evidence(self):
        plans = [{"title": str(i), "question": f"q{i}"} for i in range(4)]
        clip = {"video_id": "v", "recommended_watch_start": 10, "recommended_watch_end": 20}
        found = {"answerable": True, "results": [clip]}
        inner = Mock()
        inner.search.side_effect = [found, found, {"answerable": False, "results": []}, LLMError("offline")]
        inner.lock = threading.Lock()
        inner.query_understander.return_value = ({"english_query": "topic"}, {})
        inner._rank.return_value = [{"core_text": "topic", "context_before": "", "context_after": ""}]
        inner._gate.return_value = ({"answerable": True}, {})
        with patch("library.guided_pipeline.plan_directions", return_value=(plans, {})):
            response = GuidedPipeline(inner).explore("topic", {"v"}, "topic basics")
        self.assertEqual(len(response["learning_map"]["groups"]), 1)
        self.assertEqual(response["learning_map"]["failed_directions"], 1)
        self.assertEqual(response["results"], [clip])
        self.assertEqual([call.args for call in inner.search.call_args_list], [(f"q{i}", {"v"}) for i in range(4)])
        inner.search.side_effect = LLMError("offline")
        with patch("library.guided_pipeline.plan_directions", return_value=(plans, {})):
            with self.assertRaises(LLMError):
                GuidedPipeline(inner).explore("topic", {"v"}, "topic basics")
        inner._gate.return_value = ({"answerable": False}, {})
        inner.reset_mock()
        with patch("library.guided_pipeline.plan_directions") as plan:
            response = GuidedPipeline(inner).explore("absent topic", {"v"}, "absent topic basics")
            self.assertFalse(response["answerable"])
            plan.assert_not_called()
            inner.search.assert_not_called()
        for invalid in (None, [{}], [{"title": "x", "question": "q"}] * 5,
                        [{"title": "x", "question": ""}] * 2):
            generation = GenerationResult({"directions": invalid}, "deepseek-flash", 1, 1, 0, 1)
            with patch("library.guided_pipeline.generate_json", return_value=generation):
                with self.assertRaises(LLMError):
                    plan_directions("topic", [])


if __name__ == "__main__":
    unittest.main()
