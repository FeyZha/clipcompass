import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from library.intent_pipeline_v2 import plan_request, group_clips, IntentPipeline
from library.server import AppError
from llm.provider import GenerationResult, LLMError


def plan():
    return {"task_type": "learn", "topics": ["梯度下降"], "goal": "学习梯度下降", "english_query": "introduce gradient descent",
            "core_query": "gradient descent", "constraints": [], "assumptions": [], "clarification": "",
            "directions": [{"title": "更新", "question": "如何更新？", "english_query": "update parameters",
                            "origin": "suggested", "required": False, "depends_on": []}]}


def generation(value):
    return GenerationResult(value, "deepseek-flash", 1, 1, 0, 1)


class IntentTests(unittest.TestCase):
    def test_plan_constraints_validation_and_clarification(self):
        value = plan()
        value["constraints"] = [{"text": "不推导", "quote": "不要推导"}]
        with patch("library.intent_pipeline_v2.generate_json", return_value=generation(value)):
            result,_ = plan_request("介绍梯度下降，不要推导")
            self.assertEqual(result["task_type"], "specific")
            self.assertEqual(result["original_request"], "介绍梯度下降，不要推导")
            self.assertEqual(result["directions"], [])
        for modify in (lambda v:v.update(topics=[]), lambda v:v.update(task_type="bad"),
                       lambda v:v["directions"][0].update(depends_on=[0]),
                       lambda v:v.update(constraints=[{"text":"x","quote":"invented"}])):
            value=plan(); modify(value)
            with patch("library.intent_pipeline_v2.generate_json", return_value=generation(value)), self.assertRaises(LLMError):
                plan_request("介绍梯度下降")

    def test_grouping_preserves_points_and_verified_intervals(self):
        clip=lambda a,b:{"video_id":"v", "recommended_watch_start":a, "recommended_watch_end":b,"reason":"evidence"}
        d=plan()["directions"][0]
        groups=group_clips([(d,clip(10,30)),(d,clip(12,20)),(d,clip(40,50))])
        self.assertEqual(len(groups),2)
        self.assertEqual(len(groups[0]["points"]),2)
        self.assertEqual(groups[0]["clips"],[clip(10,30)])

    def test_provider_alias_and_missing_referent(self):
        value=plan(); value.pop("task_type")
        value.update(type="clarify",topics=[],core_query="",directions=[],clarification="你指什么？")
        with patch("library.intent_pipeline_v2.generate_json",return_value=generation(value)):
            actual,_=plan_request("这个是怎么工作的？")
        self.assertEqual(actual["task_type"],"clarify")
        value=plan(); value["core_query"]=""
        with patch("library.intent_pipeline_v2.generate_json",return_value=generation(value)), self.assertRaises(LLMError):
            plan_request("介绍梯度下降")

    def test_suggested_order_is_not_a_retrieval_veto(self):
        value=plan()
        value["directions"].append({**value["directions"][0],"title":"另一个方向","depends_on":[0],"origin":"explicit","required":True})
        with patch("library.intent_pipeline_v2.generate_json",return_value=generation(value)):
            actual,_=plan_request("介绍梯度下降")
        self.assertTrue(all(not d["required"] for d in actual["directions"]))
        self.assertEqual(actual["directions"][1]["depends_on"],[0])

    def test_multiroute_scope_partial_failures_and_original_question(self):
        inner=Mock()
        inner.chunks=[{"chunk_id":"a", "video_id":"v", "text":"evidence", "start":0,"end":20},
                      {"chunk_id":"b", "video_id":"outside", "text":"must not retrieve", "start":0,"end":20}]
        inner.source_chunks=inner.chunks
        inner.vectors=np.array([[1.,0.],[0.,1.]])
        inner.lock=threading.Lock()
        inner.metadata={"v":{"video_id":"v","title":"t","channel":"c","url":"https://www.youtube.com/watch?v=v"}}
        inner.captions={"v":[]}
        decision={"candidates":[{"candidate_id":"a","support":"sufficient","reason":"Direct evidence."}],
                  "answerable":True,"best_candidate_id":"a","support":"sufficient"}
        with tempfile.TemporaryDirectory() as folder:
            pipeline=IntentPipeline(inner,Path(folder))
            try:
                with patch("library.intent_pipeline_v2.plan_request", return_value=(plan(),{})), \
                     patch("library.intent_pipeline_v2.encode", side_effect=lambda m,t,q,**kw:np.array([[1.,0.]]*len(q))), \
                     patch("library.intent_pipeline_v2.score_pairs", return_value=[1.]), \
                     patch("library.intent_pipeline_v2.generate_json",return_value=generation(decision)) as gate, \
                     patch("library.intent_pipeline_v2.locate_span",side_effect=lambda q,c,s:(dict(c,reason="evidence"),{})):
                    response=pipeline.search("介绍梯度下降",{"v"})
                    self.assertEqual(response["status"],"supported")
                    self.assertEqual(response["results"][0]["video_id"],"v")
                    self.assertIn("梯度下降",gate.call_args.args[1])
                value=plan()
                value["directions"].append({**value["directions"][0],"title":"第二方向","depends_on":[0]})
                absent={"candidates":[{"candidate_id":"a","support":"partial","reason":"Missing part."}],
                        "answerable":False,"best_candidate_id":None,"support":"none"}
                with patch("library.intent_pipeline_v2.plan_request",return_value=(value,{})), \
                     patch("library.intent_pipeline_v2.encode",side_effect=lambda m,t,q,**kw:np.array([[1.,0.]]*len(q))), \
                     patch("library.intent_pipeline_v2.score_pairs",return_value=[1.]), \
                     patch("library.intent_pipeline_v2.generate_json",side_effect=[generation(absent),generation(decision)]), \
                     patch("library.intent_pipeline_v2.locate_span",side_effect=lambda q,c,s:(dict(c,reason="evidence"),{})):
                    response=pipeline.search("介绍梯度下降",{"v"})
                    self.assertEqual(response["status"],"partial")
                    self.assertEqual([r["status"] for r in response["coverage"]],["partial_evidence","supported"])
                value=plan(); value["task_type"]="compare"; value["directions"]=[]
                with patch("library.intent_pipeline_v2.plan_request", return_value=(value,{})), \
                     patch("library.intent_pipeline_v2.encode",return_value=np.array([[1.,0.]])), \
                     patch("library.intent_pipeline_v2.score_pairs", return_value=[1.]), \
                     patch("library.intent_pipeline_v2.generate_json",side_effect=LLMError("failed")) as gate:
                    with self.assertRaises(LLMError): pipeline.search("比较A和B，不要省略适用条件",{"v"})
                    self.assertIn("比较A和B，不要省略适用条件",gate.call_args.args[1])
                value=plan(); value.update(task_type="clarify",clarification="你指什么？")
                with patch("library.intent_pipeline_v2.plan_request",return_value=(value,{})), \
                     patch("library.intent_pipeline_v2.encode") as encode:
                    with self.assertRaises(AppError): pipeline.search("这个呢",{"v"})
                    encode.assert_not_called()
            finally:
                for handler in pipeline.logger.handlers: handler.close()


if __name__ == "__main__": unittest.main()
