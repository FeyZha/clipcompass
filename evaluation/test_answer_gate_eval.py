"""Small synthetic checks for the blind gate runner and metric denominators."""
import datetime as dt
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from evaluation.run_answer_gate_v1 import input_guard, price, summarize, temporal_hit


class GateEvaluationChecks(unittest.TestCase):
    def test_real_runner_cascade_with_mocked_provider_in_isolated_process(self):
        script = r'''
import json, tempfile
from pathlib import Path
from unittest.mock import patch
import evaluation.run_answer_gate_v1 as runner
from llm.answer_gate import GateResponseError
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    directory = root / 'evaluation_runs/answer_gate_v1/dev'
    directory.mkdir(parents=True)
    runner.ROOT, runner.RUN = root, directory.parent
    runner.PIPELINE = root / 'pipeline.json'
    runner.write_json(runner.PIPELINE, {})
    candidates = [{'candidate_id': str(i), 'rank': i, 'video_id': 'synthetic', 'start': i, 'end': i+1, 'text': 'synthetic'} for i in range(1,21)]
    runner.write_new_rows(directory / 'inputs.jsonl', [{'test_id': name, 'original_question': name, 'english_query': name, 'candidates': candidates} for name in ('early','late','absent')])
    runner.write_json(directory / 'config.json', {'status':'prepared', 'questions':3, 'gate_source_hashes':{}, 'pipeline_sha256':runner.digest(runner.PIPELINE), 'inputs_sha256':runner.digest(directory/'inputs.jsonl')})
    seen = []
    def mocked(question, english, batch):
        ranks = [c['rank'] for c in batch]
        seen.append((question, ranks))
        if len(seen) == 1:
            raise GateResponseError('synthetic schema error', {'model':'deepseek-flash','input_tokens':100,'output_tokens':10,'cached_input_tokens':0,'latency_ms':1})
        yes = question == 'early' or (question == 'late' and ranks[0] == 6)
        labels = [{'candidate_id': c['candidate_id'], 'support': 'sufficient' if yes and n == 0 else 'partial', 'reason':'Synthetic evidence judgment.'} for n,c in enumerate(batch)]
        return {'candidates':labels, 'answerable':yes, 'best_candidate_id':batch[0]['candidate_id'] if yes else None, 'support':'sufficient' if yes else 'none'}, {'model':'deepseek-flash','input_tokens':100,'output_tokens':10,'cached_input_tokens':0,'latency_ms':1}
    with patch('llm.answer_gate.gate_pass', side_effect=mocked), patch.object(runner, 'check_hashes'):
        runner.run('dev')
    predictions = runner.rows(directory/'results.jsonl')
    assert [r['pass_count'] for r in predictions] == [1,2,2]
    assert [r['answerable'] for r in predictions] == [True,True,False]
    assert [r['best_candidate_id'] for r in predictions] == ['1','6',None]
    assert [ranks for _,ranks in seen] == [list(range(1,6)),list(range(1,6)),list(range(1,6)),list(range(6,21)),list(range(1,6)),list(range(6,21))]
    calls = runner.rows(directory/'calls.jsonl')
    assert len(calls) == 6 and calls[0]['error'] == 'GateResponseError' and calls[0]['fatal'] is False
    try:
        (root/'evaluation_data/holdout_v1/private_ground_truth.jsonl').read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('Gold was not blocked')
'''
        subprocess.run([sys.executable, '-B', '-c', script], check=True, capture_output=True)

    def test_guard_and_cost(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            out = root / "evaluation_runs/answer_gate_v1/dev"
            guard, counts = input_guard(root, out)
            guard("open", (str(out / "inputs.jsonl"), "r", 0))
            for file in (root / "evaluation_data/holdout_v1/private_ground_truth.jsonl",
                         root / "evaluation_runs/reranker_v1/results.jsonl"):
                with self.assertRaises(PermissionError):
                    guard("open", (str(file), "r", 0))
            self.assertEqual(counts["denied"], 2)
        usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 200}
        saturday = dt.datetime(2026, 9, 26, tzinfo=dt.timezone.utc)
        self.assertAlmostEqual(price(usage, saturday)["estimated_cost_cny"], .001204)

    def test_conditional_recall_precision_and_ceiling_are_separate(self):
        gold = []
        inputs, predictions, audit = [], [], []
        for i, (answerable, available, accepted, support) in enumerate((
                (True, True, True, "partial"), (True, True, False, None),
                (True, False, True, "mention"), (False, False, True, "mention"),
                (False, False, False, None))):
            identifier = str(i)
            gold.append({"id": identifier, "question": "q", "answerable": answerable,
                         "gold_evidence": [{"video_id": "v", "start": 10, "end": 20}] if answerable else []})
            candidate = {"candidate_id": identifier, "video_id": "v", "start": 10 if available else 20, "end": 30}
            inputs.append({"test_id": identifier, "original_question": "q", "candidates": [candidate]})
            predictions.append({"test_id": identifier, "answerable": accepted,
                "best_candidate_id": identifier if accepted else None, "pass_count": 1 if accepted else 2,
                "gate_latency_ms": 100, "error": None})
            if accepted:
                audit.append({"test_id": identifier, "support": support})
        calls = [{"pass": p, "usage": {"estimated_cost_cny": .001, "input_tokens": 100, "output_tokens": 10}, "error": None}
                 for p in [1,1,1,1,1,2,2]]
        report = summarize(gold, inputs, predictions, audit, calls)
        self.assertAlmostEqual(report["metrics"]["answerable_recall"], 2/3)
        self.assertEqual(report["metrics"]["no_answer_accuracy"], .5)
        self.assertEqual(report["metrics"]["conditional_gate_recall"], .5)
        self.assertEqual(report["metrics"]["answerable_precision_independent_audit"], 0)
        self.assertEqual(report["metrics"]["gold_window_supported_precision"], 1/3)
        self.assertEqual(report["metrics"]["evidence_selection_accuracy"], .5)
        self.assertEqual(report["failures"], dict.fromkeys(("GATE_FALSE_POSITIVE", "GATE_FALSE_NEGATIVE", "GATE_WRONG_EVIDENCE", "RETRIEVAL_CEILING"), 1))
        self.assertFalse(temporal_hit({"video_id": "v", "start": 20, "end": 25}, gold[0]["gold_evidence"]))


if __name__ == "__main__":
    unittest.main()
