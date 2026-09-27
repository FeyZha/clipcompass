"""Synthetic-only checks. Never reads holdout questions, labels, or predictions."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.run_holdout_v1 import METRICS, ROOT, aggregate, reorder, validate_predictions, validate_questions
from evaluation.freeze_holdout_v1 import overlap_ratio


class BlindHoldoutChecks(unittest.TestCase):
    def test_dev_overlap_boundary(self):
        region = {"video_id": "v", "start": 0, "end": 10}
        self.assertEqual(overlap_ratio(region, {"video_id": "other", "start": 0, "end": 10}), 0)
        self.assertEqual(overlap_ratio(region, {"video_id": "v", "start": 2, "end": 4}), 1)
        self.assertEqual(overlap_ratio(region, {"video_id": "v", "start": 8, "end": 18}), .2)

    def test_guard_blocks_private_data_and_old_results_in_real_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "evaluation_data/holdout_v1/questions.jsonl"
            secret = root / "evaluation_data/holdout_v1/private_ground_truth.jsonl"
            old = root / "evaluation_runs/reranker_v1/results.jsonl"
            for path in (questions, secret, old):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            script = """import sys
from pathlib import Path
from evaluation.run_holdout_v1 import make_read_guard
r = Path(sys.argv[1])
q = r / 'evaluation_data/holdout_v1/questions.jsonl'
guard, counts = make_read_guard(r, q, r / 'evaluation_runs/holdout_v1')
sys.addaudithook(guard)
assert q.read_text() == '{}'
for p in ('evaluation_data/holdout_v1/private_ground_truth.jsonl', 'evaluation_runs/reranker_v1/results.jsonl'):
    try:
        (r / p).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('private file was readable')
assert counts['protected_opens_denied'] == 2
"""
            subprocess.run([sys.executable, "-B", "-c", script, str(root)], cwd=ROOT, check=True, capture_output=True)

    def test_question_schema_rejects_labels(self):
        validate_questions([{"id": "synthetic", "question": "where"}])
        for rows in ([{"id": "synthetic", "question": "where", "answerable": True}],
                     [{"id": "same", "question": "one"}, {"id": "same", "question": "two"}]):
            with self.assertRaises(ValueError):
                validate_questions(rows)

    def test_mock_ranking_and_aggregate_metrics_without_gold_leak(self):
        chunks = [{"chunk_id": f"m:synthetic:{i}", "video_id": "v", "start": i * 10,
                   "end": i * 10 + 5, "segment_ids": [i + 1], "word_count": 1, "text": "synthetic"}
                  for i in range(20)]
        candidates = [{**chunk, "rank": i + 1, "score": 1 - i / 20} for i, chunk in enumerate(chunks)]
        results = reorder(candidates, [0, 3] + [-i for i in range(2, 20)])
        questions = [{"id": "a", "question": "first"}, {"id": "b", "question": "missing"},
                     {"id": "c", "question": "none"}]
        rows = [{"test_id": q["id"], "question": q["question"], "dense_results": candidates,
                 "results": results, "system_answerable": None, "explicit_no_answer": False, "error": None}
                for q in questions]
        validate_predictions(rows, questions, chunks)
        gold = [{**questions[0], "type": "precise_retrieval", "answerable": True,
                 "gold_evidence": [{"video_id": "v", "start": 0, "end": 5}], "video_relevance": {"v": "sufficient"}},
                {**questions[1], "type": "cross_segment", "answerable": True,
                 "gold_evidence": [{"video_id": "v", "start": 5, "end": 10}], "video_relevance": {"v": "sufficient"}},
                {**questions[2], "type": "no_answer", "answerable": False,
                 "gold_evidence": [], "video_relevance": {"v": "none"}}]
        report = aggregate(gold, rows, dict.fromkeys(METRICS, 0))
        self.assertEqual(report["metrics"]["candidate_recall"], .5)
        self.assertEqual(report["metrics"]["recall_at_20"], .5)
        self.assertEqual(report["metrics"]["top_1"], 0)
        self.assertEqual(report["metrics"]["mrr_at_20"], .25)
        self.assertEqual(report["metrics"]["video_recall_at_5"], 1)
        self.assertEqual(report["failure_count"], 3)
        self.assertEqual(report["failure_category_counts"]["RETRIEVAL_MISS"], 1)
        self.assertEqual(report["failure_category_counts"]["RANKING_ERROR"], 1)
        self.assertEqual(report["failure_category_counts"]["NO_ANSWER_NOT_REJECTED"], 1)
        self.assertFalse(report["no_answer_behavior"]["gate_evaluated"])
        for forbidden in ('"question":', '"gold_evidence":', '"test_id":', '"start":', '"text":'):
            self.assertNotIn(forbidden, json.dumps(report))
        rows[0]["results"] = rows[0]["results"][:-1]
        with self.assertRaises(ValueError):
            validate_predictions(rows, questions, chunks)


if __name__ == "__main__":
    unittest.main()
