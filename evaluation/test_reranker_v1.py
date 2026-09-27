"""Model-free checks for frozen-candidate ranking and rescue/damage semantics."""
import unittest

from evaluation.run_reranker_v1 import classifications, first_gold, reorder


class RankingChecks(unittest.TestCase):
    def test_reorder_preserves_candidates_and_reports_damage_and_rescue(self):
        candidates = [{"rank": rank, "chunk_id": str(rank), "text": f"original {rank}",
                       "score": 1 / rank, "is_gold": rank in (6, 9)} for rank in range(1, 21)]
        result = reorder(candidates, [10 if c["rank"] == 6 else 0 for c in candidates])
        self.assertEqual((6, 1), (first_gold(candidates), first_gold(result)))
        self.assertEqual({c["chunk_id"] for c in candidates}, {c["chunk_id"] for c in result})
        self.assertEqual("original 6", result[0]["text"])
        self.assertEqual([1, 2, 3, 4, 5], [c["dense_rank"] for c in result[1:6]])
        self.assertIn("RANKING_RESCUE_TOP5", classifications(6, 1, True))
        self.assertIn("RANKING_RESCUE_TOP1", classifications(6, 1, True))
        self.assertEqual(["RANKING_DAMAGE"], classifications(1, 6, True))
        self.assertEqual(["RETRIEVAL_CEILING"], classifications(None, None, True))
        self.assertEqual(["NO_ANSWER_UNASSESSED"], classifications(None, None, False))
        with self.assertRaises(ValueError):
            reorder(candidates, [float("nan")] * 20)


if __name__ == "__main__":
    unittest.main()
