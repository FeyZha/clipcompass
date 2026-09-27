"""Midpoint mapping, duplicate suppression and candidate-vs-final recall."""
import unittest

from evaluation.run_rescue_v1 import candidate_pool, changes, region_mapping


class RescueChecks(unittest.TestCase):
    def test_fixed_mapping_and_dedup(self):
        m = [{"chunk_id": str(i), "video_id": video, "start": start, "end": end}
             for i, video, start, end in ((1, "v", 0, 10), (2, "v", 8, 12),
                                          (3, "v", 19, 21), (4, "other", 8, 12), (5, "v", 20, 24))]
        l = [{"chunk_id": "L1", "video_id": "v", "start": 10, "end": 20},
             {"chunk_id": "L2", "video_id": "v", "start": 9, "end": 21}]
        mapping = region_mapping(m, l)
        self.assertEqual(["2", "3"], mapping["L1"])
        ids, origins = candidate_pool(["1", "2"], ["L1", "L2"], mapping)
        self.assertEqual(["1", "2", "3"], ids)
        self.assertEqual(["L1", "L2"], origins["3"]["l_regions"])
        self.assertFalse(origins["3"]["m_top20"])
        self.assertEqual(["1", "2"], candidate_pool(["1", "2"], [], mapping)[0])

    def test_gain_damage_and_no_answer(self):
        self.assertTrue(changes({"gold_pool_rank": None}, {"gold_pool_rank": 24, "gold_answerable": True})["candidate_gain"])
        self.assertTrue(changes({"gold_pool_rank": 1}, {"gold_pool_rank": 2, "gold_answerable": True})["top1_damage"])
        self.assertTrue(changes({"gold_pool_rank": 7}, {"gold_pool_rank": 1, "gold_answerable": True})["top1_gain"])
        self.assertEqual(["NO_ANSWER_UNASSESSED"], changes({}, {"gold_answerable": False})["labels"])


if __name__ == "__main__":
    unittest.main()
