import unittest
from unittest.mock import patch

from evaluation.mixed50_eval import span_scores
from evaluation.mixed50_v1 import library


class MixedCollectionTests(unittest.TestCase):
    def test_one_unfiltered_collection_and_interval_metric(self):
        videos = [{"video_id": str(i), "url": "https://www.youtube.com/watch?v=" + str(i),
                   "title": "Different subject", "channel": "Publisher", "duration": 100,
                   "subtitle_type": "manual", "topic": "must not route"} for i in range(50)]
        with patch("evaluation.mixed50_v1.read", return_value=videos):
            value = library()
        self.assertEqual(len(value["collections"]), 1)
        self.assertEqual(len(value["collections"][0]["video_ids"]), 50)
        self.assertTrue(all("topic" not in row for row in value["videos"]))
        item = lambda video, start, end: {"video_id": video, "recommended_watch_start": start, "recommended_watch_end": end}
        result = span_scores([item("v", 0, 20), item("v", 10, 30), item("other", 0, 10)],
                             [{"video_id": "v", "start": 5, "end": 15}, {"video_id": "v", "start": 10, "end": 25}])
        self.assertEqual(result["returned_seconds"], 40)
        self.assertEqual(result["hit_seconds"], 20)
        self.assertEqual(result["time_precision"], .5)
        self.assertEqual(result["gold_coverage"], 1)
        self.assertIsNone(span_scores([], [])["time_precision"])
        with self.assertRaises(ValueError):
            span_scores([item("v", 20, 10)], [])


if __name__ == "__main__":
    unittest.main()
