import unittest

from evaluation.english_v1 import coverage, parse_vtt


class EnglishCollectionTests(unittest.TestCase):
    def test_manual_and_rolling_caption_words_and_times_are_preserved(self):
        manual = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nA &amp; B.\nSecond line.\n"
        self.assertEqual(parse_vtt(manual, 3), [{"start": 1, "end": 2, "text": "A & B. Second line."}])
        rolling = ("WEBVTT\n\n00:00:00.000 --> 00:00:01.990\n \nShe<00:00:00.500><c> said</c>\n\n"
                   "00:00:01.990 --> 00:00:02.000\nShe said\n \n\n"
                   "00:00:02.000 --> 00:00:03.990\nShe said\nshe<00:00:02.500><c> was</c>\n\n"
                   "00:00:03.990 --> 00:00:04.000\nshe was\n \n\n"
                   "00:00:04.000 --> 00:00:05.000\nshe was\nworking.\n")
        parsed = parse_vtt(rolling, 6)
        self.assertEqual([r["text"] for r in parsed], ["She said", "she was", "working."])
        self.assertEqual([r["start"] for r in parsed], [0, 2, 4])
        with self.assertRaises(ValueError):
            parse_vtt(manual, 1)

    def test_evidence_requires_the_right_video_and_sufficient_coverage(self):
        gold = {"video_id": "v", "start": 10, "end": 20}
        self.assertEqual(coverage({"video_id": "other", "start": 10, "end": 20}, gold), 0)
        self.assertEqual(coverage({"video_id": "v", "start": 18, "end": 30}, gold), .2)
        self.assertEqual(coverage({"video_id": "v", "recommended_watch_start": 5,
                                   "recommended_watch_end": 25}, gold, True), 1)


if __name__ == "__main__":
    unittest.main()
