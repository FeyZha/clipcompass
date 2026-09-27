import unittest
from unittest.mock import patch

from evaluation.lee2021_v1 import time_windows, validate_translations, LeePipeline
from library.precise_pipeline import PrecisePipeline


class LeeCollectionTests(unittest.TestCase):
    def test_chinese_windows_translation_and_original_evidence(self):
        cues = [{"start": i*5, "end": i*5+5, "text": "中文原始字幕"} for i in range(31)]
        chunks = time_windows("v", cues)
        self.assertEqual(set(range(31)), {i for c in chunks for i in c["segment_ids"]})
        self.assertTrue(all(c["end"]-c["start"] <= 60 for c in chunks))
        self.assertEqual(155, chunks[-1]["end"])
        self.assertEqual([], time_windows("v", []))
        valid = {"translations": [{"chunk_id": c["chunk_id"], "text": "English retrieval copy"} for c in chunks]}
        self.assertEqual(len(chunks), len(validate_translations(valid, chunks)))
        for invalid in ({}, {"translations": valid["translations"][:-1]},
                        {"translations": list(reversed(valid["translations"]))}, {"translations": [None]*len(chunks)}):
            with self.assertRaises(ValueError):
                validate_translations(invalid, chunks)
        pipeline = LeePipeline.__new__(LeePipeline)
        pipeline.source_chunks = chunks
        pipeline.source_by_id = {c["chunk_id"]: c for c in chunks}
        with patch.object(PrecisePipeline, "_rank", return_value=[{"candidate_id": chunks[1]["chunk_id"], "rank": 1}]):
            result = pipeline._rank("query", {"v"})[0]
        self.assertEqual(chunks[1]["text"], result["core_text"])
        self.assertEqual(chunks[0]["text"], result["context_before"])
        self.assertEqual(chunks[2]["text"], result["context_after"])


if __name__ == "__main__":
    unittest.main()
