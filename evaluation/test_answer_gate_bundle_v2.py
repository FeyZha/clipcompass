"""Synthetic bundle checks; no dataset or API access."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm.answer_gate_bundle import attach_bundles, bundle_payload, gate_pass, BUNDLE_PROMPT, GATE_SCHEMA
from evaluation.run_answer_gate_v1 import temporal_hit


class BundleChecks(unittest.TestCase):
    def test_neighbors_boundaries_payload_and_core_selection(self):
        chunks = [{'chunk_id': str(i), 'video_id': 'a' if i < 4 else 'b', 'start': 10*i,
                   'end': 10*i+15, 'text': f'Original text {i}.'} for i in range(1, 6)]
        candidates = [{**c, 'rank': n+1, 'gold_evidence': 'NEVER SEND'}
                      for n, c in enumerate([chunks[1], chunks[0], chunks[2], chunks[3]])]
        bundles = attach_bundles(candidates, list(reversed(chunks)))
        self.assertEqual([(b['previous_chunk_id'], b['next_chunk_id']) for b in bundles],
                         [('1', '3'), (None, '2'), ('2', None), (None, '5')])
        for before, after in zip(candidates, bundles):
            self.assertEqual((before['start'], before['end'], before['text']),
                             (after['core_start'], after['core_end'], after['core_text']))
        chosen = bundles[0]
        self.assertEqual((chosen['bundle_start'], chosen['bundle_end']), (10, 45))
        self.assertFalse(temporal_hit(chosen, [{'video_id': 'a', 'start': 10, 'end': 19}]))
        payload = bundle_payload(bundles)
        self.assertNotIn('gold_evidence', json.dumps(payload))
        self.assertNotIn('previous_chunk_id', json.dumps(payload))
        self.assertNotIn('text', payload[0])
        self.assertEqual(payload[0]['core_text'], chunks[1]['text'])
        decision = {'candidates': [{'candidate_id': b['candidate_id'], 'support': 'sufficient' if n == 0 else 'none',
                     'reason': 'Synthetic support.'} for n, b in enumerate(bundles)],
                    'answerable': True, 'best_candidate_id': '2', 'support': 'sufficient'}
        generation = SimpleNamespace(value=decision, usage=lambda: {'input_tokens': 1})
        with patch('llm.answer_gate_bundle.generate_json', return_value=generation) as provider:
            actual, _ = gate_pass('中文问题', 'English question', bundles)
            self.assertEqual(actual['best_candidate_id'], '2')
            args, kwargs = provider.call_args
            self.assertEqual(args[0], BUNDLE_PROMPT)
            self.assertEqual(json.loads(args[1])['candidates'], payload)
            self.assertEqual(kwargs, {'schema': GATE_SCHEMA, 'temperature': 0.0, 'max_tokens': 3000})
        with self.assertRaises(ValueError):
            bundle_payload([{**chosen, 'core_start': 0}])
        with self.assertRaises(ValueError):
            bundle_payload([{**chosen, 'bundle_end': float('inf')}])


if __name__ == '__main__':
    unittest.main()
