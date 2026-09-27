"""Dev experiment: judge fixed adjacent M context, select the unchanged core."""

import json
import math
from collections import defaultdict

from .answer_gate import GATE_PROMPT, GATE_SCHEMA, GateResponseError, candidate_payload, validate_decision
from .provider import generate_json


# Same sufficiency policy; only the permitted evidence boundary changes.
BUNDLE_PROMPT = GATE_PROMPT.replace(
    "an individual video-caption candidate", "an evidence bundle anchored to one retrieved core candidate"
).replace(
    "Evaluate EVERY candidate independently using only its own text:",
    "Evaluate EVERY candidate bundle independently using its core_text, context_before and context_after together:"
).replace(
    "outside knowledge, surrounding video content, another candidate, or the English query",
    "outside knowledge, video content not supplied in this bundle, another bundle, or the English query"
).replace(
    "Never combine multiple partial candidates into a sufficient answer, even when they overlap or are adjacent.",
    "You may combine the three text components WITHIN one bundle. Never combine separate bundles into a sufficient answer, "
    "even when they overlap or are adjacent. Select the core candidate_id; never invent an expanded playback interval."
)

FIELDS = ("candidate_id", "rank", "video_id", "core_start", "core_end", "bundle_start", "bundle_end",
          "core_text", "context_before", "context_after")


def attach_bundles(candidates, chunks):
    """No new retrieval: attach exactly the chronological previous and next M windows."""
    grouped = defaultdict(list)
    assert len({c['chunk_id'] for c in chunks}) == len(chunks), "Duplicate M IDs"
    for chunk in chunks:
        grouped[chunk['video_id']].append(chunk)
    neighbors = {}
    for windows in grouped.values():
        windows.sort(key=lambda c: (c['start'], c['end'], c['chunk_id']))
        for index, core in enumerate(windows):
            neighbors[core['chunk_id']] = (windows[index - 1] if index else None, core,
                                          windows[index + 1] if index + 1 < len(windows) else None)
    bundled = []
    for source in candidate_payload(candidates):
        before, core, after = neighbors[source['candidate_id']]
        assert all(source[k] == core[k] for k in ('video_id', 'start', 'end', 'text')), "Core changed"
        bundled.append({**source, 'core_start': core['start'], 'core_end': core['end'], 'core_text': core['text'],
                        'bundle_start': min(w['start'] for w in (before, core, after) if w is not None),
                        'bundle_end': max(w['end'] for w in (before, core, after) if w is not None),
                        'context_before': before['text'] if before else '',
                        'context_after': after['text'] if after else '',
                        'previous_chunk_id': before['chunk_id'] if before else None,
                        'next_chunk_id': after['chunk_id'] if after else None})
    return bundled


def bundle_payload(candidates):
    # Reuse the original identity/rank/core validation; aliases remain local, not model input.
    candidate_payload(candidates)
    payload = []
    for c in candidates:
        if any(c[a] != c[b] for a, b in [('core_start', 'start'), ('core_end', 'end'), ('core_text', 'text')]):
            raise ValueError('Bundle changed the core')
        if any(type(c[k]) not in (int, float) or not math.isfinite(c[k]) for k in ('bundle_start', 'bundle_end')):
            raise ValueError('Invalid bundle timestamps')
        if not 0 <= c['bundle_start'] <= c['core_start'] < c['core_end'] <= c['bundle_end']:
            raise ValueError('Bundle must contain the core')
        if any(not isinstance(c[k], str) for k in ('context_before', 'context_after')):
            raise ValueError('Invalid context text')
        payload.append({k: c[k] for k in FIELDS})
    return payload


def gate_pass(original_question, english_query, candidates):
    if any(not isinstance(v, str) or not v.strip() for v in (original_question, english_query)):
        raise ValueError('Questions must be nonempty strings')
    payload = bundle_payload(candidates)
    generation = generate_json(BUNDLE_PROMPT,
        json.dumps({'original_question': original_question, 'english_query': english_query, 'candidates': payload},
                   ensure_ascii=False, separators=(',', ':')), schema=GATE_SCHEMA, temperature=0.0, max_tokens=3000)
    usage = generation.usage()
    try:
        decision = validate_decision(generation.value, payload)
    except GateResponseError as exc:
        exc.usage, exc.decision = usage, generation.value
        raise
    return decision, usage
