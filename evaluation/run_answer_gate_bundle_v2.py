"""Dev-only fixed +/-1 bundle experiment. No Holdout or freeze command."""

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import run_answer_gate_v1 as v1
from evaluation.freeze_holdout_v1 import digest, read_json, rows, save_new, check_hashes, hashes
from llm.answer_gate_bundle import attach_bundles, bundle_payload, gate_pass, BUNDLE_PROMPT, GATE_SCHEMA

RUN = ROOT / 'evaluation_runs/answer_gate_v2_bundle'
DEV = RUN / 'dev'
DIAG = RUN / 'diagnosis'
OLD = ROOT / 'evaluation_runs/answer_gate_v1/dev'
CHUNKS = ROOT / 'evaluation_runs/window_v1/dense_window_m/chunks.jsonl'
GOLD = ROOT / 'evaluation_data/dev_v1/test_cases.jsonl'
AUDIT_POLICY = ('Independent reviewer sees only the question and selected bundle, never Gate reasons or Gold. '
                'Judge the complete bundle AND core text separately against every question requirement. '
                'Review all positives, including apparent agreements. No feedback to the model or prompt.')


def prepare():
    v1.verify_parent()
    assert read_json(DIAG / 'summary.json')['case_count'] == 13, 'Complete diagnosis first'
    _, old_config, inputs, _, _ = v1.validate_completed('dev')
    assert len(inputs) == 40 and not DEV.exists(), 'Do not overwrite a run'
    chunks = rows(CHUNKS)
    bundled = [{**row, 'candidates': attach_bundles(row['candidates'], chunks)} for row in inputs]
    for before, after in zip(inputs, bundled):
        assert len(after['candidates']) == 20
        assert all(all(a[k] == b[k] for k in a) for a, b in zip(before['candidates'], after['candidates']))
        bundle_payload(after['candidates'])
    DEV.mkdir()
    v1.write_new_rows(DEV / 'inputs.jsonl', bundled)
    sources = {**old_config['gate_source_hashes'], **hashes([ROOT / name for name in (
        'llm/answer_gate_bundle.py', 'evaluation/run_answer_gate_bundle_v2.py', 'evaluation/test_answer_gate_bundle_v2.py')])}
    save_new(DEV / 'config.json', {
        'split': 'dev', 'status': 'prepared', 'questions': 40,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'provider': 'deepseek', 'model': 'deepseek-flash', 'temperature': 0.0, 'max_tokens': 3000,
        'prompt': BUNDLE_PROMPT, 'schema': GATE_SCHEMA, 'gate_source_hashes': sources,
        'pipeline_sha256': digest(v1.PIPELINE), 'inputs_sha256': digest(DEV / 'inputs.jsonl'),
        'preserved_data_hashes': hashes([CHUNKS, GOLD, OLD / 'inputs.jsonl', OLD / 'results.jsonl',
             OLD / 'calls.jsonl', OLD / 'summary.json', DIAG / 'cases.jsonl', DIAG / 'summary.json', DIAG / 'report.md']),
        'bundle_rule': 'same video, sort (start,end,chunk_id), exactly previous/current/next; no text cleaning',
        'selection_rule': 'original core ID/start/end only; expanded evidence is not playback',
        'cascade': [[1, 5], [6, 20]], 'early_exit': 'any sufficient in pass 1',
        'schema_error_retries': 1, 'retry_policy': 'unchanged v1; identical input/model; API error stops',
        'new_query_understanding_calls': 0, 'enabled_l_rescue': False, 'holdout_run': False, 'gate_frozen': False,
        'audit_policy': AUDIT_POLICY, 'pricing': v1.PRICING,
    })
    print(json.dumps({'prepared': 'dev', 'questions': 40, 'original_top20_unchanged': True, 'diagnosis_first': True}))


def check_run():
    v1.verify_parent()
    check_hashes(read_json(DEV / 'config.json')['preserved_data_hashes'])


def run():
    check_run()
    # Process-local adapter reuses v1's exact cascade, validation, retry, cost and access guard.
    # No production source edits, no second implementation of the runner.
    with patch.object(v1, 'RUN', RUN), patch('llm.answer_gate.gate_pass', gate_pass):
        v1.run('dev')


def completed():
    check_run()
    with patch.object(v1, 'RUN', RUN):
        return v1.validate_completed('dev')


def export_audit():
    _, _, inputs, predictions, _ = completed()
    source = {r['test_id']: r for r in inputs}
    samples = []
    for result in predictions:
        if not result['answerable']:
            continue
        row = source[result['test_id']]
        chosen = next(c for c in row['candidates'] if c['candidate_id'] == result['best_candidate_id'])
        samples.append({'test_id': result['test_id'], 'original_question': row['original_question'],
                        'english_query': row['english_query'], 'selected_bundle': bundle_payload([chosen])[0]})
    v1.write_new_rows(DEV / 'audit_inputs.jsonl', samples)
    print(json.dumps({'positive_decisions_to_review': len(samples), 'gold_and_gate_reasons_hidden': True}))


def performance_extra(report, predictions, calls):
    p = report['performance']
    p['input_tokens'] = sum(c['usage']['input_tokens'] for c in calls)
    p['output_tokens'] = sum(c['usage']['output_tokens'] for c in calls)
    p['mean_tokens_per_question'] = (p['input_tokens'] + p['output_tokens']) / len(predictions)
    p['mean_tokens_per_call'] = (p['input_tokens'] + p['output_tokens']) / len(calls)
    p['pass1_early_exit_rate'] = sum(r['pass_count'] == 1 for r in predictions) / len(predictions)
    p['pass2_question_rate'] = sum(r['pass_count'] == 2 for r in predictions) / len(predictions)


def evaluate():
    _, _, inputs, predictions, calls = completed()
    assert digest(GOLD) == read_json(ROOT / 'evaluation_data/dev_v1/manifest.json')['sha256']
    gold, audit = rows(GOLD), rows(DEV / 'sufficiency_audit.jsonl')
    assert all(a['core_support'] in {'sufficient', 'partial', 'mention', 'none', 'uncertain'} for a in audit)
    report = v1.summarize(gold, inputs, predictions, audit, calls)
    assert report['counts']['gold_in_pool'] == 30, 'Original candidate ceiling changed'
    report['metric_notes'].update(independent_review=AUDIT_POLICY,
        answerable_precision='Independent AI review of entire selected bundle; core-only review reported separately; not human-verified truth',
        evidence_selection_accuracy='UNCHANGED original core overlap, never bundle overlap',
        conditional_gate_recall='UNCHANGED 30 original Top20 core-window hits; added neighbor content does not enlarge denominator')
    old = read_json(OLD / 'summary.json')
    performance_extra(report, predictions, calls)
    performance_extra(old, rows(OLD / 'results.jsonl'), rows(OLD / 'calls.jsonl'))
    source, pred, verdict = ({r['test_id']: r for r in data} for data in (inputs, predictions, audit))
    tests = {r['id']: r for r in gold}
    original = {r['test_id']: r for r in rows(OLD / 'results.jsonl')}
    recovery = []
    for d in rows(DIAG / 'cases.jsonl'):
        tid = d['test_id']
        selected = next((c for c in source[tid]['candidates'] if c['candidate_id'] == pred[tid]['best_candidate_id']), None)
        recovery.append({'test_id': tid, 'diagnosis': d['primary_type'], 'now_answerable': pred[tid]['answerable'],
            'selected_core_gold_hit': bool(selected and v1.temporal_hit(selected, tests[tid]['gold_evidence'])),
            'independent_bundle_support': verdict.get(tid, {}).get('support'),
            'independent_core_support': verdict.get(tid, {}).get('core_support'),
            'confirmed_recovery': bool(selected and v1.temporal_hit(selected, tests[tid]['gold_evidence'])
                                       and verdict[tid]['support'] == 'sufficient')})
    report['baseline'] = {'metrics': old['metrics'], 'failures': old['failures'], 'performance': old['performance']}
    report['metric_delta'] = {k: report['metrics'][k] - old['metrics'][k] for k in report['metrics']}
    report['diagnosis'] = read_json(DIAG / 'summary.json')['primary_counts']
    report['original_false_negative_recovery'] = recovery
    report['recovery_by_type'] = {kind: {'total': sum(r['diagnosis'] == kind for r in recovery),
        'now_answerable': sum(r['diagnosis'] == kind and r['now_answerable'] for r in recovery),
        'confirmed_recovery': sum(r['diagnosis'] == kind and r['confirmed_recovery'] for r in recovery)}
        for kind in {r['diagnosis'] for r in recovery}}
    report['new_rejections_of_previous_positives'] = [i for i in original if original[i]['answerable'] and not pred[i]['answerable']]
    report['false_positive_ids'] = [i for i in tests if not tests[i]['answerable'] and pred[i]['answerable']]
    report['remaining_false_negative_ids'] = [i for i in tests if tests[i]['answerable'] and not pred[i]['answerable']
        and any(v1.temporal_hit(c, tests[i]['gold_evidence']) for c in source[i]['candidates'])]
    report['core_only_audit'] = {'support_counts': dict(Counter(a['core_support'] for a in audit)),
        'precision': sum(tests[a['test_id']]['answerable'] and a['core_support'] == 'sufficient' for a in audit) / len(audit),
        'warning': 'Bundle sufficient does not prove the unchanged core playback alone is sufficient.'}
    limits = {'conditional_gate_recall': .8, 'answerable_precision_independent_audit': .85,
              'no_answer_accuracy': .85, 'evidence_selection_accuracy': .9}
    report['reference_lines'] = {k: {'required': n, 'actual': report['metrics'][k], 'met': report['metrics'][k] >= n}
                                 for k, n in limits.items()}
    report['all_reference_lines_met'] = all(v['met'] for v in report['reference_lines'].values())
    report.update(gate_frozen=False, holdout_run=False, audit_sha256=digest(DEV / 'sufficiency_audit.jsonl'),
                  pipeline_unchanged=True, ground_truth_unchanged=True)
    save_new(DEV / 'summary.json', report)
    with (DEV / 'summary.md').open('x', encoding='utf-8') as handle:
        handle.write('# Answer Gate v2 Bundle — dev only\n\nNo Holdout run; no Gate freeze.\n\n```json\n' +
                     json.dumps(report, ensure_ascii=False, indent=2) + '\n```\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'audit-export', 'evaluate'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'audit-export': export_audit, 'evaluate': evaluate}[args.action]()
