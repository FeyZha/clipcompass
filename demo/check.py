import json, pathlib
d = json.loads(pathlib.Path(__file__).with_name('data.json').read_text('utf-8'))
assert len(d['library']['videos']) == 8
assert len(d['cases']) == 14
assert any(c['error'] for c in d['cases'])
assert any(c['response'].get('answerable') for c in d['cases'])
assert any(c['response'] and not c['response']['answerable'] for c in d['cases'])
assert len({c['question'] for c in d['cases']}) == 14
print('14 frozen scenarios: normal, no evidence, clarification OK')
