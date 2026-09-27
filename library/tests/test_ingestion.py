import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from library.ingestion import Ingestion, youtube_id, prepare
from library.server import AppError, AppServer, LibraryStore, STATIC_ROOT
from library.chapter_pipeline import ChapterPipeline
from llm.provider import GenerationResult

VIDEO = '79zqFG7zdnA'
URL = 'https://www.youtube.com/watch?v=' + VIDEO


def base():
    value = ChapterPipeline.__new__(ChapterPipeline)
    value.chapters = []
    value.chapter_vectors = np.zeros((0, 384))
    value.parents = {}; value.by_id = {}; value.logger = Mock()
    value.pipeline = SimpleNamespace(metadata={}, captions={}, lock=threading.Lock(),
        chunks=[], vectors=np.zeros((0, 384)), dense_model=None, dense_tokenizer=None,
        reranker=None, reranker_tokenizer=None)
    return value


def bundle():
    return {'source': {'video_id': VIDEO, 'title': 'Test video', 'channel': 'Test', 'url': URL,
        'duration': 20, 'subtitle_type': 'manual', 'segments': [{'start': 0, 'end': 20, 'text': 'original evidence'}]},
        'chapters': [{'video_id': VIDEO, 'chapter_id': 'ch:' + VIDEO + ':0', 'title': 'Test chapter',
            'english_summary': 'summary', 'kind': 'teaching', 'start_cue': 0, 'end_cue': 0, 'start': 0, 'end': 20}],
        'vectors': [[1.] + [0.] * 383]}


class IngestionTests(unittest.TestCase):
    def test_two_video_workers_claim_once_and_preserve_both_publications(self):
        entered = threading.Barrier(3)
        release = threading.Event()
        calls = []
        ids = [VIDEO, 'xdodZEttvSY', 'FVmVP9CCRcU']
        def process(video_id, folder, progress, pipeline):
            calls.append(video_id)
            if video_id in ids[:2]:
                entered.wait(timeout=5)
                if not release.wait(5): raise RuntimeError('test timed out')
            value = bundle()
            value['source']['video_id'] = video_id
            value['chapters'][0].update(video_id=video_id, chapter_id='ch:'+video_id+':0')
            return value
        with tempfile.TemporaryDirectory() as temp:
            manager = Ingestion(temp, base(), process)
            try:
                for video_id in ids:
                    manager.enqueue({'source_url':'https://youtu.be/'+video_id})
                manager.start(); manager.start()
                entered.wait(timeout=5)
                self.assertEqual(len(manager.threads), 2)
                self.assertCountEqual(calls, ids[:2])
                self.assertEqual(manager.jobs[ids[2]]['status'], 'queued')
                # Stopping does not discard the third queued job, and both in-flight commits survive.
                manager.stopping.set(); release.set()
                for thread in manager.threads: thread.join(timeout=5)
                self.assertTrue(all(not t.is_alive() for t in manager.threads))
                self.assertEqual(set(manager.bundles), set(ids[:2]))
                self.assertEqual(len(manager.snapshot.chapters), 2)
                self.assertEqual(manager.jobs[ids[2]]['status'], 'queued')
            finally:
                release.set(); manager.close()

    def test_three_chapter_workers_order_progress_and_resume(self):
        source = bundle()['source']
        source['segments'] = [{'start':i*10,'end':i*10+9,'text':str(i)} for i in range(6)]
        barrier = threading.Barrier(3)
        calls = []
        def generate(prompt, data, **kwargs):
            payload = json.loads(data)
            if 'units' not in payload:
                value = {'starts':list(range(6))}
            else:
                a = payload['units'][0]['start_cue']
                calls.append(a)
                barrier.wait(timeout=5)  # Fails if chapter processing is still serial or below 3.
                value = {'chapters':[{'start_cue':a,'title':str(a),'summary':'原文',
                    'english_summary':'original','kind':'teaching','points':[{'title':'要点','start_cue':a}]}]}
            return GenerationResult(value,'deepseek-flash',1,1,0,1)
        with tempfile.TemporaryDirectory() as temp, \
             patch('library.ingestion.fetch_captions', return_value=source), \
             patch('llm.provider.generate_json', side_effect=generate) as model, \
             patch('library.mvp_pipeline.encode', return_value=np.zeros((6,384))):
            progress = Mock()
            value = prepare(VIDEO, Path(temp), progress, base().pipeline)
            self.assertCountEqual(calls, list(range(6)))
            self.assertEqual([c['start_cue'] for c in value['chapters']], list(range(6)))
            messages = [c.args[1] for c in progress.call_args_list if '已完成' in c.args[1]]
            self.assertEqual(messages, [f'正在并行归纳章节 · 已完成 {i}/6' for i in range(7)])
            self.assertEqual(model.call_count, 8)
            prepare(VIDEO, Path(temp), Mock(), base().pipeline)
            self.assertEqual(model.call_count, 8)

    def test_url_boundaries(self):
        self.assertEqual(youtube_id(URL), VIDEO)
        for bad in [None, 'file:///tmp/a', 'http://youtube.com/watch?v=' + VIDEO,
                    'https://youtube.com.evil.test/watch?v=' + VIDEO,
                    'https://user@youtube.com/watch?v=' + VIDEO,
                    'https://youtube.com:443/watch?v=' + VIDEO, 'https://youtu.be/../../a']:
            with self.assertRaises(AppError): youtube_id(bad)

    def test_chapter_failure_joins_writers_and_retries_only_missing_unit(self):
        source = bundle()['source']
        source['segments'] = [{'start':i*10,'end':i*10+9,'text':str(i)} for i in range(3)]
        barrier = threading.Barrier(3)
        calls = []
        failing = True
        def generate(prompt, data, **kwargs):
            payload = json.loads(data)
            if 'units' not in payload:
                value = {'starts':[0,1,2]}
            else:
                a = payload['units'][0]['start_cue']; calls.append(a)
                if failing:
                    barrier.wait(timeout=5)
                    if a == 0: raise AppError('simulated provider failure')
                value = {'chapters':[{'start_cue':a,'title':str(a),'summary':'原文',
                    'english_summary':'original','kind':'teaching','points':[{'title':'要点','start_cue':a}]}]}
            return GenerationResult(value,'deepseek-flash',1,1,0,1)
        with tempfile.TemporaryDirectory() as temp, \
             patch('library.ingestion.fetch_captions', return_value=source), \
             patch('llm.provider.generate_json', side_effect=generate), \
             patch('library.mvp_pipeline.encode', return_value=np.zeros((3,384))):
            with self.assertRaises(AppError): prepare(VIDEO, Path(temp), Mock(), base().pipeline)
            self.assertTrue((Path(temp)/'label-1.json').exists())
            self.assertTrue((Path(temp)/'label-2.json').exists())
            failing = False
            value = prepare(VIDEO, Path(temp), Mock(), base().pipeline)
            self.assertEqual(len(value['chapters']), 3)
            self.assertEqual([calls.count(i) for i in range(3)], [2,1,1])

    def test_durable_queue_publication_and_search(self):
        with tempfile.TemporaryDirectory() as temp:
            processor = Mock(return_value=bundle())
            manager = Ingestion(temp, base(), processor)
            body = {'source_url': URL}
            first = manager.enqueue(body)
            self.assertEqual(manager.enqueue(body), first)
            old_snapshot = manager.snapshot
            self.assertEqual(manager.jobs[VIDEO]['status'], 'queued')
            self.assertTrue(manager.run_one())
            self.assertFalse(manager.run_one())
            self.assertEqual(processor.call_count, 1)
            self.assertEqual(old_snapshot.chapters, [])
            value = manager.library({'collections': [{'video_ids': []}], 'videos': []})
            self.assertEqual(value['collections'][0]['video_ids'], [VIDEO])
            self.assertEqual(value['imports'][0]['status'], 'ready')
            plan = {'task_type':'specific','english_query':'test','core_query':'test','directions':[], 'constraints':[]}
            answer = GenerationResult({'items':[{'chapter_id':'ch:'+VIDEO+':0','reason':'evidence',
                'key_start_cue':0,'key_end_cue':0,'covered_directions':[]}]}, 'deepseek-flash',1,1,0,1)
            with patch('library.chapter_pipeline.plan_request', return_value=(plan, {})), \
                 patch('library.chapter_pipeline.encode', return_value=np.array([[1.]+[0.]*383])), \
                 patch('library.chapter_pipeline.score_pairs', return_value=[1.]), \
                 patch('library.chapter_pipeline.generate_json', return_value=answer):
                self.assertEqual(manager.search('test', {VIDEO})['results'][0]['video_id'], VIDEO)
                self.assertFalse(manager.search('test', set())['answerable'])
            manager.close()
            restored = Ingestion(temp, base(), processor)
            self.assertEqual(len(restored.snapshot.chapters), 1)
            self.assertFalse(restored.run_one())
            restored.close()

    def test_failure_retry_resume_and_single_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            processor = Mock(side_effect=AppError('No captions'))
            manager = Ingestion(temp, base(), processor)
            manager.enqueue({'source_url': URL})
            manager.run_one()
            self.assertEqual(manager.jobs[VIDEO]['status'], 'failed')
            self.assertEqual(manager.snapshot.chapters, [])
            self.assertEqual(manager.enqueue({'source_url': URL})['status'], 'failed')
            manager.enqueue({'source_url': URL, 'retry': True})
            manager.progress(VIDEO, 'chapters', 'interrupted')
            with self.assertRaises(RuntimeError): Ingestion(temp, base(), processor)
            manager.close()
            restored = Ingestion(temp, base(), Mock(return_value=bundle()))
            self.assertEqual(restored.jobs[VIDEO]['status'], 'queued')
            restored.run_one()
            self.assertEqual(restored.jobs[VIDEO]['status'], 'ready')
            restored.close()

    def test_http_auth_validation_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = Ingestion(Path(temp)/'live', base(), Mock(return_value=bundle()))
            server = AppServer(('127.0.0.1', 0), LibraryStore(Path(temp)/'store'), STATIC_ROOT, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                for path in ('/api/search', '/api/videos', '/api/events', '/api/imports', '/api/obsidian'):
                    for content_type in ('text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data', 'text/plain; application/json'):
                        req = urllib.request.Request(url+path, data=b'{}', headers={
                            'Content-Type': content_type, 'Origin': 'https://evil.example',
                            'X-ClipCompass-Import': manager.token})
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            urllib.request.urlopen(req)
                        self.assertEqual(error.exception.code, 415)
                        error.exception.close()
                self.assertEqual(manager.jobs, {})
                req = urllib.request.Request(url+'/api/library', headers={'Host':'evil.example'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(req)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
                def post(token):
                    req = urllib.request.Request(url+'/api/imports', data=json.dumps({'source_url': URL}).encode(),
                        headers={'Content-Type':'application/json','X-ClipCompass-Import':token})
                    return urllib.request.urlopen(req)
                with self.assertRaises(urllib.error.HTTPError) as error: post('wrong')
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
                with post(manager.token) as response: self.assertEqual(response.status, 202)
                with urllib.request.urlopen(url+'/api/library') as response:
                    data = json.load(response)
                self.assertEqual(data['imports'][0]['status'], 'queued')
                self.assertEqual(data['videos'], [])
                manager.run_one()
                with urllib.request.urlopen(url+'/api/library') as response:
                    self.assertEqual(json.load(response)['videos'][0]['video_id'], VIDEO)
            finally:
                server.shutdown(); server.server_close(); thread.join(); manager.close()

    def test_prepare_checkpoints_reuse_without_new_model_calls(self):
        label = {'start_cue':0,'title':'章节','summary':'原文归纳','english_summary':'original summary',
                 'kind':'teaching','points':[{'title':'要点','start_cue':0},{'title':'同字幕另一要点','start_cue':0}]}
        results = [GenerationResult(v,'deepseek-flash',1,1,0,1) for v in
                   [{'starts':[0]},{'starts':[0]},{'chapters':[label]}]]
        with tempfile.TemporaryDirectory() as temp, \
             patch('library.ingestion.fetch_captions', return_value=bundle()['source']) as fetch, \
             patch('llm.provider.generate_json', side_effect=results) as generate, \
             patch('library.mvp_pipeline.encode', return_value=np.array(bundle()['vectors'])):
            for _ in range(2):
                value = prepare(VIDEO, Path(temp), Mock(), base().pipeline)
                self.assertEqual(value['chapters'][0]['start'], 0)
                self.assertEqual(len(value['chapters'][0]['points']), 1)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(generate.call_count, 3)


if __name__ == '__main__': unittest.main()
