"""Single-user durable ingestion; frozen corpora are read-only, live assets are separate."""
import copy
import json
import re
import sys
import threading
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from library.server import AppError, atomic_write_json, load_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {'queued', 'captions', 'chapters', 'indexing'}
VIDEO_WORKERS = 2
CHAPTER_WORKERS = 3


def youtube_id(url):
    try:
        p = urlparse(url)
        if p.scheme != 'https' or p.username or p.password or p.port:
            raise ValueError()
        if p.hostname == 'youtu.be':
            value = p.path.strip('/')
        elif p.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
            value = parse_qs(p.query).get('v', [''])[0] if p.path == '/watch' else (
                p.path.split('/')[2] if re.fullmatch(r'/(shorts|live)/[^/]+', p.path) else '')
        else:
            raise ValueError()
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
            raise ValueError()
        return value
    except (TypeError, ValueError):
        raise AppError('请输入有效的 HTTPS YouTube 视频链接。', code='youtube_url_invalid')


def fetch_captions(video_id, folder):
    from evaluation.english_v1 import parse_vtt
    # Reuse original public captions already present locally; never mutate frozen data.
    existing = ROOT / 'evaluation_data/mixed50_v1/normalized' / f'{video_id}.json'
    if existing.exists():
        return load_json(existing, None)
    sys.path.insert(0, str(ROOT / 'work/caption-tools'))
    try:
        import yt_dlp
    except ImportError:
        raise AppError('本地字幕工具未安装，请配置 yt-dlp 后重试。')
    options = {'quiet': True, 'no_warnings': True, 'skip_download': True, 'noplaylist': True,
               'socket_timeout': 20, 'retries': 1, 'extractor_retries': 1,
               'subtitlesformat': 'vtt', 'outtmpl': str(folder / '%(id)s.%(ext)s')}
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            if info.get('id') != video_id or info.get('is_live') or not 0 < (info.get('duration') or 0) <= 10800:
                raise AppError('当前支持有完整字幕、时长不超过 3 小时的非直播视频。')
            selected = None
            for key, kind in [('subtitles', 'manual'), ('automatic_captions', 'auto')]:
                tracks = info.get(key) or {}
                languages = sorted(tracks, key=lambda x: (x not in ('zh-Hans', 'zh-Hant', 'en', 'en-orig'), x))
                for lang in languages:
                    if not re.fullmatch(r'[a-zA-Z0-9-]+', lang):
                        continue
                    if lang.startswith(('zh', 'en')) and any(t.get('ext') == 'vtt' and 'tlang=' not in t.get('url', '') for t in tracks[lang]):
                        selected = (lang, kind)
                        break
                if selected:
                    break
            if not selected:
                raise AppError('未找到可用的中英文原字幕；视频已收藏，可稍后重试。')
            lang, kind = selected
            downloader.params.update(writesubtitles=kind == 'manual', writeautomaticsub=kind == 'auto', subtitleslangs=[lang])
            downloader.process_ie_result(info, download=True)
        cues = parse_vtt((folder / f'{video_id}.{lang}.vtt').read_text(encoding='utf-8'), info['duration'])
        return {'video_id': video_id, 'title': info['title'], 'channel': info.get('channel') or '',
                'url': f'https://www.youtube.com/watch?v={video_id}', 'duration': info['duration'],
                'subtitle_language': lang, 'subtitle_type': kind, 'segments': cues}
    except AppError:
        raise
    except Exception as exc:
        # Do not expose provider URLs, credentials or raw platform responses in the UI.
        raise AppError('字幕获取失败：平台限制、视频不可用或网络异常。请稍后重试。') from exc


def prepare(video_id, folder, progress, pipeline):
    from evaluation.chapter_pilot import (BOUNDARIES, SINGLE_LABEL, SPLIT_PROMPT, MERGE_PROMPT,
                                          LABEL_PROMPT, units, materialize)
    from library.intent_pipeline_v2 import validate
    from library.mvp_pipeline import encode
    from llm.provider import generate_json
    folder.mkdir(parents=True, exist_ok=True)
    progress('captions', '正在获取字幕')
    source = load_json(folder / 'source.json', None)
    if source is None:
        source = fetch_captions(video_id, folder)
        atomic_write_json(folder / 'source.json', source)
    cues = source['segments']
    payload = [{'id': i, 'text': c['text']} for i, c in enumerate(cues)]
    # ponytail: bounded full-transcript processing; hierarchical chapter splitting needed above this limit.
    if len(json.dumps(payload, ensure_ascii=False)) > 200000:
        raise AppError('字幕超过本版整理上限，已保留收藏；不会截断内容冒充完成。')

    def generate(name, prompt, data, schema, check, tokens):
        path = folder / f'{name}.json'
        saved = load_json(path, None)
        if saved is not None:
            validate(saved['value'], schema)
            check(saved['value'])
            return saved['value']
        result = generate_json(prompt, json.dumps(data, ensure_ascii=False), schema=schema, max_tokens=tokens)
        try:
            validate(result.value, schema)
            check(result.value)
        except Exception:
            atomic_write_json(folder / f'rejected-{name}-{uuid.uuid4().hex}.json',
                              {'value': result.value, 'usage': result.usage()})
            raise
        atomic_write_json(path, {'value': result.value, 'usage': result.usage()})
        return result.value

    progress('chapters', '正在划分完整内容章节')
    split = generate('split', SPLIT_PROMPT, {'cues': payload}, BOUNDARIES,
                     lambda x: units(cues, x['starts']), 2000)

    def check_merge(value):
        units(cues, value['starts'])
        if not set(value['starts']) <= set(split['starts']):
            raise AppError('章节整合边界无效，请重试。')

    merged = generate('merge', MERGE_PROMPT, {'proposed_starts': split['starts'], 'cues': payload},
                      BOUNDARIES, check_merge, 2000)
    spans = units(cues, merged['starts'])
    def label_unit(a, b):
        def check_label(value):
            rows = value['chapters']
            if len(rows) != 1 or rows[0]['start_cue'] != a:
                raise AppError('章节归纳边界无效，请重试。')
            rows[0]['points'].sort(key=lambda p: p['start_cue'])
            ids = [p['start_cue'] for p in rows[0]['points']]
            if any(not a <= i <= b for i in ids):
                raise AppError('章节关键点越界，请重试。')
            if not ids:
                raise AppError('章节关键点缺失，请重试。')
            # Multiple ideas can share one original cue; render one navigation anchor, not an error.
            points = {}
            for point in rows[0]['points']:
                points.setdefault(point['start_cue'], []).append(point['title'])
            rows[0]['points'] = [{'start_cue': cue, 'title': '；'.join(dict.fromkeys(titles))[:240]}
                                 for cue, titles in points.items()]

        label = generate(f'label-{a}', LABEL_PROMPT,
                         {'units': [{'start_cue': a, 'end_cue': b, 'cues': payload[a:b+1]}]},
                         SINGLE_LABEL, check_label, 1800)
        return label['chapters'][0]

    labels = [None] * len(spans)
    progress('chapters', f'正在并行归纳章节 · 已完成 0/{len(spans)}')
    pool = ThreadPoolExecutor(max_workers=CHAPTER_WORKERS, thread_name_prefix='clipcompass-chapter')
    try:
        futures = {pool.submit(label_unit, a, b): i for i, (a, b) in enumerate(spans)}
        for completed, future in enumerate(as_completed(futures), 1):
            labels[futures[future]] = future.result()
            progress('chapters', f'正在并行归纳章节 · 已完成 {completed}/{len(spans)}')
    finally:
        # Finish in-flight writes before marking failure/retry; cancel work not yet started.
        pool.shutdown(wait=True, cancel_futures=True)
    chapters = materialize(video_id, cues, merged['starts'], {'chapters': labels})
    progress('indexing', '正在建立检索索引')
    with pipeline.lock:
        vectors = encode(pipeline.dense_model, pipeline.dense_tokenizer,
                         [c['english_summary'] for c in chapters], query=False, batch_size=16)
    return {'source': source, 'chapters': chapters, 'vectors': vectors.tolist()}


class Ingestion:
    def __init__(self, root, base, processor=prepare):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.claim()
        self.token = secrets.token_urlsafe(32)
        self.base = base
        self.snapshot = base
        self.processor = processor
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.threads = []
        self.jobs_path = self.root / 'jobs.json'
        self.jobs = load_json(self.jobs_path, {})
        from library.obsidian import ObsidianExport
        self.obsidian = ObsidianExport(self.root)
        self.obsidian_error = ''
        for job in self.jobs.values():
            if job['status'] in ACTIVE:
                job.update(status='queued', message='服务恢复，等待继续整理')
        self.bundles = {}
        for video_id, job in self.jobs.items():
            if job['status'] == 'ready':
                try:
                    self.publish(video_id, load_json(self.root / video_id / 'bundle.json', None))
                except Exception:
                    job.update(status='failed', message='本地索引读取失败，请重试恢复')
        self.persist()

    def persist(self):
        atomic_write_json(self.jobs_path, self.jobs)

    def claim(self):
        # One service owns this live library. Do not run competing writers on another port.
        self.lease = (self.root / 'worker.lock').open('a+b')
        try:
            if sys.platform == 'win32':
                import msvcrt
                self.lease.seek(0)
                msvcrt.locking(self.lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lease.close()
            raise RuntimeError('此收藏整理目录已被另一服务使用。')

    def start(self):
        if self.threads or self.stopping.is_set():
            return
        self.sync_obsidian()
        self.threads = [threading.Thread(target=self.work, name=f'clipcompass-import-{i}', daemon=True)
                        for i in range(VIDEO_WORKERS)]
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stopping.set()
        self.wake.set()
        for thread in self.threads:
            thread.join(timeout=1)
        if not any(thread.is_alive() for thread in self.threads):
            self.lease.close()

    def enqueue(self, body):
        video_id = youtube_id(body.get('source_url'))
        with self.lock:
            if video_id in {c['video_id'] for c in self.base.chapters}:
                return {'video_id': video_id, 'status': 'ready', 'message': '已纳入检索'}
            previous = self.jobs.get(video_id)
            if previous and (previous['status'] != 'failed' or body.get('retry') is not True):
                return copy.deepcopy(previous)
            if sum(j['status'] in ACTIVE for j in self.jobs.values()) >= 100:
                raise AppError('整理队列已满，请稍后再试。', status=429)
            self.jobs[video_id] = {'video_id': video_id, 'source_url': f'https://www.youtube.com/watch?v={video_id}',
                'title': str(body.get('title') or 'YouTube 视频')[:240], 'status': 'queued',
                'message': '已排队，等待整理', 'saved_at': (previous or {}).get('saved_at', utc_now())}
            self.persist()
            self.wake.set()
            return copy.deepcopy(self.jobs[video_id])

    def progress(self, video_id, status, message):
        with self.lock:
            self.jobs[video_id].update(status=status, message=message, updated_at=utc_now())
            self.persist()

    def publish(self, video_id, bundle):
        import numpy as np
        if not bundle or bundle['source']['video_id'] != video_id or not bundle['chapters']:
            raise ValueError('Invalid bundle')
        vectors = np.asarray(bundle['vectors'], dtype=np.float32)
        if vectors.shape != (len(bundle['chapters']), 384) or not np.isfinite(vectors).all():
            raise ValueError('Invalid vectors')
        bundles = {**self.bundles, video_id: bundle}
        snapshot = copy.copy(self.base)
        snapshot.pipeline = copy.copy(self.base.pipeline)
        snapshot.pipeline.metadata = dict(self.base.pipeline.metadata)
        snapshot.pipeline.captions = dict(self.base.pipeline.captions)
        snapshot.chapters = list(self.base.chapters)
        vector_rows = [self.base.chapter_vectors]
        for item in bundles.values():
            source = item['source']
            snapshot.pipeline.metadata[source['video_id']] = source
            snapshot.pipeline.captions[source['video_id']] = source['segments']
            snapshot.chapters.extend(item['chapters'])
            vector_rows.append(np.asarray(item['vectors'], dtype=np.float32))
        snapshot.by_id = {c['chapter_id']: c for c in snapshot.chapters}
        snapshot.chapter_vectors = np.concatenate(vector_rows)
        # Immutable snapshots keep searches consistent while a background import finishes.
        self.bundles, self.snapshot = bundles, snapshot

    def run_one(self):
        with self.lock:
            if self.stopping.is_set():
                return False
            video_id = next((v for v, j in self.jobs.items() if j['status'] == 'queued'), None)
            if video_id is None:
                return False
            self.progress(video_id, 'captions', '正在获取字幕')
        try:
            folder = self.root / video_id
            bundle = load_json(folder / 'bundle.json', None)
            if bundle is None:
                bundle = self.processor(video_id, folder,
                    lambda status, message: self.progress(video_id, status, message), self.base.pipeline)
                atomic_write_json(folder / 'bundle.json', bundle)
            with self.lock:
                self.publish(video_id, bundle)
                self.jobs[video_id]['title'] = bundle['source']['title']
                self.progress(video_id, 'ready', f"已纳入检索 · {len(bundle['chapters'])} 个内容单元")
            self.sync_obsidian()
        except Exception as exc:
            message = str(exc) if isinstance(exc, AppError) else '整理失败，已保存完成的步骤；请点击重试。'
            self.progress(video_id, 'failed', message)
        return True

    def work(self):
        while not self.stopping.is_set():
            self.wake.clear()
            if not self.run_one():
                self.wake.wait(5)

    def library(self, value):
        with self.lock:
            value = copy.deepcopy(value)
            value['imports_enabled'] = True
            value['obsidian'] = {**self.obsidian.status(), 'error': self.obsidian_error}
            value['import_token'] = self.token
            value['imports'] = copy.deepcopy(list(self.jobs.values()))
            collection = value['collections'][0]
            for video_id, bundle in self.bundles.items():
                if video_id in collection['video_ids']:
                    continue
                source = bundle['source']
                collection['video_ids'].append(video_id)
                value['videos'].append({'video_id': video_id, 'title': source['title'], 'creator': source['channel'],
                    'source_url': source['url'], 'duration': source['duration'], 'chapters': bundle['chapters'],
                    'subtitle_type': source['subtitle_type'], 'segments': [], 'platform': 'youtube'})
            value['title'] = collection['title'] = f"我的视频收藏 · {len(collection['video_ids'])} 个视频"
            return value

    def search(self, question, allowed_video_ids):
        return self.snapshot.search(question, allowed_video_ids)

    def sync_obsidian(self):
        try:
            snapshot = self.snapshot
            grouped = {}
            for chapter in snapshot.chapters:
                grouped.setdefault(chapter['video_id'], []).append(chapter)
            for video_id, chapters in grouped.items():
                source = {**snapshot.pipeline.metadata[video_id], 'segments': snapshot.pipeline.captions[video_id]}
                self.obsidian.export(source, chapters)
            self.obsidian_error = ''
        except Exception:
            self.obsidian_error = '笔记同步未完成，请检查本地磁盘后重新同步；视频检索不受影响。'

    def configure_obsidian(self, body):
        self.obsidian.configure(body)
        self.sync_obsidian()
        return {**self.obsidian.status(), 'error': self.obsidian_error}
