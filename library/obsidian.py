"""Opt-in, one-way Markdown export. Existing notes are never overwritten."""
import html
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path

from library.server import AppError, atomic_write_json, load_json, utc_now


def plain(value):
    value = html.escape(' '.join(str(value).split()), quote=False)
    return re.sub(r'([\\`*_{}\[\]()#!|^])', r'\\\1', value)


def stamp(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid timestamp')
    return f'{int(seconds)//60}:{int(seconds)%60:02}'


def documents(source, chapters):
    video_id = source['video_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
        raise ValueError('Invalid video ID')
    title = str(source['title'])
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\[\]#^]', '_', title)[:80].strip(' .') or '视频'
    stem = f'{name} [{video_id}]'
    notes_name, captions_name = f'{stem}.md', f'{stem} - 原始字幕.md'
    url = f'https://www.youtube.com/watch?v={video_id}'
    metadata = '\n'.join(f'{k}: {json.dumps(v, ensure_ascii=False)}' for k, v in {
        'title': title, 'video_id': video_id, 'source': url, 'creator': source.get('channel', ''),
        'subtitle_type': source.get('subtitle_type', 'unknown'),
        'subtitle_language': source.get('subtitle_language', 'unknown'), 'generated_by': 'ClipCompass'}.items())
    header = f'---\n{metadata}\n---\n\n# {plain(title)}\n\n[观看原视频]({url})\n\n'
    # Fixed local links contain only filenames we generated, never transcript-supplied targets.
    from urllib.parse import quote
    notes = header + '> 以下为 AI 根据字幕归纳的章节，不是原文，也不代表已理解视频画面。\n\n'
    notes += f'[查看全部原始字幕]({quote(captions_name)})\n\n'
    for index, chapter in enumerate(chapters, 1):
        start, end = chapter['start'], chapter['end']
        notes += f"## {index}. {plain(chapter['title'])}\n\n"
        notes += f'[{stamp(start)}–{stamp(end)} · 看原视频]({url}&t={int(start)}s) · '
        notes += f"[核对原字幕]({quote(captions_name)}#%5Ecue-{chapter['start_cue']})\n\n"
        notes += plain(chapter.get('summary', '')) + '\n\n'
        for point in chapter.get('points', []):
            notes += f"- [{stamp(point['start'])}]({url}&t={int(point['start'])}s) {plain(point['title'])}\n"
        notes += '\n'
    notes += '## 我的批注\n\n'
    captions = header + '> 以下为获取到的字幕文本（可能为自动字幕，未人工校订），不是 AI 归纳。\n\n'
    captions += f'[返回章节笔记]({quote(notes_name)})\n\n'
    for index, cue in enumerate(source['segments']):
        captions += f"[{stamp(cue['start'])}–{stamp(cue['end'])}]({url}&t={int(cue['start'])}s)\n\n"
        captions += f"> {plain(cue['text'])}\n\n^cue-{index}\n\n"
    return {captions_name: captions, notes_name: notes}


def vault_path(value):
    if not isinstance(value, str) or not value.strip():
        raise AppError('请填写已有 Obsidian 库的完整路径。')
    path = Path(value.strip())
    if not path.is_absolute() or not path.is_dir() or not (path / '.obsidian').is_dir():
        raise AppError('请选择已有的 Obsidian 库文件夹，其中应包含 .obsidian 文件夹。')
    path = path.resolve()
    if path == path.parent:
        raise AppError('不能使用磁盘根目录。')
    return path


def create_note(path, text):
    """Atomic create-only publication, including races with an editor creating the same path."""
    if path.is_symlink() or path.is_junction():
        raise AppError('笔记路径是链接，已停止写入。')
    payload = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            raise AppError('已有同名笔记，已保留原内容，不覆盖批注。请将冲突笔记改名后重试。')
        return
    fd, temporary = tempfile.mkstemp(prefix='.clipcompass-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # Unlike replace(), this never overwrites an existing note.
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise AppError('笔记发生写入冲突，已保留原内容。')
    finally:
        Path(temporary).unlink(missing_ok=True)


class ObsidianExport:
    def __init__(self, root):
        self.path = Path(root) / 'obsidian.json'
        self.lock = threading.RLock()
        self.config = load_json(self.path, {'enabled': False, 'vault': '', 'items': {}})

    def status(self):
        with self.lock:
            items = list(self.config.get('items', {}).values())
            return {'enabled': self.config['enabled'], 'vault': self.config['vault'],
                    'synced': sum(v['status'] == 'synced' for v in items),
                    'errors': [v for v in items if v['status'] == 'failed']}

    def configure(self, body):
        if type(body.get('enabled')) is not bool:
            raise AppError('同步开关无效。')
        with self.lock:
            if body['enabled']:
                vault = str(vault_path(body.get('vault')))
                if vault != self.config['vault']:
                    self.config['items'] = {}
                self.config['vault'] = vault
            self.config['enabled'] = body['enabled']
            atomic_write_json(self.path, self.config)

    def export(self, source, chapters):
        with self.lock:
            if not self.config['enabled']:
                return
            video_id = source['video_id']
            previous = self.config.get('items', {}).get(video_id)
            # Export once. Later edits to the exported notes belong to the user.
            if previous and previous['status'] == 'synced':
                return
            result = {'video_id': video_id, 'title': source['title'], 'updated_at': utc_now()}
            try:
                vault = vault_path(self.config['vault'])
                folder = vault / 'ClipCompass'
                if folder.is_symlink() or folder.is_junction():
                    raise AppError('ClipCompass 文件夹不能是链接或目录联接。')
                folder.mkdir(exist_ok=True)
                if folder.resolve().parent != vault:
                    raise AppError('输出目录不在指定知识库内。')
                for name, text in documents(source, chapters).items():
                    create_note(folder / name, text)
                result.update(status='synced', message='已同步')
            except (OSError, ValueError, KeyError, AppError) as exc:
                result.update(status='failed', message=str(exc) if isinstance(exc, AppError)
                              else '同步失败：请检查库目录、写入权限和磁盘空间。')
            self.config.setdefault('items', {})[video_id] = result
            atomic_write_json(self.path, self.config)
