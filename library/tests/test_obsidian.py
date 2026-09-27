import tempfile
import unittest
from pathlib import Path

from library.obsidian import ObsidianExport, documents, vault_path
from library.server import AppError
from library.tests.test_ingestion import bundle
from library.tests.test_ingestion import base, VIDEO, URL
from library.ingestion import Ingestion
from unittest.mock import Mock, patch


class ObsidianTests(unittest.TestCase):
    def test_new_ready_video_auto_exports_and_disk_failure_keeps_searchable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); vault = root/'vault'; vault.mkdir(); (vault/'.obsidian').mkdir()
            manager = Ingestion(root/'live', base(), Mock(return_value=bundle()))
            try:
                manager.configure_obsidian({'enabled':True,'vault':str(vault)})
                manager.enqueue({'source_url':URL}); manager.run_one()
                self.assertEqual(manager.obsidian.status()['synced'], 1)
                self.assertEqual(manager.jobs[VIDEO]['status'], 'ready')
                with patch.object(manager.obsidian, 'export', side_effect=OSError('disk full')):
                    manager.sync_obsidian()
                self.assertTrue(manager.obsidian_error)
                self.assertEqual(manager.jobs[VIDEO]['status'], 'ready')
                self.assertEqual(len(manager.snapshot.chapters), 1)
            finally:
                manager.close()

    def test_notes_have_original_evidence_timestamps_and_safe_text(self):
        value = bundle()
        value['source']['title'] = '../Bad: <script> [[link]]'
        value['source']['segments'][0]['text'] = '<script> ![image](https://evil.test)'
        value['chapters'][0]['summary'] = 'AI summary'
        docs = documents(value['source'], value['chapters'])
        self.assertEqual(len(docs), 2)
        for name in docs:
            self.assertEqual(Path(name).name, name)
            self.assertNotIn(':', name)
        captions = next(v for k,v in docs.items() if k.endswith('原始字幕.md'))
        self.assertIn('^cue-0', captions)
        self.assertIn('&t=0s', captions)
        self.assertNotIn('<script>', captions.split('---\n', 2)[-1])
        self.assertIn('不是 AI 归纳', captions)
        notes = next(v for k,v in docs.items() if not k.endswith('原始字幕.md'))
        self.assertIn('AI summary', notes)
        self.assertIn('#%5Ecue-0', notes)

    def test_opt_in_idempotent_restart_disable_and_user_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); vault = root/'vault'; vault.mkdir(); (vault/'.obsidian').mkdir()
            exporter = ObsidianExport(root/'state')
            value = bundle()
            exporter.export(value['source'], value['chapters'])
            self.assertFalse((vault/'ClipCompass').exists())
            exporter.configure({'enabled':True,'vault':str(vault)})
            exporter.export(value['source'], value['chapters'])
            self.assertEqual(exporter.status()['synced'], 1)
            files = list((vault/'ClipCompass').glob('*.md')); self.assertEqual(len(files), 2)
            files[0].write_text('my personal notes', encoding='utf-8')
            restored = ObsidianExport(root/'state')
            restored.export(value['source'], value['chapters'])
            self.assertEqual(files[0].read_text(encoding='utf-8'), 'my personal notes')
            restored.configure({'enabled':False})
            self.assertFalse(restored.status()['enabled'])
            self.assertEqual(len(list((vault/'ClipCompass').glob('*.md'))), 2)

    def test_conflict_partial_export_retry_and_invalid_vault(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(AppError): vault_path(str(root))
            with self.assertRaises(AppError): vault_path('relative')
            (root/'.obsidian').mkdir()
            folder = root/'ClipCompass'; folder.mkdir()
            value = bundle(); names = documents(value['source'], value['chapters'])
            conflict = folder/next(iter(names)); conflict.write_text('mine',encoding='utf-8')
            exporter = ObsidianExport(root/'state')
            exporter.configure({'enabled':True,'vault':str(root)})
            exporter.export(value['source'], value['chapters'])
            self.assertEqual(len(exporter.status()['errors']), 1)
            self.assertEqual(conflict.read_text(encoding='utf-8'), 'mine')
            conflict.rename(folder/'user-note.md')
            exporter.export(value['source'], value['chapters'])
            self.assertEqual(exporter.status()['synced'], 1)
            self.assertEqual(exporter.status()['errors'], [])


if __name__ == '__main__': unittest.main()
