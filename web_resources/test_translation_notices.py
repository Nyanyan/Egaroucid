from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from check_translation_notices import FULL_TRANSLATIONS, NOTICE, check_page
from check_translation_notices import check_translation_notices


class TranslationNoticeTests(unittest.TestCase):
    def test_verified_translations_accept_markdown_and_html_links(self):
        markdown = NOTICE.replace('the Japanese version.',
                                  '[the Japanese version](../../ja/usage/).', 1)
        html = NOTICE.replace('the Japanese version.',
                              '<a href="../../ja/usage/">the Japanese version</a>.', 1)
        for page in FULL_TRANSLATIONS:
            with self.subTest(page=page):
                self.assertEqual(check_page('**' + markdown + '**', page + '.md'), [])
                self.assertEqual(check_page('<p><b>' + html + '</b></p>', page + '.html'), [])

    def test_partial_translations_and_unknown_pages_reject_notices(self):
        for page in ['index', 'download/index', 'console/index', 'new-page/index']:
            for notice in [NOTICE, NOTICE.replace('translated', 'translated by Gemini CLI')]:
                with self.subTest(page=page, notice=notice):
                    self.assertTrue(check_page(notice, page + '.md'))
                    self.assertTrue(check_page('<p>' + notice + '</p>', page + '.html'))

    def test_verified_translations_reject_tool_names(self):
        for tool in ['Gemini CLI', 'Google Translate', 'another tool']:
            with self.subTest(tool=tool):
                self.assertTrue(check_page(NOTICE.replace('translated', 'translated by ' + tool),
                                           'usage/index.md'))

    def test_verified_translations_require_one_notice(self):
        for text in ['', NOTICE + '\n' + NOTICE]:
            with self.subTest(text=text):
                self.assertTrue(check_page(text, 'usage/index.md'))

    def test_legacy_notices_are_rejected(self):
        notice = ('This page is a machine translation of the Japanese version. '
                  'Please refer to the Japanese version for the official content.')
        self.assertTrue(check_page(notice, 'console/index.md'))
        self.assertTrue(check_page(notice, 'usage/index.md'))

    def test_reference_to_another_translated_article_is_allowed(self):
        text = ('I wrote [Technology Explanation](./explanation/). '
                'This article is machine-translated and the original version is in Japanese.')
        self.assertEqual(check_page(text, 'technology/index.md'), [])
        self.assertEqual(check_page('# Egaroucid\n\nExisting English text.', 'index.md'), [])

    def make_web_root(self, directory):
        web_root = Path(directory) / 'web_resources'
        for page in FULL_TRANSLATIONS:
            path = web_root / 'en' / (page + '.md')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Translation\n\n**' + NOTICE + '**\n', encoding='utf-8')
        return web_root

    def test_html_trees_are_checked(self):
        for tree in ['generated', 'docs']:
            with self.subTest(tree=tree), tempfile.TemporaryDirectory() as directory:
                web_root = self.make_web_root(directory)
                html_root = web_root / 'generated' if tree == 'generated' else web_root.parent / 'docs'
                path = html_root / 'en' / 'download' / 'index.html'
                path.parent.mkdir(parents=True)
                path.write_text('<p>' + NOTICE + '</p>', encoding='utf-8')
                check_translation_notices(web_root)
                with self.assertRaisesRegex(ValueError, 'verified full-page'):
                    check_translation_notices(web_root, include_generated=True, include_docs=True)

    def test_generator_stops_before_network_or_deleting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            web_root = self.make_web_root(directory)
            invalid = web_root / 'en' / 'index.md'
            invalid.write_text(NOTICE, encoding='utf-8')
            output = web_root / 'generated' / 'en' / 'sentinel.txt'
            output.parent.mkdir(parents=True)
            output.write_text('keep', encoding='utf-8')
            result = self.run_entrypoint(web_root, 'generate_html.py', 'en')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Translation notice check failed', result.stderr)
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep')

    def test_copy_stops_before_deleting_published_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            web_root = self.make_web_root(directory)
            invalid = web_root / 'generated' / 'en' / 'console' / 'index.html'
            invalid.parent.mkdir(parents=True)
            invalid.write_text('<p>' + NOTICE + '</p>', encoding='utf-8')
            for language in ['ja', 'en']:
                published = web_root.parent / 'docs' / language / 'sentinel.txt'
                published.parent.mkdir(parents=True)
                published.write_text('keep', encoding='utf-8')
            result = self.run_entrypoint(web_root, 'copy_to_docs.py')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Translation notice check failed', result.stderr)
            for language in ['ja', 'en']:
                self.assertEqual((web_root.parent / 'docs' / language / 'sentinel.txt')
                                 .read_text(encoding='utf-8'), 'keep')

    def run_entrypoint(self, web_root, script, *args):
        source = Path(__file__).absolute().parent
        for name in ['check_translation_notices.py', script]:
            shutil.copyfile(source / name, web_root / name)
        return subprocess.run([sys.executable, script, *args], cwd=web_root,
                              capture_output=True, text=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
