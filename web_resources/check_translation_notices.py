"""Check page-level translation notices before generating or copying web pages."""

import argparse
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re


# Commits that added these pages as complete English translations.
FULL_TRANSLATIONS = {
    'usage/index': '77669cf42',
    'usage/book/index': 'b04f9c568',
    'technology/explanation/index': 'e97ab2c71',
}
NOTICE = (
    'This page was machine translated from the Japanese version. '
    'Please refer to the Japanese version for the official content.'
)
NOTICE_SENTENCE = re.compile(
    r'\bThis\s+page\s+(?:was|is|has\s+been)\s+(?:a\s+)?'
    r'(?:machine[\s-]+translat(?:ed|ion)|automatically\s+translated)\b[^.!?]*\.',
    re.IGNORECASE,
)


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'br', 'td', 'th', 'pre'}:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        self.handle_starttag(tag, [])


def check_page(text, relative_path):
    """Return errors for one Markdown or HTML page, ignoring link markup."""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text).replace('**', '')
    parser = _PageText()
    parser.feed(text)
    plain = ' '.join(''.join(parser.parts).split())
    notices = NOTICE_SENTENCE.findall(plain)
    page = PurePosixPath(relative_path).with_suffix('').as_posix()
    if page not in FULL_TRANSLATIONS:
        if notices:
            return ['page-level notice requires verified full-page translation history']
        return []
    if len(notices) != 1 or plain.count(NOTICE) != 1:
        return ['use exactly one standard translation notice without a tool name']
    return []


def check_translation_notices(web_root, *, include_generated=False, include_docs=False):
    """Raise before any writes if source or selected HTML trees violate policy."""
    web_root = Path(web_root)
    source = web_root / 'en'
    errors = []
    for page in FULL_TRANSLATIONS:
        if not (source / (page + '.md')).is_file():
            errors.append(f'{source / (page + ".md")}: verified translation is missing')
    trees = [source]
    if include_generated:
        trees.append(web_root / 'generated' / 'en')
    if include_docs:
        trees.append(web_root.parent / 'docs' / 'en')
    for tree in trees:
        if not tree.is_dir():
            continue
        for path in sorted(tree.rglob('*')):
            if not path.is_file() or path.suffix not in {'.md', '.html', '.htm'}:
                continue
            for error in check_page(path.read_text(encoding='utf-8-sig'),
                                    path.relative_to(tree).as_posix()):
                errors.append(f'{path}: {error}')
    if errors:
        raise ValueError('Translation notice check failed:\n' + '\n'.join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only', action='store_true',
                        help='check English source pages only')
    args = parser.parse_args()
    try:
        check_translation_notices(
            Path(__file__).absolute().parent,
            include_generated=not args.sources_only,
            include_docs=not args.sources_only,
        )
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
    print('Translation notice check passed.')


if __name__ == '__main__':
    main()
