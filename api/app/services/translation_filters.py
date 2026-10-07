import re
import unicodedata


_DATE_OR_TIME = re.compile(r'\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?|\d{1,2}:\d{2}')
_JAPANESE = re.compile(r'[\u3040-\u30ff\u3400-\u9fff]')


def is_graphic_text(text: str) -> bool:
    return bool(_DATE_OR_TIME.fullmatch(unicodedata.normalize('NFKC', text).strip()))


def usable_translation(text: str) -> bool:
    text = text.strip()
    return bool(text and not _JAPANESE.search(text))
