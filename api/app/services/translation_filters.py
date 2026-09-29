"""Lightweight filters for OCR and translated text in the test pipeline."""
import re


_FAILURE = re.compile(r'(?:ERROR|REQUEST|TIMEOUT|HTTP|NO API|RATE LIMIT|CONNECTION FAILED)', re.I)
_DATE_OR_TIME = re.compile(r'\b\d{1,4}[/-]\d{1,2}(?:[/-]\w+)?|\b\d{1,2}:\d{2}', re.I)
_JAPANESE = re.compile(r'[\u3040-\u30ff\u3400-\u9fff]')


def is_failure_translation(text: str) -> bool:
    return bool(_FAILURE.search(text))


def is_graphic_text(text: str) -> bool:
    return bool(_DATE_OR_TIME.search(text))


def usable_translation(text: str) -> bool:
    text = text.strip()
    return bool(text and not is_failure_translation(text) and not is_graphic_text(text)
                and not _JAPANESE.search(text))
