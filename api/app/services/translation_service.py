import asyncio
import re
import json
import logging
import time
import unicodedata
from urllib.parse import urlsplit
from typing import Dict, List, Optional, Tuple, Any
import httpx

from app.core.config import settings
from app.schemas import DetectedBubble

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    pass


def validate_translations(
    dialogue_items: List[Dict[str, Any]], translations: Dict[int, str], engine: str
) -> Dict[int, str]:
    cleaned = {}
    missing = []
    untranslated = []
    for item in dialogue_items:
        bubble_id = item["id"]
        if not (item.get("text") or "").strip():
            continue
        text = translations.get(bubble_id)
        if not isinstance(text, str) or not text.strip():
            missing.append(bubble_id)
            continue
        normalized = unicodedata.normalize("NFKC", text)
        if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', normalized):
            untranslated.append(bubble_id)
            continue
        cleaned[bubble_id] = text.strip()
    errors = []
    if missing:
        errors.append(f"missing or empty translations for bubble IDs {missing}")
    if untranslated:
        errors.append(f"untranslated Japanese text for bubble IDs {untranslated}")
    if errors:
        raise TranslationError(f"{engine} returned " + "; ".join(errors) + ".")
    return cleaned


def extract_json_from_text(text: str) -> Optional[Any]:
    # Some reasoning models include a separate thought block before the answer.
    # Never interpret JSON examples from that block as actual translations.
    text = re.sub(r"^\s*<think>[\s\S]*?</think>", "", text, count=1, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = re.search(r"[\[{]", text)
    if start:
        try:
            parsed, end = json.JSONDecoder().raw_decode(text[start.start():])
            # Do not accept only the first of several JSON answers.
            if not re.search(r"[\[{]", text[start.start() + end:]):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


class MangaTranslationService:
    _shared_instance: Optional["MangaTranslationService"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        self.api_key = (api_key if api_key is not None else settings.LLM_API_KEY).strip()
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        if not self.api_key or self.api_key.lower() in ("your_api_key_here", "none", ""):
            return urlsplit(self.base_url).hostname in (
                "localhost", "127.0.0.1", "::1"
            )
        return True

    def _build_system_prompt(self, target_lang: str) -> str:
        lang_name = "Indonesian" if target_lang.lower() == "id" else "English"
        examples = (
            "(e.g., 'いや、' -> 'Tidak,' / 'Bukan,', 'いい' -> 'Baik' / 'Bagus', '......' -> '...')"
            if target_lang.lower() == "id"
            else "(e.g., 'いや、' -> 'No,' / 'Not really,', 'いい' -> 'Good' / 'Fine', '......' -> '...')"
        )
        return (
            f"You are a professional manga and comic localization translator.\n"
            f"Your task is to translate Japanese manga dialogue bubbles into natural, contextually cohesive {lang_name} ({target_lang}).\n\n"
            f"CRITICAL TRANSLATION RULES:\n"
            f"1. Target Language: Every dialogue in the translation field MUST be translated into {lang_name}. "
            f"NEVER repeat or echo original Japanese characters (Kanji, Hiragana, Katakana) in the translation field.\n"
            f"2. Short Expressions: Even short interjections, reactions, and words MUST be translated into {lang_name} "
            f"{examples}.\n"
            f"3. Contextual Cohesion: Speech bubbles are provided in authentic manga reading order (#1, #2, #3...). "
            f"Maintain speaker consistency, dialogue continuation, and conversational tone across bubbles.\n"
            f"4. Tone & Slang: Adapt manga colloquialisms, character personality, emotional shouts, and exclamation marks accurately.\n"
            f"5. Honorifics: Handle Japanese honorifics naturally for manga localization.\n"
            f"6. Output Format: You MUST output ONLY a valid JSON object matching the following structure:\n"
            f'{{\n  "translations": [\n    {{"id": 1, "translation": "..."}},\n    {{"id": 2, "translation": "..."}}\n  ]\n}}\n'
            f"Do not include any conversational preface, explanation, or markdown fences outside the JSON."
        )

    def _build_user_prompt(self, bubbles_data: List[Dict[str, Any]], target_lang: str) -> str:
        lang_name = "Indonesian" if target_lang.lower() == "id" else "English"
        lines = [
            f"Translate these Japanese manga bubbles into natural {lang_name}. "
            f"Do not copy or output Japanese characters:"
        ]
        for item in bubbles_data:
            b_id = item["id"]
            text = item.get("text", "").strip()
            lines.append(f"[Bubble #{b_id}]: {text}")
        return "\n".join(lines)


    async def _call_google_async(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        results: Dict[int, str] = {}
        target_code = "id" if target_lang.lower() in ("id", "indonesian") else "en"

        pending: List[Tuple[int, str]] = []
        for item in dialogue_items:
            b_id = item["id"]
            original = item.get("text", "").strip()
            if not original:
                results[b_id] = ""
                continue

            compact = re.sub(r"\s+", "", original)
            if re.fullmatch(r"[.．…・·｡。⋯･]+", compact):
                results[b_id] = "..."
            elif re.fullmatch(r"[!！]+", compact):
                results[b_id] = "!"
            elif re.fullmatch(r"[?？]+", compact):
                results[b_id] = "?"
            else:
                pending.append((b_id, original))

        if not pending:
            return results

        batches: List[List[Tuple[int, str]]] = []
        current_batch: List[Tuple[int, str]] = []
        current_chars = 0
        for entry in pending:
            encoded_size = len(entry[1].encode("utf-8")) * 3
            if current_batch and (len(current_batch) >= 50 or current_chars + encoded_size > 6000):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(entry)
            current_chars += encoded_size
        if current_batch:
            batches.append(current_batch)

        url = "https://clients5.google.com/translate_a/t"
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 Chrome/131.0.0.0 Safari/537.36",
        }
        timeout = httpx.Timeout(30.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            for batch in batches:
                params: List[Tuple[str, str]] = [
                    ("client", "dict-chrome-ex"),
                    ("sl", "ja"),
                    ("tl", target_code),
                ]
                params.extend(("q", original) for _, original in batch)

                response: Optional[httpx.Response] = None
                for attempt in range(3):
                    try:
                        response = await client.get(url, params=params)
                    except httpx.TransportError as exc:
                        if attempt == 2:
                            raise TranslationError(
                                f"Google Translate could not be reached: {type(exc).__name__}"
                            ) from exc
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue

                    if response.status_code == 200:
                        break
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            retry_after = response.headers.get("Retry-After", "")
                            delay = float(retry_after) if retry_after.isdigit() else 0.5 * (2 ** attempt)
                            await asyncio.sleep(min(delay, 5.0))
                            continue
                    break

                if response is None or response.status_code != 200:
                    status = response.status_code if response is not None else "no response"
                    logger.error("Google Translate request failed with HTTP %s", status)
                    raise TranslationError(f"Google Translate failed (HTTP {status})")

                try:
                    translated_values = response.json()
                except ValueError as exc:
                    raise TranslationError("Google Translate returned an invalid response") from exc

                if not isinstance(translated_values, list) or len(translated_values) != len(batch):
                    raise TranslationError(
                        "Google Translate returned an unexpected number of results"
                    )

                for (b_id, _), translated_value in zip(batch, translated_values):
                    if not isinstance(translated_value, str) or not translated_value.strip():
                        continue
                    translated_text = unicodedata.normalize("NFKC", translated_value).strip()
                    translated_text = re.sub(r"(?:\s*\.){2,}", "...", translated_text)
                    results[b_id] = translated_text

        return results

    async def translate_bubbles_async(
        self,
        bubbles: List[DetectedBubble],
        target_lang: str = "id",
        translator: str = "llm"
    ) -> List[DetectedBubble]:
        if not bubbles:
            return bubbles

        dialogue_items = [
            {"id": b.id, "text": b.text or ""}
            for b in bubbles
            if (b.text and b.text.strip())
        ]

        if not dialogue_items:
            return bubbles

        start_time = time.perf_counter()
        engine_label = f"llm ({self.model})" if translator != "google" and self.model else translator
        logger.info("[Translate] Engine: %s | Lang: %s | Bubbles: %d", engine_label, target_lang, len(dialogue_items))

        if translator == "google":
            translations_map = await self._call_google_async(dialogue_items, target_lang=target_lang)
        else:
            translations_map = await self._call_llm_async(dialogue_items, target_lang=target_lang)

        validated = validate_translations(dialogue_items, translations_map, "Google Translate" if translator == "google" else "LLM")
        elapsed = time.perf_counter() - start_time
        logger.info("[Translate] Done: %s -> %s (%d bubbles in %.2fs)", translator, target_lang, len(validated), elapsed)

        for b in bubbles:
            if b.id in validated:
                b.translation = validated[b.id]

        return bubbles

    async def _call_llm_async(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        if not dialogue_items:
            return {}
        if not self.model.strip():
            raise TranslationError("LLM model ID is empty. Set LLM_MODEL in api/.env.")
        if not self.is_configured():
            raise TranslationError("The LLM endpoint requires an API key. Set LLM_API_KEY in api/.env.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(target_lang)},
                {"role": "user", "content": self._build_user_prompt(dialogue_items, target_lang)}
            ],
            "temperature": 0.3
        }

        try:
            # Both attempts share the original deadline. Format recovery must
            # not silently double the maximum wait for a page.
            async with asyncio.timeout(self.timeout_seconds):
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    for attempt in range(2):
                        response = await client.post(url, headers=headers, json=payload)
                        if response.status_code != 200:
                            raise TranslationError(f"LLM translation failed (HTTP {response.status_code}). Check the model, endpoint, API key, or quota.")
                        try:
                            data = response.json()
                            choice = data["choices"][0]
                            message = choice["message"]
                            raw_content = message.get("content")
                        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                            raise TranslationError("LLM endpoint returned an invalid chat-completion envelope.") from exc
                        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                            raise TranslationError("LLM refused the translation request or filtered its content.")
                        try:
                            if choice.get("finish_reason") == "length":
                                raise TranslationError("LLM translation output was truncated (finish_reason=length).")
                            return self._parse_llm_response(raw_content, dialogue_items)
                        except TranslationError as exc:
                            if attempt:
                                raise TranslationError(f"LLM response remained invalid after one format retry: {exc}") from exc
                            logger.warning("[Translate] LLM response rejected; retrying once: %s", exc)
                            payload["messages"].append({
                                "role": "user",
                                "content": (
                                    f"The previous response failed validation: {exc} "
                                    "Return the complete translation again as ONLY a JSON object "
                                    'with a "translations" array of {"id": integer, "translation": string}. '
                                    "Include each requested ID exactly once. No reasoning or markdown."
                                ),
                            })

        except TranslationError:
            raise
        except httpx.ConnectError as e:
            raise TranslationError("Cannot connect to the configured LLM base URL. Verify the address and start the LLM service before retrying.") from e
        except (httpx.TimeoutException, TimeoutError) as e:
            raise TranslationError(f"LLM request timed out after {self.timeout_seconds}s.") from e
        except Exception as e:
            raise TranslationError(f"Invalid LLM translation response: {type(e).__name__}.") from e

    def _parse_llm_response(
        self,
        raw_content: str,
        dialogue_items: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        if not isinstance(raw_content, str):
            raise TranslationError("LLM message content must be text containing a JSON object.")
        parsed = extract_json_from_text(raw_content)
        if not isinstance(parsed, (dict, list)):
            raise TranslationError("LLM returned invalid translation JSON; expected an object or an array of ID-tagged translations.")
        result: Dict[int, str] = {}
        expected_ids = {item["id"] for item in dialogue_items}

        if isinstance(parsed, list):
            entries = parsed
        elif "translations" in parsed:
            entries = parsed["translations"]
            if not isinstance(entries, list):
                raise TranslationError("LLM JSON field 'translations' must be a list.")
        else:
            entries = [{"id": key, "translation": value} for key, value in parsed.items()]

        for item in entries:
            if not isinstance(item, dict) or "id" not in item or "translation" not in item:
                raise TranslationError("Each LLM translation must contain an ID and translation text.")
            raw_id = item["id"]
            if type(raw_id) is int:
                bubble_id = raw_id
            elif isinstance(raw_id, str) and re.fullmatch(r"-?[0-9]+", raw_id):
                bubble_id = int(raw_id)
            else:
                raise TranslationError("LLM translation IDs must be integers.")
            if bubble_id not in expected_ids:
                raise TranslationError(f"LLM returned an unknown bubble ID: {bubble_id}.")
            if bubble_id in result:
                raise TranslationError(f"LLM returned duplicate translations for bubble ID {bubble_id}.")
            if not isinstance(item["translation"], str):
                raise TranslationError(f"LLM translation for bubble ID {bubble_id} must be text.")
            result[bubble_id] = item["translation"].strip()

        return validate_translations(dialogue_items, result, "LLM")


def get_translation_service(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> MangaTranslationService:
    if (
        MangaTranslationService._shared_instance is not None
        and api_key is None
        and base_url is None
        and model is None
    ):
        return MangaTranslationService._shared_instance

    instance = MangaTranslationService(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    if api_key is None and base_url is None and model is None:
        MangaTranslationService._shared_instance = instance
    return instance
