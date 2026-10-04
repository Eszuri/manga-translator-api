import asyncio
import re
import json
import logging
from urllib.parse import urlsplit
from typing import Dict, List, Optional, Tuple, Any
import httpx

from app.core.config import settings
from app.schemas import DetectedBubble

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """A translation failure that must not be rendered over source dialogue."""


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Safely extracts and parses a JSON object from LLM response text,
    stripping markdown code fences (```json ... ```) or surrounding text.
    """
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

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


class MangaTranslationService:
    """
    Universal OpenAI-compatible contextual translation service for manga dialogues.
    Works with official OpenAI, OpenRouter, Groq, DeepSeek, local Ollama, vLLM, etc.
    """
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
        """Returns True if a real API key is configured (or if using a local provider like Ollama)."""
        if not self.api_key or self.api_key.lower() in ("your_api_key_here", "none", ""):
            # If pointing to localhost/ollama, api_key is optional
            return urlsplit(self.base_url).hostname in (
                "localhost", "127.0.0.1", "::1", "host.docker.internal"
            )
        return True

    def _build_system_prompt(self, target_lang: str) -> str:
        lang_name = "Indonesian" if target_lang.lower() == "id" else "English"
        return (
            f"You are a professional manga and comic localization translator.\n"
            f"Your task is to translate Japanese manga dialogue bubbles into natural, contextually cohesive {lang_name} ({target_lang}).\n\n"
            f"CRITICAL TRANSLATION RULES:\n"
            f"1. Target Language: Every dialogue in the translation field MUST be translated into {lang_name}. "
            f"NEVER repeat or echo original Japanese characters (Kanji, Hiragana, Katakana) in the translation field.\n"
            f"2. Short Expressions: Even short interjections, reactions, and words MUST be translated into {lang_name} "
            f"(e.g., 'いや、' -> 'Tidak,' / 'Bukan,', 'いい' -> 'Baik' / 'Bagus', '......' -> '...').\n"
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
        """Translate dialogue items with Google's Chrome translation endpoint.

        Requests are batched to avoid sending one HTTP request per speech bubble.
        Translation failures are raised so the pipeline cannot silently inpaint the
        source text and render the untranslated Japanese text again.
        """
        import unicodedata

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

        # Keep GET URLs comfortably below common proxy/server URL limits while
        # still translating most manga pages in one request.
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
                    except (httpx.ConnectError, httpx.TimeoutException) as exc:
                        if attempt == 2:
                            raise RuntimeError(
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
                    raise RuntimeError(f"Google Translate failed (HTTP {status})")

                try:
                    translated_values = response.json()
                except ValueError as exc:
                    raise RuntimeError("Google Translate returned an invalid response") from exc

                if not isinstance(translated_values, list) or len(translated_values) != len(batch):
                    raise RuntimeError(
                        "Google Translate returned an unexpected number of results"
                    )

                for (b_id, _), translated_value in zip(batch, translated_values):
                    if not isinstance(translated_value, str) or not translated_value.strip():
                        raise RuntimeError(
                            f"Google Translate did not return text for bubble {b_id}"
                        )
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
        """
        Asynchronous batch translation of DetectedBubble objects in context.
        Populates bubble.translation for each bubble.
        Supports both 'llm' (OpenAI/Ollama) and 'google' (Google Translate).
        """
        if not bubbles:
            return bubbles

        dialogue_items = [
            {"id": b.id, "text": b.text or ""}
            for b in bubbles
            if (b.text and b.text.strip())
        ]

        if not dialogue_items:
            return bubbles

        if translator == "google":
            translations_map = await self._call_google_async(dialogue_items, target_lang=target_lang)
        else:
            translations_map = await self._call_llm_async(dialogue_items, target_lang=target_lang)

        for b in bubbles:
            if b.id in translations_map:
                b.translation = translations_map[b.id]

        return bubbles

    async def _call_llm_async(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        """Asynchronous HTTP call to the OpenAI-compatible endpoint."""
        if not self.model.strip():
            raise TranslationError("LLM model ID is empty. Set the model ID in GUI Settings > Translation or LLM_MODEL for the command-line server.")
        if not self.is_configured():
            raise TranslationError("The LLM endpoint requires an API key. Set it in GUI Settings > Translation or LLM_API_KEY for the command-line server.")

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
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                raise TranslationError(f"LLM translation failed (HTTP {response.status_code}). Check the model, endpoint, API key, or quota.")

            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return self._parse_llm_response(raw_content, dialogue_items)

        except TranslationError:
            raise
        except httpx.ConnectError as e:
            raise TranslationError("Cannot connect to the configured LLM base URL. Verify the address and start the LLM service before retrying.") from e
        except httpx.TimeoutException as e:
            raise TranslationError(f"LLM request timed out after {self.timeout_seconds}s.") from e
        except Exception as e:
            raise TranslationError(f"Invalid LLM translation response: {type(e).__name__}.") from e

    def _parse_llm_response(
        self,
        raw_content: str,
        dialogue_items: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Require complete LLM output before any original text is erased."""
        parsed = extract_json_from_text(raw_content)
        result: Dict[int, str] = {}

        if parsed and isinstance(parsed, dict):
            translations_list = parsed.get("translations")
            if isinstance(translations_list, list):
                for item in translations_list:
                    if isinstance(item, dict) and "id" in item and "translation" in item:
                        if not isinstance(item["translation"], str):
                            raise TranslationError("LLM translation values must be text.")
                        result[int(item["id"])] = item["translation"].strip()
            elif isinstance(parsed, dict):
                for k, v in parsed.items():
                    if str(k).isdigit() and isinstance(v, str):
                        result[int(k)] = v.strip()

        for item in dialogue_items:
            b_id = item["id"]
            if b_id not in result or not result[b_id]:
                raise TranslationError(f"LLM did not return a valid translation for bubble {b_id}.")

        return result


def get_translation_service(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> MangaTranslationService:
    """Returns singleton instance of MangaTranslationService."""
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
