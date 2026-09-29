import os
import re
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import httpx

from app.core.config import settings
from app.schemas import DetectedBubble, DialogueItem, TranslatedDialogueItem

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Safely extracts and parses a JSON object from LLM response text,
    stripping markdown code fences (```json ... ```) or surrounding text.
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find outer-most { ... } braces
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
            return "localhost" in self.base_url or "127.0.0.1" in self.base_url
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


    def _generate_fallback_translations(
        self,
        bubbles_data: List[Dict[str, Any]],
        reason: str
    ) -> Dict[int, str]:
        """Provides informative fallback translations when API call is not available."""
        result = {}
        for item in bubbles_data:
            b_id = item["id"]
            orig = item.get("text", "")
            if not orig:
                result[b_id] = ""
            else:
                result[b_id] = f"[{reason}]"
        return result

    def translate_bubbles(
        self,
        bubbles: List[DetectedBubble],
        target_lang: str = "id"
    ) -> List[DetectedBubble]:
        """
        Synchronous batch translation of DetectedBubble objects in context.
        Populates bubble.translation for each bubble.
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

        translations_map = self._call_llm_sync(dialogue_items, target_lang=target_lang)

        for b in bubbles:
            if b.id in translations_map:
                b.translation = translations_map[b.id]

        return bubbles

    async def _call_google_async(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        """Translates dialogue items using Google Translate (fast & free)."""
        import unicodedata

        results: Dict[int, str] = {}
        target_code = "id" if target_lang.lower() in ("id", "indonesian") else "en"

        async with httpx.AsyncClient(timeout=15.0) as client:
            for item in dialogue_items:
                b_id = item["id"]
                original = item.get("text", "").strip()
                if not original:
                    results[b_id] = ""
                    continue

                compact = re.sub(r"\s+", "", original)
                if re.fullmatch(r"[.．…・·｡。⋯･]+", compact):
                    results[b_id] = "..."
                    continue
                elif re.fullmatch(r"[!！]+", compact):
                    results[b_id] = "!"
                    continue
                elif re.fullmatch(r"[?？]+", compact):
                    results[b_id] = "?"
                    continue

                try:
                    url = "https://translate.googleapis.com/translate_a/single"
                    params = {
                        "client": "gtx",
                        "sl": "ja",
                        "tl": target_code,
                        "dt": "t",
                        "q": original
                    }
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        translated_text = "".join([segment[0] for segment in data[0] if segment and segment[0]])
                        translated_text = unicodedata.normalize("NFKC", translated_text).strip()
                        translated_text = re.sub(r"(?:\s*\.){2,}", "...", translated_text)
                        results[b_id] = translated_text
                    else:
                        results[b_id] = original
                except Exception as e:
                    logger.warning(f"Google translate error for bubble {b_id}: {e}")
                    results[b_id] = original

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

    def _call_llm_sync(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        """Synchronous HTTP call to the OpenAI-compatible endpoint."""
        if not self.is_configured():
            logger.warning("LLM API Key is not configured. Returning placeholder message.")
            return self._generate_fallback_translations(
                dialogue_items,
                "No API Key: Set LLM_API_KEY in .env"
            )

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
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code == 401:
                logger.error("LLM API returned 401 Unauthorized: Invalid API key.")
                return self._generate_fallback_translations(dialogue_items, "Error: Invalid API Key")
            elif response.status_code == 429:
                logger.warning("LLM API returned 429 Rate Limit / Quota Exceeded.")
                return self._generate_fallback_translations(dialogue_items, "Error: Rate Limit Exceeded")
            elif response.status_code != 200:
                logger.error(f"LLM API returned unexpected status {response.status_code}: {response.text[:200]}")
                return self._generate_fallback_translations(dialogue_items, f"Error: HTTP {response.status_code}")

            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return self._parse_llm_response(raw_content, dialogue_items)

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to LLM endpoint at {self.base_url}: {e}")
            return self._generate_fallback_translations(dialogue_items, "Error: Connection Failed")
        except httpx.TimeoutException as e:
            logger.error(f"LLM request timed out ({self.timeout_seconds}s): {e}")
            return self._generate_fallback_translations(dialogue_items, "Error: Request Timeout")
        except Exception as e:
            logger.error(f"Unexpected error during translation: {e}")
            return self._generate_fallback_translations(dialogue_items, f"Error: {type(e).__name__}")

    async def _call_llm_async(
        self,
        dialogue_items: List[Dict[str, Any]],
        target_lang: str = "id"
    ) -> Dict[int, str]:
        """Asynchronous HTTP call to the OpenAI-compatible endpoint."""
        if not self.is_configured():
            logger.warning("LLM API Key is not configured. Returning placeholder message.")
            return self._generate_fallback_translations(
                dialogue_items,
                "No API Key: Set LLM_API_KEY in .env"
            )

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

            if response.status_code == 401:
                logger.error("LLM API returned 401 Unauthorized: Invalid API key.")
                return self._generate_fallback_translations(dialogue_items, "Error: Invalid API Key")
            elif response.status_code == 429:
                logger.warning("LLM API returned 429 Rate Limit / Quota Exceeded.")
                return self._generate_fallback_translations(dialogue_items, "Error: Rate Limit Exceeded")
            elif response.status_code != 200:
                logger.error(f"LLM API returned status {response.status_code}: {response.text[:200]}")
                return self._generate_fallback_translations(dialogue_items, f"Error: HTTP {response.status_code}")

            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return self._parse_llm_response(raw_content, dialogue_items)

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to LLM endpoint at {self.base_url}: {e}")
            return self._generate_fallback_translations(dialogue_items, "Error: Connection Failed")
        except httpx.TimeoutException as e:
            logger.error(f"LLM request timed out ({self.timeout_seconds}s): {e}")
            return self._generate_fallback_translations(dialogue_items, "Error: Request Timeout")
        except Exception as e:
            logger.error(f"Unexpected error during translation: {e}")
            return self._generate_fallback_translations(dialogue_items, f"Error: {type(e).__name__}")

    def _parse_llm_response(
        self,
        raw_content: str,
        dialogue_items: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Parses LLM JSON output and falls back cleanly if output format is irregular."""
        parsed = extract_json_from_text(raw_content)
        result: Dict[int, str] = {}

        if parsed and isinstance(parsed, dict):
            # Check for "translations" key or direct mapping
            translations_list = parsed.get("translations")
            if isinstance(translations_list, list):
                for item in translations_list:
                    if isinstance(item, dict) and "id" in item and "translation" in item:
                        result[int(item["id"])] = str(item["translation"]).strip()
            elif isinstance(parsed, dict):
                # Might be {"1": "...", "2": "..."}
                for k, v in parsed.items():
                    if str(k).isdigit() and isinstance(v, str):
                        result[int(k)] = v.strip()

        # If any bubble ID is missing from parsed result, provide fallback
        for item in dialogue_items:
            b_id = item["id"]
            if b_id not in result or not result[b_id]:
                # If parsed failed entirely, provide raw text if single item, else fallback
                if len(dialogue_items) == 1 and not result:
                    result[b_id] = raw_content.strip()
                else:
                    result[b_id] = item.get("text", "")

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
