"""Persist desktop settings without reading or modifying the backend .env."""
import base64
import ctypes
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from server_gui.paths import default_model_dir, normalize_model_dir, user_data_dir

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434/v1"


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt_secret(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Secure API key storage requires Windows DPAPI.")
    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    operation = library.CryptProtectData if protect else library.CryptUnprotectData
    operation.restype = ctypes.c_int
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(ctypes.cast(output.data, ctypes.c_void_p))


@dataclass
class Preferences:
    model_dir: str = field(default_factory=lambda: str(default_model_dir()))
    port: int = 8000
    allow_network: bool = False
    translator: str = "google"
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = ""
    llm_timeout: float = 30.0
    api_key: str = ""

    def validate(self) -> None:
        for name in ("model_dir", "translator", "llm_base_url", "llm_model", "api_key"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("The server port must be between 1 and 65535.")
        if type(self.allow_network) is not bool:
            raise ValueError("Network access must be true or false.")
        if self.translator not in ("google", "llm"):
            raise ValueError("The default translator must be Google or LLM.")
        if type(self.llm_timeout) not in (float, int) or not 1 <= self.llm_timeout <= 3600:
            raise ValueError("The LLM timeout must be between 1 and 3600 seconds.")
        base_url = self.llm_base_url.strip()
        if not base_url and self.translator == "llm":
            raise ValueError("Enter an LLM base URL before selecting LLM as the default translator.")
        if base_url:
            try:
                url = urlsplit(base_url)
                if (url.scheme not in ("http", "https") or not url.hostname
                        or url.query or url.fragment or url.username is not None
                        or url.password is not None or any(char.isspace() for char in base_url)
                        or any(ord(char) < 32 for char in base_url)):
                    raise ValueError
                if url.port is not None and not 1 <= url.port <= 65535:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("Use an HTTP(S) LLM base URL with a valid host and port, without credentials, spaces, query, or fragment.") from exc
            if url.path.rstrip("/").endswith("/chat/completions"):
                raise ValueError("Remove /chat/completions from the LLM base URL. The server adds this path when sending a translation request.")
        if self.translator == "llm" and not self.llm_model.strip():
            raise ValueError("Enter an LLM model ID before selecting LLM as the default translator.")
        if not self.model_dir.strip():
            raise ValueError("Select the folder containing the detector and manga-ocr models.")

    def effective_llm_base_url(self) -> str:
        """Keep the backend's optional LLM configuration valid in Google mode."""
        return self.llm_base_url.strip().rstrip("/") or DEFAULT_LLM_BASE_URL


def load_preferences(path: Path | None = None) -> Preferences:
    path = path or user_data_dir() / "settings.json"
    if not path.is_file():
        return Preferences()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The settings file must contain a JSON object.")
    protected = data.pop("api_key_protected", "")
    if not isinstance(protected, str):
        raise ValueError("The protected API key in the settings file must be a string.")
    # Never accept a plaintext API key from the persistent settings file.
    data.pop("api_key", None)
    unknown = set(data) - Preferences.__dataclass_fields__.keys()
    if unknown:
        raise ValueError("Unrecognized settings fields: " + ", ".join(sorted(unknown)) + ".")
    prefs = Preferences(**data)
    if protected:
        try:
            prefs.api_key = _crypt_secret(base64.b64decode(protected, validate=True), False).decode("utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError("The saved API key cannot be decrypted by this Windows account. Re-enter the key in Settings.") from exc
    prefs.validate()
    return prefs


def save_preferences(prefs: Preferences, path: Path | None = None) -> Path:
    prefs.validate()
    path = path or user_data_dir() / "settings.json"
    data = asdict(prefs)
    # The worker has a different working directory; never persist an ambiguous
    # relative model path even when the caller did not normalize its UI input.
    data["model_dir"] = str(normalize_model_dir(prefs.model_dir))
    secret = data.pop("api_key")
    data["api_key_protected"] = base64.b64encode(_crypt_secret(secret.encode(), True)).decode() if secret else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
