"""External user data and bundled assets for the portable desktop application."""
import os
import sys
from pathlib import Path


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def default_model_dir() -> Path:
    """Find existing models without copying, downloading, or scanning user disks."""
    directory = application_dir()
    if not getattr(sys, "frozen", False):
        return directory / "models"
    candidates = [directory / "models"]
    # A development build is normally at <project>/dist/MangaTranslatorServer.
    # Look only at this known layout; arbitrary ancestor folders are not searched.
    if directory.parent.name.casefold() == "dist":
        candidates.append(directory.parent.parent / "models")
    for candidate in candidates:
        if _complete_model_dir(candidate):
            return candidate
    return candidates[0]


def _complete_model_dir(directory: Path) -> bool:
    from app.core.paths import missing_model_files

    try:
        return not missing_model_files(directory)
    except OSError:
        return False


def normalize_model_dir(value: str | Path) -> Path:
    """Accept a complete model folder or a known parent/child folder selection.

    An incomplete or custom location is retained so the GUI can report that
    exact selection instead of silently loading models from somewhere else.
    """
    directory = Path(value).expanduser().resolve()
    # Migrate a saved selection from the previous source layout.
    if tuple(part.casefold() for part in directory.parts[-3:]) == ("api", "app", "models"):
        replacement = directory.parents[2] / "models"
        if _complete_model_dir(replacement):
            return replacement
    candidates = [directory]
    if directory.name.casefold() == "manga-ocr":
        candidates.append(directory.parent)
    candidates.append(directory / "models")
    for candidate in candidates:
        if _complete_model_dir(candidate):
            return candidate
    return directory


def model_directory_status(value: str | Path) -> tuple[Path, list[str]]:
    """Return the selected model root and its missing or empty required files."""
    from app.core.paths import missing_model_files

    directory = normalize_model_dir(value)
    return directory, missing_model_files(directory)


def user_data_dir() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "MangaTranslatorServer"


def icon_path() -> Path:
    bundled = Path(__file__).resolve().parent / "assets" / "icon128.png"
    if bundled.is_file():
        return bundled
    directory = application_dir()
    if getattr(sys, "frozen", False) and directory.parent.name.casefold() == "dist":
        directory = directory.parent.parent
    return directory / "extension" / "icons" / "icon128.png"
