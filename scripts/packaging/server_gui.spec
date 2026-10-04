# Windows onedir package: runtime dependencies are bundled; models stay external.
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
api = root / "api"
# Resolve native dependencies only from this Python environment and Windows.
# Developer tools on PATH (e.g. Poppler) may provide incompatible same-name DLLs.
windows = Path(os.environ["SystemRoot"])
search_roots = [Path(sys.prefix), Path(sys.base_prefix), windows]
os.environ["PATH"] = os.pathsep.join(str(path) for path in (
    Path(sys.executable).parent, Path(sys.base_prefix), windows / "System32", windows,
))
datas = [(str(api / "app/assets/fonts"), "app/assets/fonts"),
         (str(root / "extension/icons/icon128.png"), "server_gui/assets")]
# Transformers' lazy import structure scans source files at runtime.
datas += collect_data_files("transformers", include_py_files=True)
datas += collect_data_files("unidic_lite")
for distribution in ("transformers", "tokenizers", "fugashi", "unidic-lite", "sentencepiece",
                     "onnxruntime-directml", "PySide6-Essentials", "shiboken6"):
    datas += copy_metadata(distribution)

a = Analysis(
    [str(api / "tools/run_gui.py")], pathex=[str(api)],
    binaries=collect_dynamic_libs("onnxruntime"), datas=datas,
    hiddenimports=["transformers.models.vit.image_processing_pil_vit",
                   "transformers.models.bert_japanese.tokenization_bert_japanese",
                   "uvicorn.logging", "uvicorn.loops.asyncio", "uvicorn.protocols.http.h11_impl",
                   "uvicorn.lifespan.on"],
    hookspath=[], runtime_hooks=[],
    excludes=["torch", "tensorflow", "jax", "keras", "matplotlib", "IPython",
              "pytest", "scipy", "sklearn", "pandas", "PySide6.QtQml", "PySide6.QtQuick",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"],
    noarchive=False,
)
# Windows Qt uses the OS ICU API, not the versioned ICU build shipped by Poppler.
# Bundling another icuuc.dll shadows Windows and makes QtCore fail with WinError 127.
system_icu = {"icuuc.dll", "icuin.dll", "icudt.dll"}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in system_icu]
foreign_dlls = [entry[1] for entry in a.binaries
                if Path(entry[1]).suffix.lower() in (".dll", ".pyd")
                and not any(Path(entry[1]).resolve().is_relative_to(path.resolve()) for path in search_roots)]
if foreign_dlls:
    raise RuntimeError(f"Unexpected native dependency outside Python/Windows: {foreign_dlls}")
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MangaTranslatorServer",
          debug=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MangaTranslatorServer")
