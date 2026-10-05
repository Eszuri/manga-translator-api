"""Freeze Transformers lazy-import tables without bundling its model catalog."""

from nuitka.plugins.standard import TransformersPlugin


class MangaOcrPlugin(TransformersPlugin.NuitkaPluginTransformers):
    plugin_name = "manga-ocr-transformers"
    plugin_desc = "Compile OCR import tables without requiring Python source at runtime."

    def getImplicitImports(self, module):
        # OCR imports its processor/tokenizer explicitly. Keep the standard
        # plugin's source-scan replacement, but not its full-catalog inclusion.
        return ()
