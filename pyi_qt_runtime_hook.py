"""Avoid inheriting another application's Qt plugins or DLL search paths."""

import os
import sys
from pathlib import Path


_DLL_HANDLES = []
_BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

for _key in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
    os.environ.pop(_key, None)

if os.name == "nt":
    for _directory in (_BUNDLE, _BUNDLE / "PySide6", _BUNDLE / "shiboken6"):
        if _directory.is_dir():
            os.environ["PATH"] = str(_directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(str(_directory)))
