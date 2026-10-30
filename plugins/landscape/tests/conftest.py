"""The tests read the Italian texts of the plugin, whatever the language of the system; the plugin's
folder on the path, so that the tests run also without the plugin installed (the engine's tests run
in a Python with Landlab and without the program)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
for folder in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
