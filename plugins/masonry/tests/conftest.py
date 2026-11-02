"""The tests read the Italian texts of the plugin, whatever the language of the system; the plugin's folder on the
path, so that the tests run also without the plugin installed; the preferences of a window of the tests in a temporary
folder, never the user's; the program finds no plugin but those the tests give it."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
for folder in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

_TMP = Path(tempfile.mkdtemp(prefix="ppg-masonry-tests-"))
os.environ.setdefault("PPG_PLUGINS_DIR", str(_TMP / "plugins"))
os.environ.setdefault("PPG_PLUGINS_ENVIRONMENT", "0")
try:
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(_TMP / "settings"))
except ImportError:  # the engine's tests without the program's interface
    pass


def pytest_sessionfinish(session, exitstatus):
    import shutil

    shutil.rmtree(_TMP, ignore_errors=True)
