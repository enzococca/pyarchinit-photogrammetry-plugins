"""The tests read the Italian texts of the plugin; the plugin's folder on the path, the preferences
of a window of the tests in a temporary folder, never the user's."""
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

_TMP = Path(tempfile.mkdtemp(prefix="ppg-shipwreck-tests-"))
os.environ.setdefault("PPG_PLUGINS_DIR", str(_TMP / "plugins"))
os.environ.setdefault("PPG_PLUGINS_ENVIRONMENT", "0")

from PySide6.QtCore import QSettings  # noqa: E402

QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(_TMP / "settings"))


def pytest_sessionfinish(session, exitstatus):
    import shutil

    shutil.rmtree(_TMP, ignore_errors=True)
