"""The tests read the Italian texts of the plugin, whatever the language of the system; the plugin's folder on the
path, so that the tests run also without the plugin installed; the preferences of a window of the tests in a temporary
folder, never the user's; the program finds no plugin but those the tests give it; Qt's garbage collected between the
tests in the main thread."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
for folder in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

_TMP = Path(tempfile.mkdtemp(prefix="ppg-conservation-tests-"))
os.environ.setdefault("PPG_PLUGINS_DIR", str(_TMP / "plugins"))
os.environ.setdefault("PPG_PLUGINS_ENVIRONMENT", "0")
try:
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(_TMP / "settings"))
except ImportError:  # the engine's tests without the program's interface
    pass


@pytest.fixture(autouse=True)
def _qt_garbage_between_tests():
    """The windows and widgets a test leaves behind are destroyed here, between the tests, in the main thread (as the
    program's own tests do), and not by a garbage collection that a worker thread of a later test starts: Qt cannot
    stop a window's timers from another thread, and the next ``processEvents`` would hand a timer event to the
    deleted object (the likely cause of the rare failure seen once in 8 full runs). The objects waiting for
    ``deleteLater`` go too."""
    yield
    import gc

    if "PySide6.QtCore" not in sys.modules:
        return
    from PySide6.QtCore import QCoreApplication, QEvent

    gc.collect()
    if QCoreApplication.instance() is not None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    import shutil

    shutil.rmtree(_TMP, ignore_errors=True)
    try:
        import conservation_samples

        conservation_samples.cleanup()
    except ImportError:
        pass
