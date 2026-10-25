import os

import pytest

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")


@pytest.fixture(scope="session")
def qt_application():
    """Keep a single QApplication alive until every GUI test has released its widgets."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def qt_app(qt_application):
    """Destroy test windows through Qt before Python tears down their signal callbacks."""
    import gc
    from PySide6.QtCore import QCoreApplication, QEvent

    yield qt_application
    for widget in qt_application.topLevelWidgets():
        if widget.parentWidget() is None:
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_application.processEvents()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
