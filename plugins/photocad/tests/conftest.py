"""The tests read the Italian texts of the plugin, whatever the language of the system, and never
touch the network nor the real keychain: PhotoCAD is a fake opener (``fakecad``), the keychain a
dictionary, the API key never written anywhere."""
import os

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")
os.environ.pop("PHOTOCAD_API_KEY", None)  # a key of the person running the tests never reaches them

import pyarchinit_photogrammetry  # noqa: E402,F401 - first: it points rasterio at its own PROJ data
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network_no_keychain(monkeypatch):
    """Every test: the opener refuses the network unless the test gives the fake server, the
    keychain is a dictionary, the waits are instant."""
    import fakecad

    from ppg_photocad import client, keys

    def refuse(request, timeout):  # pragma: no cover - only when a test forgot the fake server
        raise AssertionError(f"the network was asked: {request.full_url}")

    monkeypatch.setattr(client, "_urlopen", refuse)
    monkeypatch.setattr(client, "_sleep", lambda s: None)
    store = fakecad.FakeKeyring()
    monkeypatch.setattr(keys, "_backend", lambda: store)
    monkeypatch.delenv("PHOTOCAD_API_KEY", raising=False)
    monkeypatch.delenv("PPG_PHOTOCAD_URL", raising=False)
    yield store
