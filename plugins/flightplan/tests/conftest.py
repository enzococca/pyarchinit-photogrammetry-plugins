"""The tests read the Italian texts of the plugin, whatever the language of the system, and never touch
the network."""
import os
import socket

os.environ.setdefault("PPG_LANG", "it")
os.environ.setdefault("PPG_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PPG_RENDERER", "software")

import pyarchinit_photogrammetry  # noqa: E402,F401 - first: it points rasterio at its own PROJ data
import pytest  # noqa: E402

from planfixtures import reference_plan  # noqa: E402


@pytest.fixture
def plan() -> dict:
    return reference_plan()


from ppg_flightplan import tiles as _tiles  # noqa: E402

REAL_TILE_FROM_NETWORK = _tiles._tile_from_network


_CONNECT, _CONNECT_EX = socket.socket.connect, socket.socket.connect_ex
_LOOPBACK = ("127.", "::1", "localhost")


def _outside(sock, address) -> bool:
    """A connection that would leave the machine: an internet socket to an address that is not loopback."""
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return False                       # local sockets (AF_UNIX: resource tracker, IPC) are fine
    host = str(address[0]) if isinstance(address, tuple) and address else str(address)
    return not host.startswith(_LOOPBACK)


def _refuse(self, address, *args):
    if _outside(self, address):
        raise RuntimeError(f"the tests do not reach the network ({address!r})")
    return _CONNECT(self, address, *args)


def _refuse_ex(self, address, *args):
    if _outside(self, address):
        raise RuntimeError(f"the tests do not reach the network ({address!r})")
    return _CONNECT_EX(self, address, *args)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test reaches the network: the public terrain tiles answer "no data" unless a test fakes them."""
    monkeypatch.setattr(_tiles, "_tile_from_network", lambda z, x, y: None)
    # and nothing else does either: the maps of the legs draw on the orthophoto or on grey, never on tiles
    # fetched from a server; a socket that tries to go out fails the test
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_ex)
    _tiles.clear_cache()
    _tiles.set_cache_dir(None)
    yield
    _tiles.clear_cache()
    _tiles.set_cache_dir(None)
