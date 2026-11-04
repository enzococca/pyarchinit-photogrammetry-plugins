"""The API key (environment, then the system keychain; nowhere else) and the client of the PhotoCAD API:
the requests as documented, the errors in plain words, 429 asked again after Retry-After, a failed
conversion, a cancelled one deleted on the server."""
from __future__ import annotations

import fakecad
import pytest

from ppg_photocad import client, keys


def test_key_from_the_environment_first_then_the_keychain(monkeypatch, _no_network_no_keychain):
    store = _no_network_no_keychain
    assert keys.get_key() is None and keys.key_source() is None
    keys.set_key("  pcad_live_abc  ")
    assert store.store == {("pyarchinit-photogrammetry", "photocad"): "pcad_live_abc"}
    assert keys.get_key() == "pcad_live_abc" and keys.key_source() == "keyring"
    monkeypatch.setenv("PHOTOCAD_API_KEY", "pcad_live_env")
    assert keys.get_key() == "pcad_live_env" and keys.key_source() == "env"
    monkeypatch.delenv("PHOTOCAD_API_KEY")
    keys.forget_key()
    assert keys.get_key() is None and store.store == {}
    keys.forget_key()  # nothing to forget: no error


def test_without_a_keychain_only_the_environment(monkeypatch):
    monkeypatch.setattr(keys, "_backend", lambda: None)
    assert not keys.keyring_available() and keys.get_key() is None
    with pytest.raises(keys.KeyringUnavailable) as err:
        keys.set_key("pcad_live_abc")
    assert "PHOTOCAD_API_KEY" in str(err.value)
    monkeypatch.setenv("PHOTOCAD_API_KEY", "pcad_live_env")
    assert keys.get_key() == "pcad_live_env"


def test_the_key_is_masked_in_any_text():
    k = fakecad.KEY
    assert k not in keys.mask(f"refused {k} by the server", k)
    assert keys.mask(f"x {k}", k).endswith("pcad_live_…cdef")
    assert keys.mask("nothing", None) == "nothing"


def _client(monkeypatch, fake):
    monkeypatch.setattr(client, "_urlopen", fake)
    return client.Client(fakecad.KEY)


def test_account_convert_poll_download(monkeypatch):
    fake = fakecad.FakePhotoCAD(credits=5, polls=1)
    c = _client(monkeypatch, fake)
    assert client.credits(c.account()) == 5
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray(fakecad.square_image(60, 40, (10, 10, 20)).transpose(1, 2, 0)).save(buf, "PNG")
    cid = c.convert(buf.getvalue(), "image.png", "image/png", detailed=False, remove_background=True)
    assert cid == "conv_0001" and fake.options[cid] == {"detailed": "false", "remove_background": "true"}
    seen = []
    done = client.wait(c, cid, lambda f, m: seen.append(m), interval=5.0)
    assert done["status"] == "complete" and client.credits(done) == 4 and len(fake.calls("GET", "/conversions/")) == 2
    dxf = c.file(cid, "dxf", units="m", longest=1.5, margin=0)
    assert b"OUTLINE" in dxf
    assert "units=m" in fake.calls("GET", "/conversions/conv_0001/files/dxf")[0]
    assert all(h["authorization"] == f"Bearer {fakecad.KEY}" for _m, _p, h in fake.requests)
    assert all("photocad" in h["user-agent"] for _m, _p, h in fake.requests)


@pytest.mark.parametrize("code, words", [(401, "chiave"), (402, "credit"), (413, "grande"), (422, "accettata")])
def test_errors_in_plain_words(monkeypatch, code, words):
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [code]})
    c = _client(monkeypatch, fake)
    with pytest.raises(client.PhotoCADError) as err:
        c.convert(b"\x89PNG", "image.png", "image/png")
    assert err.value.status == code and words in str(err.value).lower()
    assert fakecad.KEY not in str(err.value)


def test_a_refused_key(monkeypatch):
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    with pytest.raises(client.PhotoCADError) as err:
        client.Client("pcad_live_wrong").account()
    assert err.value.status == 401 and "pcad_live_wrong" not in str(err.value)


def test_too_many_requests_asked_again_after_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(client, "_sleep", waits.append)
    fake = fakecad.FakePhotoCAD(fail={"GET /account": [429, 429]})
    c = _client(monkeypatch, fake)
    assert client.credits(c.account()) == 5
    assert waits == [2.0, 2.0] and len(fake.calls("GET", "/account")) == 3
    fake = fakecad.FakePhotoCAD(fail={"GET /account": [429] * 4})
    c = _client(monkeypatch, fake)
    with pytest.raises(client.PhotoCADError) as err:
        c.account()
    assert err.value.status == 429 and len(fake.calls("GET", "/account")) == 4  # 1 + 3 new attempts


def test_a_failed_conversion_says_why(monkeypatch):
    fake = fakecad.FakePhotoCAD(status_failed=True)
    c = _client(monkeypatch, fake)
    fake.images["conv_0009"] = b""
    with pytest.raises(client.PhotoCADError) as err:
        client.wait(c, "conv_0009", None)
    assert "No drawing found" in str(err.value) and "rimborsat" in str(err.value)


def test_cancel_deletes_the_conversion(monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    fake = fakecad.FakePhotoCAD(polls=10)
    c = _client(monkeypatch, fake)
    fake.images["conv_0001"] = b""  # still processing: the image is never read
    p = Progress()
    polls = []

    def sleep(s):
        polls.append(s)
        if len(polls) == 3:
            p.cancel()

    monkeypatch.setattr(client, "_sleep", sleep)
    with pytest.raises(Cancelled):
        client.wait(c, "conv_0001", p, interval=5.0)
    assert fake.deleted == ["conv_0001"]
