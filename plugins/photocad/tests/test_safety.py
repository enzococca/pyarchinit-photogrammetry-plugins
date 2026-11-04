"""What protects the key and the credits: the key never follows a redirect to another origin (two real local
servers), an upload never sent twice on a 5xx, a paid conversion never lost (kept from the 202, polling through
network glitches, fetched again after a failed download or a cancel once it is complete), the drawing's frame
checked against the image (a server that scales the content instead of the page is caught)."""
from __future__ import annotations

import http.server
import json
import threading

import fakecad
import numpy as np
import pytest

from ppg_photocad import client, keys, project


# ------------------------------------------------------------------ redirects (two real servers)
class _Server:
    """A local HTTP server that records the headers of every request and answers from ``routes``:
    path -> (status, headers, body)."""

    def __init__(self):
        self.seen: list[tuple[str, dict]] = []
        self.routes: dict = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                status, headers, body = outer.routes.get(self.path, (404, {}, b"{}"))
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setattr(client, "_urlopen", client.urlopen_no_redirect)  # the real network, to localhost only
    a, b = _Server(), _Server()
    yield a, b
    a.close()
    b.close()


def test_the_key_never_follows_a_redirect_to_another_origin(servers):
    api, storage = servers
    api.routes["/api/v1/conversions/c1/files/dxf"] = (302, {"Location": storage.url + "/bucket/c1.dxf?sig=x"}, b"")
    storage.routes["/bucket/c1.dxf?sig=x"] = (200, {}, b"0\nSECTION\n")
    c = client.Client(fakecad.KEY, base=api.url + "/api/v1")
    assert c.file("c1", "dxf", units="m") == b"0\nSECTION\n"  # the download still works
    assert api.seen[0][1]["authorization"] == f"Bearer {fakecad.KEY}"
    assert storage.seen and "authorization" not in storage.seen[0][1]  # never the key to the storage


def test_a_redirect_on_the_same_origin_keeps_the_key(servers):
    api, _storage = servers
    api.routes["/api/v1/account"] = (301, {"Location": "/api/v1/me"}, b"")
    api.routes["/api/v1/me"] = (200, {"Content-Type": "application/json"}, b'{"credits_remaining": 4}')
    c = client.Client(fakecad.KEY, base=api.url + "/api/v1")
    assert client.credits(c.account()) == 4
    assert [h.get("authorization") for _p, h in api.seen] == [f"Bearer {fakecad.KEY}"] * 2


def test_redirect_rules():
    base = "https://photocad.ai/api/v1"
    assert client.same_origin(base, "https://photocad.ai/x")
    assert not client.same_origin(base, "https://storage.example.com/x")
    assert not client.same_origin(base, "http://photocad.ai/x")
    assert not client.same_origin(base, "https://photocad.ai:8443/x")
    with pytest.raises(client.PhotoCADError):  # https -> http refused
        client.check_redirect(base, "https://photocad.ai/a", "http://storage.example.com/b")


# ------------------------------------------------------------------ never pay twice, never lose what was paid
def test_the_upload_is_not_sent_again_on_a_5xx(monkeypatch):
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [502]})
    monkeypatch.setattr(client, "_urlopen", fake)
    c = client.Client(fakecad.KEY)
    with pytest.raises(client.PhotoCADError) as err:
        c.convert(b"x", "image.jpg", "image/jpeg")
    assert len(fake.calls("POST")) == 1 and err.value.status == 502
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [429]})  # the rate limit: not processed, asked again
    monkeypatch.setattr(client, "_urlopen", fake)
    assert client.Client(fakecad.KEY).convert(b"x", "image.jpg", "image/jpeg") == "conv_0001"
    assert len(fake.calls("POST")) == 2
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [0]})  # a broken connection: not sent again either
    monkeypatch.setattr(client, "_urlopen", fake)
    with pytest.raises(client.PhotoCADError):
        client.Client(fakecad.KEY).convert(b"x", "image.jpg", "image/jpeg")
    assert len(fake.calls("POST")) == 1


def test_polling_goes_through_network_glitches(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(fail={"GET /conversions/conv_0001": [0, 503, 0]})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    assert rec["status"] == "done" and rec["counts"]["polygons"] == 1


def test_a_failed_download_keeps_the_paid_conversion_and_fetch_gets_it(tmp_path, monkeypatch, capsys):
    fake = fakecad.FakePhotoCAD(fail={"GET /conversions/conv_0001/files/dxf": [500] * 8})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(client.PhotoCADError) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "conv_0001" in str(err.value) and "fetch" in str(err.value)
    (rec,) = project.records(ch)
    assert rec["status"] == "complete" and rec["photocad_id"] == "conv_0001"
    folder = project.folder_of(ch, rec)
    on_disk = json.loads((folder / "conversion.json").read_text())
    assert on_disk["photocad_id"] == "conv_0001" and (folder / "image.jpg").exists()
    assert fake.deleted == [] and project.features(ch, rec) == []
    # fetched again: no new upload, the drawing in place
    rec = project.fetch(ch, rec["id"], fakecad.KEY)
    assert rec["status"] == "done" and len(fake.calls("POST")) == 1
    (square,) = [g for g, _a in project.features(ch, rec) if g.geom_type == "Polygon"]
    assert np.allclose(square.bounds, fakecad.square_world(), rtol=0, atol=fakecad.RES / 2)
    assert len(project.records(ch)) == 1


def test_written_right_after_the_upload(tmp_path, monkeypatch):
    """The PhotoCAD id is on disk before the first poll."""
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    seen = {}
    real = client.Client.conversion

    def first_poll(self, cid):
        if not seen:
            files = list((ch.data_dir / "photocad").rglob("conversion.json"))
            seen["json"] = json.loads(files[0].read_text()) if files else None
        return real(self, cid)

    monkeypatch.setattr(client.Client, "conversion", first_poll)
    project.convert(ch, "ortho", fakecad.KEY)
    assert seen["json"]["photocad_id"] == "conv_0001" and seen["json"]["status"] == "pending"


def test_cancel_after_completion_keeps_it_cancel_before_deletes_it(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    p = Progress()

    def report(fraction, message=""):
        if fraction >= 0.92:  # the conversion is complete, downloading
            p.cancel()
        p(fraction, message)

    with pytest.raises(Cancelled):
        project.convert(ch, "ortho", fakecad.KEY, progress=report)
    (rec,) = project.records(ch)
    assert rec["status"] == "complete" and fake.deleted == []
    assert project.fetch(ch, rec["photocad_id"], fakecad.KEY)["status"] == "done"  # by the PhotoCAD id too
    # before completion: deleted on PhotoCAD, nothing kept
    fake = fakecad.FakePhotoCAD(polls=50)
    monkeypatch.setattr(client, "_urlopen", fake)
    q = Progress()
    naps = []

    def nap(s):
        naps.append(s)
        if len(naps) == 3:
            q.cancel()

    monkeypatch.setattr(client, "_sleep", nap)
    with pytest.raises(Cancelled):
        project.convert(ch, "ortho", fakecad.KEY, progress=q)
    assert fake.deleted == ["conv_0001"] and len(project.records(ch)) == 1


def test_ctrl_c_while_waiting_deletes_the_conversion(monkeypatch):
    fake = fakecad.FakePhotoCAD(polls=50)
    monkeypatch.setattr(client, "_urlopen", fake)
    fake.images["conv_0001"] = b""

    def interrupt(s):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "_sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        client.wait(client.Client(fakecad.KEY), "conv_0001", None)
    assert fake.deleted == ["conv_0001"]


def test_fetch_from_the_command_line(tmp_path, monkeypatch, capsys):
    from ppg_photocad import PhotoCADPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    app.document = None
    plugin = PhotoCADPlugin()
    plugin.activate(Context("photocad", tmp_path / "plugin"))
    assert "Recupera una conversione..." in [a.title for a in plugin.context.contributions.actions]
    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    fake = fakecad.FakePhotoCAD(fail={"GET /conversions/conv_0001/files/dxf": [500] * 8})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path / "p")
    path = str(doc.path)
    assert plugin.command("convert", path, "--source", "ortho", "--yes") == 1
    out = capsys.readouterr().out
    assert "conv_0001" in out and "fetch" in out
    assert plugin.command("fetch", path, "--id", "conv_0001", "--out", str(tmp_path / "f.gpkg")) == 0
    assert (tmp_path / "f.gpkg").exists() and len(fake.calls("POST")) == 1
    assert plugin.command("fetch", path, "--id", "conv_9999") == 1
    import pyarchinit_photogrammetry as ppg

    (rec,) = project.records(ppg.Document.open(path).chunk)
    assert rec["status"] == "done"


# ------------------------------------------------------------------ the frame of the drawing
def test_a_drawing_away_from_the_edges_is_placed_when_the_page_is_scaled(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(diagonal=False)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    assert rec["check"]["ok"] and abs(rec["check"]["scale_ratio"] - 1) < 0.01
    (square,) = [g for g, _a in project.features(ch, rec)]
    assert np.allclose(square.bounds, fakecad.square_world(), rtol=0, atol=fakecad.RES / 2)


def test_content_scaled_instead_of_the_page_is_caught(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(scale="content", diagonal=False)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(project.FrameMismatch) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "conv_0001" in str(err.value) and "PhotoCAD" in str(err.value)
    (rec,) = project.records(ch)
    assert rec["status"] == "mismatch" and not rec["check"]["ok"] and rec["check"]["problems"]
    folder = project.folder_of(ch, rec)
    assert (folder / "photocad.dxf").exists() and (folder / "photocad_mm.dxf").exists()
    assert not (folder / "drawing.gpkg").exists() and project.features(ch, rec) == []


# ------------------------------------------------------------------ minors
def test_minors(tmp_path, monkeypatch, capsys):
    import ezdxf

    # DXF extents moved with the drawing
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    h = ezdxf.readfile(project.path_of(ch, rec, "dxf")).header
    assert h["$EXTMIN"][0] == pytest.approx(fakecad.X0) and h["$EXTMAX"][0] == pytest.approx(
        fakecad.X0 + fakecad.W * fakecad.RES, abs=1e-6)
    assert h["$EXTMAX"][1] == pytest.approx(fakecad.Y1, abs=1e-6)
    # Retry-After as an HTTP date
    assert client.retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 1.0 and client.retry_after("7") == 7.0
    # a short key never shown whole
    assert "pcad_abcdefgh" not in keys.masked("pcad_abcdefgh")
    # no terminal: declined, no traceback
    from ppg_photocad import PhotoCADPlugin
    from pyarchinit_photogrammetry.plugins import Context

    plugin = PhotoCADPlugin()
    plugin.activate(Context("photocad", tmp_path / "plugin"))
    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)

    def eof(prompt=""):
        raise EOFError

    import builtins

    monkeypatch.setattr(builtins, "input", eof)
    posts = len(fake.calls("POST"))
    assert plugin.command("convert", str(doc.path), "--source", "ortho") == 1
    assert len(fake.calls("POST")) == posts


def test_the_dem_nodata_at_the_partial_edge(tmp_path):
    """A DEM with numeric nodata, reduced by k > 1: the partial edge blocks average the valid cells only."""
    import rasterio
    from rasterio.transform import from_origin

    from ppg_photocad import source

    doc, ch = fakecad.project(tmp_path)
    path = ch.data_dir / "dem2.tif"
    z = np.full((10, 7301), 50.0, np.float32)
    z[:, -1] = -9999.0  # the last column (in the partial block of k = 4) is nodata
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                       crs=fakecad.CRS, transform=from_origin(fakecad.X0, fakecad.Y1, 0.01, 0.01), nodata=-9999) as d:
        d.write(z, 1)
    with rasterio.open(path) as src:
        c0, r0, nc, nr, k = source._window(src, None, 2400)
        a = source._read(src, [1], c0, r0, nc, nr, k, np.nan, src.nodata)[0]
    assert k == 4 and np.nanmax(a) == pytest.approx(50.0) and np.nanmin(a) == pytest.approx(50.0)


def test_failed_detail_is_masked(monkeypatch):
    class Leaky(client.Client):
        def conversion(self, cid):
            return {"status": "failed", "error": {"detail": f"bad key {fakecad.KEY}"}}

    with pytest.raises(client.PhotoCADError) as err:
        client.wait(Leaky(fakecad.KEY), "c", None)
    assert fakecad.KEY not in str(err.value)

