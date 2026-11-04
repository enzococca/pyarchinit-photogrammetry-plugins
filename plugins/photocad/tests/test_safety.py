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
    api.routes["/api/v1/conversions/c1/files/dxf?units=m"] = (302, {"Location": storage.url + "/bucket/c1.dxf?sig=x"}, b"")
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
def test_the_upload_is_not_sent_again_on_a_5xx_but_again_on_429_and_a_lost_connection(monkeypatch):
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [502]})
    monkeypatch.setattr(client, "_urlopen", fake)
    c = client.Client(fakecad.KEY)
    with pytest.raises(client.PhotoCADError) as err:
        c.convert(b"x", "image.jpg", "image/jpeg", idempotency_key="k1")
    assert len(fake.calls("POST")) == 1 and err.value.status == 502
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [429]})  # the rate limit: not processed, asked again
    monkeypatch.setattr(client, "_urlopen", fake)
    assert client.Client(fakecad.KEY).convert(b"x", "image.jpg", "image/jpeg", idempotency_key="k2") == "conv_0001"
    assert len(fake.calls("POST")) == 2
    # the connection lost AFTER the server made the conversion: asked again with the same key, charged once
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [-1]})
    monkeypatch.setattr(client, "_urlopen", fake)
    assert client.Client(fakecad.KEY).convert(b"x", "image.jpg", "image/jpeg", idempotency_key="k3") == "conv_0001"
    keys_sent = [h.get("idempotency-key") for m, _p, h in fake.requests if m == "POST"]
    assert keys_sent == ["k3", "k3"] and fake.charged == 1
    # without a key a lost connection is not asked again
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [0]})
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
    fake = fakecad.FakePhotoCAD(fail={"GET /conversions/conv_0001/files/dxf": [500] * 4})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(client.PhotoCADError) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "conv_0001" in str(err.value) and "fetch" in str(err.value)
    (rec,) = project.records(ch)
    assert rec["status"] == "complete" and rec["tiles"][0]["photocad_id"] == "conv_0001"
    folder = project.folder_of(ch, rec)
    on_disk = json.loads((folder / "conversion.json").read_text())
    assert on_disk["tiles"][0]["photocad_id"] == "conv_0001" and (folder / "t0" / "image.jpg").exists()
    assert fake.deleted == [] and project.features(ch, rec) == []
    # fetched again: no new upload, the drawing in place
    rec = project.fetch(ch, rec["id"], fakecad.KEY)
    assert rec["status"] == "done" and len(fake.calls("POST")) == 1
    (square,) = [g for g, _a in project.features(ch, rec) if g.geom_type == "Polygon"]
    assert np.abs(np.asarray(square.bounds) - fakecad.square_world()).max() <= 1.5 * fakecad.RES / 2
    assert len(project.records(ch)) == 1


def test_written_before_the_upload_and_with_the_id_after_it(tmp_path, monkeypatch):
    """The idempotency key is on disk before the POST, the PhotoCAD id before the first poll."""
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    seen = {}
    real_convert, real_poll = client.Client.convert, client.Client.conversion

    def read():
        files = list((ch.data_dir / "photocad").rglob("conversion.json"))
        return json.loads(files[0].read_text()) if files else None

    def convert(self, *a, **k):
        seen["before_post"] = read()
        return real_convert(self, *a, **k)

    def first_poll(self, cid):
        seen.setdefault("first_poll", read())
        return real_poll(self, cid)

    monkeypatch.setattr(client.Client, "convert", convert)
    monkeypatch.setattr(client.Client, "conversion", first_poll)
    project.convert(ch, "ortho", fakecad.KEY)
    tile = seen["before_post"]["tiles"][0]
    assert tile["idempotency_key"] and tile.get("photocad_id") is None and tile["status"] == "new"
    tile = seen["first_poll"]["tiles"][0]
    assert tile["photocad_id"] == "conv_0001" and tile["status"] == "pending"
    sent = [h.get("idempotency-key") for m, _p, h in fake.requests if m == "POST"]
    assert sent == [seen["before_post"]["tiles"][0]["idempotency_key"]]


def test_a_lost_answer_is_sent_again_once_with_the_same_key(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(fail={"POST /convert": [-1]})  # made on the server, the answer lost
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    assert rec["status"] == "done" and fake.charged == 1 and len(fake.calls("POST")) == 2
    assert rec["tiles"][0]["posts"] == 2


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
    assert project.fetch(ch, rec["tiles"][0]["photocad_id"], fakecad.KEY)["status"] == "done"  # by the PhotoCAD id
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
    fake = fakecad.FakePhotoCAD(fail={"GET /conversions/conv_0001/files/dxf": [500] * 4})
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


def test_a_conversion_of_version_0_1_0_dev_is_fetched(tmp_path, monkeypatch):
    """A record made before the tiles (one PhotoCAD id, files in the folder itself), refused by the old frame check:
    fetch places it by registration."""
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    folder = project.folder_of(ch, rec)
    tile = rec["tiles"][0]
    (folder / "image.jpg").write_bytes((folder / "t0" / "image.jpg").read_bytes())
    old = {k: v for k, v in rec.items() if k not in ("tiles", "counts", "files")}
    old.update(photocad_id=tile["photocad_id"], status="mismatch", image=dict(tile["image"]),
               files={"image": "image.jpg"})
    ch.meta["photocad"]["conversions"] = [old]
    rec = project.fetch(ch, tile["photocad_id"], fakecad.KEY)
    assert rec["status"] == "done" and len(fake.calls("POST")) == 1
    (square,) = [g for g, _a in project.features(ch, rec) if g.geom_type == "Polygon"]
    assert np.abs(np.asarray(square.bounds) - fakecad.square_world()).max() <= 1.5 * fakecad.RES / 2


# ------------------------------------------------------------------ minors
def test_minors(tmp_path, monkeypatch, capsys):
    import ezdxf

    # DXF extents moved with the drawing
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    h = ezdxf.readfile(project.path_of(ch, rec, "dxf")).header
    x0, y0, x1, y1 = fakecad.square_world()
    tol = 1.5 * fakecad.RES / 2
    assert abs(h["$EXTMIN"][0] - x0) <= tol and abs(h["$EXTMIN"][1] - y0) <= tol
    assert abs(h["$EXTMAX"][0] - x1) <= tol and abs(h["$EXTMAX"][1] - y1) <= tol
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


def test_the_dem_nodata_is_white_not_a_cliff(tmp_path):
    """A DEM with numeric nodata: those cells white in the hillshade, the valid ones untouched (no cliff)."""
    import io

    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    from ppg_photocad import source

    doc, ch = fakecad.project(tmp_path)
    path = ch.data_dir / "dem2.tif"
    z = np.full((700, 900), 50.0, np.float32)
    z[:, -50:] = -9999.0
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                       crs=fakecad.CRS, transform=from_origin(fakecad.X0, fakecad.Y1, 0.01, 0.01), nodata=-9999) as d:
        d.write(z, 1)
    ch._set_product("dem", path)
    img = source.render(ch, "dem", window=(0, 0, 900, 700), zoom=1, fmt="png")
    a = np.asarray(Image.open(io.BytesIO(img.data)).convert("L"), float)
    assert a[:, -40:].min() >= 250 and np.ptp(a[5:-5, 5:-60]) <= 2


def test_failed_detail_is_masked(monkeypatch):
    class Leaky(client.Client):
        def conversion(self, cid):
            return {"status": "failed", "error": {"detail": f"bad key {fakecad.KEY}"}}

    with pytest.raises(client.PhotoCADError) as err:
        client.wait(Leaky(fakecad.KEY), "c", None)
    assert fakecad.KEY not in str(err.value)



# ------------------------------------------------------------------ the peak of the registration (review I4)
def test_a_textured_wall_placed_right_passes_with_a_clear_peak(tmp_path, monkeypatch):
    doc, ch, boxes = fakecad.masonry_project(tmp_path)
    fake = fakecad.FakePhotoCAD(boxes_px=boxes)
    monkeypatch.setattr(client, "_urlopen", fake)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    reg = rec["tiles"][0]["registration"]
    assert reg["ok"] and reg["peak_ratio"] >= 1.3 and not reg["at_border"]
    polys = sorted((g.bounds for g, _a in project.features(ch, rec) if g.geom_type == "Polygon"))
    c0, r0, c1, r1 = boxes[0]
    truth = (fakecad.X0 + c0 * fakecad.RES, fakecad.Y1 - r1 * fakecad.RES, fakecad.X0 + c1 * fakecad.RES,
             fakecad.Y1 - r0 * fakecad.RES)
    assert min(np.abs(np.asarray(b) - truth).max() for b in polys) <= 1.5 * fakecad.RES


def test_a_wall_drawn_beyond_the_search_is_refused(tmp_path, monkeypatch):
    doc, ch, boxes = fakecad.masonry_project(tmp_path)
    fake = fakecad.FakePhotoCAD(boxes_px=boxes, offset_px=(130, 0))  # 1.3 m off: beyond the widened search
    monkeypatch.setattr(client, "_urlopen", fake)
    with pytest.raises(project.FrameMismatch):
        project.convert(ch, "ortho", fakecad.KEY)
    reg = project.records(ch)[0]["tiles"][0]["registration"]
    assert not reg["ok"] and "peak_ratio" in reg


def test_random_lines_over_a_wall_are_refused(tmp_path, monkeypatch):
    doc, ch, boxes = fakecad.masonry_project(tmp_path)
    fake = fakecad.FakePhotoCAD(boxes_px=boxes, random_lines=2000)
    monkeypatch.setattr(client, "_urlopen", fake)
    with pytest.raises(project.FrameMismatch):
        project.convert(ch, "ortho", fakecad.KEY)
    reg = project.records(ch)[0]["tiles"][0]["registration"]
    assert reg["peak_ratio"] < 1.3


# ------------------------------------------------------------------ the idempotency key not honoured (review I5)
def test_without_idempotency_at_most_one_resend_per_tile(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(honour_keys=False, fail={"POST /convert": [-1, -1, -1, -1]})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(client.PhotoCADError) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert len(fake.calls("POST")) == 2 and "photocad.ai" in str(err.value)
    (rec,) = project.records(ch)
    assert rec["tiles"][0]["status"] == "unknown" and rec["tiles"][0]["posts"] == 2
    with pytest.raises(client.PhotoCADError):
        project.fetch(ch, rec["id"], fakecad.KEY)
    assert len(fake.calls("POST")) == 2  # fetch never sends a tile of unknown outcome again


def test_a_crashed_upload_is_sent_once_more_by_fetch(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD()
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    real = client.Client.convert

    def crash(self, *a, before_post=None, **k):  # the program dies while the POST is on its way
        if before_post is not None:
            before_post("attempt")
        raise KeyboardInterrupt

    monkeypatch.setattr(client.Client, "convert", crash)
    with pytest.raises(KeyboardInterrupt):
        project.convert(ch, "ortho", fakecad.KEY)
    (rec,) = project.records(ch)
    assert rec["tiles"][0]["posts"] == 1 and rec["tiles"][0]["status"] == "new"
    monkeypatch.setattr(client.Client, "convert", real)
    rec = project.fetch(ch, rec["id"], fakecad.KEY)
    assert rec["status"] == "done" and len(fake.calls("POST")) == 1


def test_a_cancel_whose_delete_fails_keeps_the_record(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    fake = fakecad.FakePhotoCAD(polls=50, fail={"DELETE /conversions/": [500] * 8})
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    q = Progress()
    naps = []

    def nap(s):
        naps.append(s)
        if len(naps) == 3:
            q.cancel()

    monkeypatch.setattr(client, "_sleep", nap)
    with pytest.raises(Cancelled):
        project.convert(ch, "ortho", fakecad.KEY, progress=q)
    (rec,) = project.records(ch)  # PhotoCAD may still make it (and charge it): kept, to check or fetch
    assert rec["tiles"][0]["status"] == "pending"
