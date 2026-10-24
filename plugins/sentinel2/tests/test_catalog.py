"""The STAC client with a fake catalog: pages followed, clouds filtered by the server and again
here, busy servers asked again, the query extension refused (then CQL2), the items grouped into
acquisitions (two tiles of a datatake together, a tile processed twice kept once), the scale and
offset of every band; never the network."""
from __future__ import annotations

import pytest
import s2synth

from ppg_sentinel2 import stac


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    return s2synth.world(tmp_path_factory.mktemp("cogs"))


BBOX = (9.0, 45.17, 9.03, 45.20)


def test_search_follows_the_pages_and_filters_the_clouds(items):
    fake = s2synth.FakeCatalog(items, page=3)
    seen = []
    found = stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=fake,
                        progress=lambda f, m: seen.append((f, m)))
    # nine items match the dates; the server forgets to drop the stormy one, the client does
    assert len(fake.requests) == 3 and all(method == "POST" for _url, method, _b in fake.requests)
    first = fake.requests[0][2]
    assert first["collections"] == ["sentinel-2-l2a"] and first["bbox"] == list(BBOX)
    assert first["datetime"] == "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z"
    assert first["query"] == {"eo:cloud_cover": {"lte": 30.0}} and "next" not in first
    assert [b.get("next") for _u, _m, b in fake.requests[1:]] == ["3", "6"]
    assert len(found) == 8 and all(it["properties"]["eo:cloud_cover"] <= 30 for it in found)
    assert "S2B_32TMR_20231001_0_L2A" not in {it["id"] for it in found}
    assert seen and seen[-1][1] == "catalog: 9 scenes of 9"  # what the server sent, the stormy one too
    # dates outside the span are not asked for
    fake = s2synth.FakeCatalog(items, page=50)
    assert len(stac.search(BBOX, "2023-06-01", "2023-07-31", 30, s2synth.CATALOG, opener=fake)) == 3


def test_busy_servers_are_asked_again_and_refusals_explained(items, monkeypatch):
    monkeypatch.setattr(stac, "RETRY_DELAY", 0.0)
    fake = s2synth.FakeCatalog(items, page=100, fail=2)
    assert len(stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=fake)) == 8
    assert len(fake.requests) == 3
    fake = s2synth.FakeCatalog(items, page=100, fail=10)
    with pytest.raises(stac.CatalogError, match="HTTP 503"):
        stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=fake)
    # a server without the query extension: CQL2, the same scenes
    fake = s2synth.FakeCatalog(items, page=100, refuse_query=True)
    found = stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=fake)
    assert len(found) == 8
    assert "query" in fake.requests[0][2] and fake.requests[1][2]["filter-lang"] == "cql2-json"
    assert fake.requests[1][2]["filter"] == {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 30.0]}

    def broken(request, timeout):
        raise OSError("no route to host")

    with pytest.raises(stac.CatalogError, match="no route to host"):
        stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=broken)


def test_not_a_stac_answer(monkeypatch):
    def opener(request, timeout):
        return s2synth.Response(b'{"hello": "world"}')

    with pytest.raises(stac.CatalogError, match="not a STAC item collection"):
        stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=opener)

    def html(request, timeout):
        return s2synth.Response(b"<html>maintenance</html>")

    with pytest.raises(stac.CatalogError, match="not JSON"):
        stac.search(BBOX, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=html)


def test_items_become_acquisitions(items):
    found = stac.keep_clear(items, 30)
    acqs = stac.acquisitions(found)
    assert [a["id"] for a in acqs] == ["20230410_S2B", "20230520_S2B", "20230614_S2B", "20230709_S2B",
                                       "20230803_S2B", "20230912_S2B"]
    june = acqs[2]
    assert june["tiles"] == ["32TMR", "32TNR"] and len(june["items"]) == 2 and june["cloud"] == 4.0
    august = acqs[4]
    assert [it["id"] for it in august["items"]] == ["S2B_32TMR_20230803_1_L2A"]  # the later processing
    assert june["date"] == "2023-06-14" and june["platform"] == "S2B" and june["orbit"] == 65
    # scale and offset: from raster:bands, or from the processing baseline (old products: no offset)
    red = june["items"][0]["assets"]["B04"]
    assert red["href"].endswith("2023-06-14_32TMR_B04.tif") and (red["scale"], red["offset"]) == (0.0001, -0.1)
    old = acqs[0]["items"][0]["assets"]
    assert (old["B04"]["scale"], old["B04"]["offset"]) == (0.0001, 0.0)
    assert (old["SCL"]["scale"], old["SCL"]["offset"]) == (1.0, 0.0)
    # harmonised by Earth Search: no offset, whatever raster:bands says
    aug = august["items"][0]["assets"]["B04"]
    assert (aug["scale"], aug["offset"]) == (0.0001, 0.0)
    # a baseline from 04.00 without raster:bands: the offset of ESA
    item = dict(found[0])
    item["properties"] = {**item["properties"], "s2:processing_baseline": "04.00"}
    assert stac.asset_of(item, "B04")["offset"] == -0.1
    # a COG preferred to its JP2, the asset found by its band name too
    assert not stac.asset_of(found[0], "B08")["href"].endswith(".jp2")
    renamed = {"properties": {}, "assets": {"B11": {"href": "x/B11.tif"}}}
    assert stac.asset_of(renamed, "B11")["href"] == "x/B11.tif" and stac.asset_of(renamed, "B04") is None


def test_dates_of_the_search():
    from datetime import timezone

    assert stac.parse_time("2023-06-14T10:20:31.1234567Z").microsecond == 123456
    assert stac.parse_time("2023-06-14T10:20:31.5Z").microsecond == 500000
    assert stac.parse_time("2023-06-14T10:20:31Z").tzinfo == timezone.utc
    assert stac.parse_time("2023-06-14").day == 14
    assert stac.interval("2023-06-01", "2023-06-30") == "2023-06-01T00:00:00Z/2023-06-30T23:59:59Z"
    assert stac.interval(None, "2023-06-30") == "../2023-06-30T23:59:59Z"
    with pytest.raises(ValueError):
        stac.interval("2023-13-01", None)
    body = stac.search_body(BBOX, "2023-01-01", "2023-02-01", 100)
    assert "query" not in body and "filter" not in body
