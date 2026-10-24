"""A small STAC API client: the Sentinel-2 L2A scenes over an area and a span of dates.

The default catalog is Earth Search v1 (Element 84: the collection ``sentinel-2-l2a``, public
Cloud-Optimized GeoTIFFs on AWS, no key); any STAC API with item search works. ``pystac_client``
is used when it can be imported, the standard library otherwise (the packaged application has
no pystac-client): POST ``/search`` with ``bbox``, ``datetime``, ``collections`` and a filter on
``eo:cloud_cover`` (the query extension; CQL2 when the server refuses it, nothing when it refuses
both), the pages followed through the ``next`` links (POST with their body, merged when the link
says so, or GET). The cloud cover is checked again here, so a server that ignores the filter
does not let cloudy scenes through.

The items become *acquisitions*: the tiles of one datatake (a satellite passing over the area on
one day) belong together, the copies of a tile processed again kept once (the latest), every
band an asset with its href and the scale and offset that turn its numbers into reflectance
(the ``raster:bands`` of the asset; without them 1/10000 and, from processing baseline 04.00,
the offset of -0.1 that ESA added to the L2A products in January 2022; none when Earth Search
says it took that offset away from the numbers already, ``earthsearch:boa_offset_applied``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime

log = logging.getLogger("pyarchinit_photogrammetry.plugins.sentinel2")

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B05", "B08", "B8A", "B11")  # the reflectance bands read, then SCL
ASSET_KEYS = {"B02": ("blue", "B02", "b02"), "B03": ("green", "B03", "b03"), "B04": ("red", "B04", "b04"),
              "B05": ("rededge1", "B05", "b05"), "B08": ("nir", "B08", "b08"), "B8A": ("nir08", "B8A", "b8a"),
              "B11": ("swir16", "B11", "b11"), "SCL": ("scl", "SCL")}
RESOLUTION = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20, "B8A": 20, "B11": 20, "SCL": 20}
PAGE = 100  # items per page asked to the server
MAX_ITEMS = 2000  # a search never brings back more (a wide span of years over a large area)
TIMEOUT = 60.0
RETRIES = 3
RETRY_DELAY = 1.0  # seconds before the first new attempt, doubled each time
RETRY_STATUS = (408, 429, 500, 502, 503, 504)
CLIENT_ENV = "PPG_SENTINEL2_CLIENT"  # "stdlib" or "pystac": the client used (auto by default)


class CatalogError(RuntimeError):
    """The catalog could not be searched: no network, a refusal, an answer that is not STAC."""


# ------------------------------------------------------------------ the network
def _ssl_context():
    """The certificates of certifi when present (a packaged application may not find those of
    the Python it was built with), else the system ones."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _urlopen(request, timeout: float):
    """The network: urllib with the certificates of certifi. The tests replace this function."""
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())


def user_agent() -> str:
    try:
        from pyarchinit_photogrammetry import __version__ as app
    except Exception:  # noqa: BLE001
        app = "?"
    from . import __version__

    return f"pyarchinit-photogrammetry/{app} ppg-sentinel2/{__version__}"


def _detail(exc: urllib.error.HTTPError) -> str:
    """The first words of the server's explanation of a refusal."""
    try:
        text = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""
    try:
        data = json.loads(text)
        text = str(data.get("description") or data.get("detail") or data.get("message") or data.get("code") or text)
    except ValueError:
        pass
    text = " ".join(text.split())
    return f": {text[:300]}" if text else ""


def request_json(url: str, body: dict | None = None, opener=None, method: str | None = None,
                 timeout: float = TIMEOUT, retries: int = RETRIES, check=None) -> dict:
    """``url`` asked with ``body`` as JSON (POST) or without (GET); the answer read as JSON.
    Busy servers and broken connections are asked again (``retries`` times, waiting longer each
    time); a refusal raises :class:`CatalogError` with what the server said. ``check()`` is called
    before each attempt (it raises when the user cancels)."""
    opener = opener or _urlopen
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"User-Agent": user_agent(), "Accept": "application/geo+json, application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data is not None else "GET"))
    delay = RETRY_DELAY
    for attempt in range(retries + 1):
        if check is not None:
            check()
        try:
            with opener(req, timeout) as answer:
                raw = answer.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt < retries:
                log.info("%s: HTTP %s, again in %.0f s", url, exc.code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            raise CatalogError(f"{url}: HTTP {exc.code} {exc.reason}{_detail(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt < retries:
                log.info("%s: %s, again in %.0f s", url, getattr(exc, "reason", exc), delay)
                time.sleep(delay)
                delay *= 2
                continue
            raise CatalogError(f"{url}: {getattr(exc, 'reason', exc)}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CatalogError(f"{url}: the answer is not JSON") from exc


# ------------------------------------------------------------------ the search
def _day(value, end: bool = False) -> str:
    """``2023-06-14`` (a date, a datetime or a string) as the start or the end of that day in UTC."""
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        value = value.isoformat()
    value = str(value).strip()
    if "T" in value:
        return value if value.endswith("Z") or "+" in value[10:] else value + "Z"
    date.fromisoformat(value)  # a readable date, or ValueError
    return f"{value}T23:59:59Z" if end else f"{value}T00:00:00Z"


def interval(start, end) -> str:
    """The STAC ``datetime`` of the days from ``start`` to ``end`` (either may be None: open)."""
    return f"{_day(start) if start else '..'}/{_day(end, end=True) if end else '..'}"


def search_body(bbox, start, end, max_cloud: float | None = None, collection: str = COLLECTION,
                limit: int = PAGE, cloud_filter: str = "query") -> dict:
    """The body of a POST ``/search``; ``cloud_filter``: ``query`` (the query extension), ``cql2``
    (the filter extension, CQL2 JSON) or ``none``."""
    body = {"collections": [collection], "bbox": [round(float(v), 7) for v in bbox],
            "datetime": interval(start, end), "limit": int(limit)}
    if max_cloud is not None and float(max_cloud) < 100 and cloud_filter != "none":
        if cloud_filter == "cql2":
            body["filter-lang"] = "cql2-json"
            body["filter"] = {"op": "<=", "args": [{"property": "eo:cloud_cover"}, float(max_cloud)]}
        else:
            body["query"] = {"eo:cloud_cover": {"lte": float(max_cloud)}}
    return body


def _next_link(answer: dict) -> dict | None:
    return next((link for link in answer.get("links") or [] if link.get("rel") == "next" and link.get("href")), None)


def _have_pystac() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pystac_client") is not None
    except (ImportError, ValueError):
        return False


def search(bbox, start, end, max_cloud: float | None = 30.0, catalog: str = EARTH_SEARCH,
           collection: str = COLLECTION, limit: int = PAGE, max_items: int = MAX_ITEMS, opener=None,
           client: str | None = None, progress=None) -> list[dict]:
    """The STAC items of ``collection`` over ``bbox`` (west, south, east, north in degrees) from
    ``start`` to ``end`` with a cloud cover of ``max_cloud`` % or less, as dictionaries.

    ``client``: ``pystac`` (pystac-client), ``stdlib`` (the standard library) or ``auto`` (the
    default, also from the environment variable ``PPG_SENTINEL2_CLIENT``): pystac-client when it
    can be imported and no ``opener`` is given, the standard library otherwise or when
    pystac-client fails. ``progress(fraction, message)`` after each page."""
    progress = progress or (lambda *a: None)
    which = (client or os.environ.get(CLIENT_ENV) or "auto").lower()
    items = None
    if opener is None and which in ("auto", "pystac") and _have_pystac():
        try:
            items = _search_pystac(bbox, start, end, max_cloud, catalog, collection, limit, max_items, progress)
        except Exception as exc:  # noqa: BLE001 - the standard library is always there
            if which == "pystac":
                raise CatalogError(f"{catalog}: {exc}") from exc
            log.warning("pystac-client could not search %s (%s): the standard library does", catalog, exc)
            items = None
    elif which == "pystac" and opener is None:
        raise CatalogError("pystac-client is not installed (pip install pystac-client)")
    if items is None:
        items = _search_stdlib(bbox, start, end, max_cloud, catalog, collection, limit, max_items, opener, progress)
    return keep_clear(items, max_cloud)


def keep_clear(items: list[dict], max_cloud: float | None) -> list[dict]:
    """The items of ``max_cloud`` % or less of clouds (those without a cloud cover kept)."""
    if max_cloud is None or float(max_cloud) >= 100:
        return list(items)
    out = []
    for item in items:
        cloud = (item.get("properties") or {}).get("eo:cloud_cover")
        if cloud is None or float(cloud) <= float(max_cloud):
            out.append(item)
    return out


def _search_stdlib(bbox, start, end, max_cloud, catalog, collection, limit, max_items, opener, progress) -> list[dict]:
    url = catalog.rstrip("/") + "/search"
    check = getattr(progress, "check", None)
    answer = None
    body = None
    for mode in ("query", "cql2", "none"):  # the cloud filter the server accepts
        body = search_body(bbox, start, end, max_cloud, collection, limit, mode)
        try:
            answer = request_json(url, body, opener, check=check)
            break
        except CatalogError as exc:
            refused = re.search(r"HTTP (400|422|501)\b", str(exc))
            if not refused or mode == "none" or "query" not in body and "filter" not in body:
                raise
            log.info("%s refused the cloud filter as %s (%s): asking again", url, mode, exc)
    items: list[dict] = []
    seen: set[str] = set()
    page = 0
    while True:
        features = answer.get("features") if isinstance(answer, dict) else None
        if not isinstance(features, list):
            raise CatalogError(f"{url}: the answer is not a STAC item collection")
        items += [f for f in features if isinstance(f, dict)]
        page += 1
        matched = answer.get("numberMatched") or (answer.get("context") or {}).get("matched")
        total = min(int(matched), max_items) if isinstance(matched, (int, float)) and matched else None
        progress(min(0.95, len(items) / total) if total else min(0.9, 0.1 * page),
                 f"catalog: {len(items)} scenes" + (f" of {total}" if total else ""))
        link = _next_link(answer)
        if link is None or not features or len(items) >= max_items or (total is not None and len(items) >= total):
            break  # (a server may hand a next link after its last item: an empty page saved)
        method = str(link.get("method") or "GET").upper()
        if method == "POST":
            nbody = {**body, **(link.get("body") or {})} if link.get("merge") else (link.get("body") or body)
            key = "POST " + link["href"] + json.dumps(nbody, sort_keys=True)
            if key in seen:  # a server that hands the same page again
                break
            seen.add(key)
            body = nbody
            answer = request_json(link["href"], nbody, opener, check=check)
        else:
            key = "GET " + link["href"]
            if key in seen:
                break
            seen.add(key)
            answer = request_json(link["href"], None, opener, method="GET", check=check)
    return items[:max_items]


def _search_pystac(bbox, start, end, max_cloud, catalog, collection, limit, max_items, progress) -> list[dict]:
    from pystac_client import Client

    client = Client.open(catalog, headers={"User-Agent": user_agent()})
    kwargs = {"collections": [collection], "bbox": [float(v) for v in bbox], "datetime": interval(start, end),
              "limit": int(limit), "max_items": int(max_items)}
    if max_cloud is not None and float(max_cloud) < 100:
        kwargs["query"] = {"eo:cloud_cover": {"lte": float(max_cloud)}}
    items = []
    for i, item in enumerate(client.search(**kwargs).items_as_dicts()):
        items.append(item)
        if i % 50 == 49:
            progress(min(0.9, len(items) / max_items), f"catalog: {len(items)} scenes")
    return items


# ------------------------------------------------------------------ the items as acquisitions
def platform_of(item: dict) -> str:
    """``S2A``, ``S2B``, ``S2C`` (from the platform, or from the item id)."""
    p = str((item.get("properties") or {}).get("platform") or "").lower()
    m = re.match(r"sentinel-?2([a-z])", p)
    if m:
        return "S2" + m.group(1).upper()
    m = re.match(r"(S2[A-Z])_", str(item.get("id", "")))
    return m.group(1) if m else "S2"


def tile_of(item: dict) -> str:
    """The MGRS tile (``33TUG``) of an item."""
    props = item.get("properties") or {}
    code = props.get("grid:code") or props.get("s2:mgrs_tile") or props.get("tile")
    if code:
        return str(code).replace("MGRS-", "")
    if props.get("mgrs:utm_zone") is not None:
        return f"{props['mgrs:utm_zone']}{props.get('mgrs:latitude_band', '')}{props.get('mgrs:grid_square', '')}"
    m = re.search(r"_(\d{1,2}[A-Z]{3})_", str(item.get("id", "")))
    return m.group(1) if m else ""


def epsg_of(item: dict) -> int | None:
    props = item.get("properties") or {}
    if props.get("proj:epsg"):
        return int(props["proj:epsg"])
    code = props.get("proj:code")
    if code and str(code).upper().startswith("EPSG:"):
        return int(str(code).split(":")[1])
    return None


def orbit_of(item: dict) -> int | None:
    props = item.get("properties") or {}
    if props.get("sat:relative_orbit") is not None:
        return int(props["sat:relative_orbit"])
    m = re.search(r"_R(\d{3})_", str(props.get("s2:product_uri") or ""))
    return int(m.group(1)) if m else None


def baseline_of(item: dict) -> float | None:
    value = (item.get("properties") or {}).get("s2:processing_baseline")
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.search(r"_N(\d{2})(\d{2})_", str((item.get("properties") or {}).get("s2:product_uri") or ""))
        return float(f"{m.group(1)}.{m.group(2)}") if m else None


def asset_of(item: dict, band: str) -> dict | None:
    """``{"href", "scale", "offset"}`` of a band of an item (None when the item has not got it):
    the asset by its common name (Earth Search v1) or its band name; a COG preferred to a JP2."""
    assets = item.get("assets") or {}
    found = None
    for key in ASSET_KEYS[band]:
        a = assets.get(key)
        if isinstance(a, dict) and a.get("href"):
            found = a
            break
    if found is None:  # any asset whose eo:bands name the band
        for a in assets.values():
            names = {str(b.get("name", "")).upper() for b in a.get("eo:bands") or [] if isinstance(b, dict)}
            if band in names and "jp2" not in str(a.get("type", "")).lower():
                found = a
                break
    if found is None:
        return None
    rb = (found.get("raster:bands") or [{}])[0] if isinstance(found.get("raster:bands"), list) else {}
    props = item.get("properties") or {}
    if band == "SCL":
        scale, offset = 1.0, 0.0
    elif props.get("earthsearch:boa_offset_applied") is True:
        # Earth Search v1 harmonised these numbers (the offset of baseline 04.00 already taken
        # away) but its raster:bands may still list the offset: the numbers win
        scale, offset = float(rb.get("scale", 1e-4)), 0.0
    elif "scale" in rb or "offset" in rb:
        scale, offset = float(rb.get("scale", 1e-4)), float(rb.get("offset", 0.0))
    else:  # no raster:bands: the rules of ESA's products
        baseline = baseline_of(item)
        scale, offset = 1e-4, (-0.1 if baseline is not None and baseline >= 4.0 else 0.0)
    return {"href": str(found["href"]), "scale": scale, "offset": offset}


def parse_time(value: str) -> datetime:
    """An ISO 8601 time as STAC servers write it (``Z``, fractions of a second of any length:
    Python 3.10 reads only 3 or 6 digits); a day alone at midnight UTC."""
    s = str(value).strip().replace("Z", "+00:00")
    m = re.match(r"^(.*T\d{2}:\d{2}:\d{2})\.(\d+)(.*)$", s)
    if m:
        s = f"{m.group(1)}.{(m.group(2) + '000000')[:6]}{m.group(3)}"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s[:10] + "T00:00:00+00:00")


def _when(item: dict) -> datetime:
    value = str((item.get("properties") or {}).get("datetime") or "")
    if not value:
        value = str((item.get("properties") or {}).get("start_datetime") or "1970-01-01T00:00:00Z")
    return parse_time(value)


def _sequence(item: dict) -> tuple:
    props = item.get("properties") or {}
    try:
        seq = int(props.get("s2:sequence", 0))
    except (TypeError, ValueError):
        seq = 0
    return seq, str(props.get("updated") or props.get("created") or ""), str(item.get("id", ""))


def acquisitions(items: list[dict]) -> list[dict]:
    """The items grouped by datatake (or day and satellite), sorted by time: ``{"id"
    ("20230614_S2B"), "date", "datetime", "platform", "orbit", "cloud" (the most of its tiles),
    "tiles", "items": [{"id", "tile", "epsg", "cloud", "assets": {band: {"href", "scale",
    "offset"}}}]}``. A tile processed more than once (``_0_L2A``, ``_1_L2A``) is kept once, the
    latest; items without every band are left out."""
    latest: dict[tuple, dict] = {}
    for item in items:
        when = _when(item)
        key = (tile_of(item) or item.get("id"), when.date().isoformat(), platform_of(item))
        if key not in latest or _sequence(item) > _sequence(latest[key]):
            latest[key] = item
    groups: dict[str, list[dict]] = {}
    for item in latest.values():
        props = item.get("properties") or {}
        when = _when(item)
        take = str(props.get("s2:datatake_id") or f"{when.date().isoformat()}_{platform_of(item)}_{orbit_of(item)}")
        groups.setdefault(take, []).append(item)
    out = []
    for take, members in groups.items():
        members.sort(key=lambda it: ((it.get("properties") or {}).get("eo:cloud_cover") or 0.0, tile_of(it)))
        parts = []
        for item in members:
            assets = {band: asset_of(item, band) for band in BANDS + ("SCL",)}
            if any(a is None for a in assets.values()):
                log.info("%s: bands missing, left out", item.get("id"))
                continue
            cloud = (item.get("properties") or {}).get("eo:cloud_cover")
            parts.append({"id": str(item.get("id", "")), "tile": tile_of(item), "epsg": epsg_of(item),
                          "cloud": None if cloud is None else float(cloud), "assets": assets})
        if not parts:
            continue
        first = min(members, key=_when)
        when = _when(first)
        clouds = [p["cloud"] for p in parts if p["cloud"] is not None]
        out.append({"id": f"{when:%Y%m%d}_{platform_of(first)}", "date": when.date().isoformat(),
                    "datetime": when.isoformat().replace("+00:00", "Z"), "platform": platform_of(first),
                    "orbit": orbit_of(first), "datatake": take, "cloud": max(clouds) if clouds else None,
                    "tiles": sorted({p["tile"] for p in parts}), "items": parts})
    out.sort(key=lambda a: (a["datetime"], a["id"]))
    seen: dict[str, int] = {}
    for a in out:  # one satellite twice on a day (far north): the orbit tells them apart
        if a["id"] in seen:
            a["id"] = f"{a['id']}_R{a['orbit'] or seen[a['id']] + 1:03d}"
        seen[a["id"]] = seen.get(a["id"], 0) + 1
    return out
