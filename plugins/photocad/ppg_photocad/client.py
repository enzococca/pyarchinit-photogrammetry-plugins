"""A small client of the PhotoCAD API (https://photocad.ai/api) on the standard library.

* ``POST /convert`` (multipart ``image_file``, ``detailed``, ``remove_background``) -> 202 ``{id, status}``;
* ``GET /conversions/{id}`` -> ``status`` processing | complete | failed (``error``), ``credits_remaining``,
  ``width_mm``, ``height_mm``, ``files``;
* ``GET /conversions/{id}/files/{format}`` with ``units``, ``longest``, ``margin``, ``dxf_version``...;
* ``GET /account`` (the credits), ``DELETE /conversions/{id}``.

The key goes in ``Authorization: Bearer``; it never appears in an error (:func:`keys.mask`). Errors come
as ``{"errors": [{code, title, detail}]}``: :class:`PhotoCADError` says them in plain words; 429 is asked
again after ``Retry-After`` (3 more times at most). ``PPG_PHOTOCAD_URL`` points the client at another
server (a test server of the user's own). One conversion costs one credit.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .keys import mask
from .texts import t

log = logging.getLogger("pyarchinit_photogrammetry.plugins.photocad")

API = "https://photocad.ai/api/v1"
URL_ENV = "PPG_PHOTOCAD_URL"
TIMEOUT = 120.0
RETRIES = 3  # new attempts after a 429 (or a busy server)
POLL = 5.0  # seconds between two questions on a conversion
WAIT_LIMIT = 900.0  # a conversion not done after 15 minutes is given up (and deleted)
RETRY_STATUS = (429, 502, 503, 504)


class PhotoCADError(RuntimeError):
    """PhotoCAD refused or failed: ``status`` (HTTP or 0), ``code`` (its slug), the text in plain words."""

    def __init__(self, text: str, status: int = 0, code: str = ""):
        super().__init__(text)
        self.status = status
        self.code = code


def _ssl_context():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _urlopen(request, timeout: float):
    """The network. The tests replace this function (and :func:`_sleep`)."""
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def base_url() -> str:
    return (os.environ.get(URL_ENV) or API).rstrip("/")


def user_agent() -> str:
    try:
        from pyarchinit_photogrammetry import __version__ as app
    except Exception:  # noqa: BLE001
        app = "?"
    from . import __version__

    return f"pyarchinit-photogrammetry/{app} ppg-photocad/{__version__}"


def _plain(status: int, slug: str, detail: str) -> str:
    """What a refusal means, in the language of the interface."""
    detail = f" ({detail})" if detail else ""
    if status == 401:
        return t("PhotoCAD ha rifiutato la chiave API: controllala in Plugin > PhotoCAD > Chiave API.",
                 "PhotoCAD refused the API key: check it in Plugin > PhotoCAD > API key.") + detail
    if status == 402:
        return t("Credito PhotoCAD esaurito: ricarica su photocad.ai.",
                 "PhotoCAD credit used up: top it up on photocad.ai.") + detail
    if status == 413:
        return t("Immagine troppo grande per PhotoCAD.", "The image is too large for PhotoCAD.") + detail
    if status == 422:
        return t("Immagine non accettata da PhotoCAD.", "The image was not accepted by PhotoCAD.") + detail
    if status == 429:
        return t("Troppe richieste a PhotoCAD (20 conversioni al minuto): riprova fra poco.",
                 "Too many requests to PhotoCAD (20 conversions a minute): try again shortly.") + detail
    if status == 404:
        return t("Conversione non trovata su PhotoCAD.", "Conversion not found on PhotoCAD.") + detail
    return t("PhotoCAD ha risposto con un errore {0}", "PhotoCAD answered with an error {0}").format(
        f"HTTP {status} {slug}".strip()) + detail


def _refusal(exc: urllib.error.HTTPError, key: str | None) -> PhotoCADError:
    slug, detail = "", ""
    try:
        data = json.loads(exc.read().decode("utf-8", "replace"))
        first = (data.get("errors") or [{}])[0]
        slug = str(first.get("code") or "")
        detail = str(first.get("detail") or first.get("title") or "")
    except Exception:  # noqa: BLE001 - an answer that is not the JSON of the API
        pass
    detail = " ".join(mask(detail, key).split())[:300]
    return PhotoCADError(mask(_plain(exc.code, slug, detail), key), exc.code, slug)


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "ppg-" + uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, ctype) in files.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                   f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


class Client:
    """The API with ``key`` (never kept anywhere but in this object)."""

    def __init__(self, key: str, base: str | None = None, timeout: float = TIMEOUT):
        if not key:
            raise PhotoCADError(t("Manca la chiave API PhotoCAD.", "The PhotoCAD API key is missing."), 401)
        self._key = key
        self.base = (base or base_url()).rstrip("/")
        self.timeout = timeout

    def __repr__(self) -> str:  # never the key
        return f"Client({self.base!r})"

    def _request(self, method: str, path: str, query: dict | None = None, data: bytes | None = None,
                 ctype: str | None = None, raw: bool = False):
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": user_agent(),
                   "Accept": "*/*" if raw else "application/json"}
        if ctype:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        for attempt in range(RETRIES + 1):
            try:
                with _urlopen(req, self.timeout) as answer:
                    body = answer.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in RETRY_STATUS and attempt < RETRIES:
                    try:
                        wait = float(exc.headers.get("Retry-After") or 0) if exc.headers is not None else 0.0
                    except ValueError:
                        wait = 0.0
                    wait = min(max(wait, 1.0 if exc.code == 429 else 2.0 ** attempt), 120.0)
                    log.info("PhotoCAD %s %s: HTTP %s, again in %.0f s", method, path, exc.code, wait)
                    _sleep(wait)
                    continue
                raise _refusal(exc, self._key) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                reason = mask(str(getattr(exc, "reason", exc)), self._key)
                raise PhotoCADError(t("PhotoCAD non raggiungibile: {0}", "PhotoCAD cannot be reached: {0}")
                                    .format(reason)) from None
        if raw:
            return body
        if not body.strip():
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise PhotoCADError(t("Risposta di PhotoCAD non leggibile.", "PhotoCAD's answer cannot be read.")) from None

    # ------------------------------------------------------------------ the endpoints
    def account(self) -> dict:
        return self._request("GET", "/account")

    def convert(self, data: bytes, filename: str, content_type: str, detailed: bool = True,
                remove_background: bool = False) -> str:
        """Upload the image: the id of the conversion (one credit)."""
        body, ctype = _multipart({"detailed": "true" if detailed else "false",
                                  "remove_background": "true" if remove_background else "false"},
                                 {"image_file": (filename, data, content_type)})
        answer = self._request("POST", "/convert", data=body, ctype=ctype)
        cid = str(answer.get("id") or (answer.get("data") or {}).get("id") or "")
        if not cid:
            raise PhotoCADError(t("PhotoCAD non ha dato l'id della conversione.", "PhotoCAD gave no conversion id."))
        return cid

    def conversion(self, cid: str) -> dict:
        return self._request("GET", f"/conversions/{urllib.parse.quote(cid, safe='')}")

    def file(self, cid: str, fmt: str, **params) -> bytes:
        query = {k: (f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else v)
                 for k, v in params.items() if v is not None}
        return self._request("GET", f"/conversions/{urllib.parse.quote(cid, safe='')}/files/{fmt}", query, raw=True)

    def delete(self, cid: str) -> None:
        self._request("DELETE", f"/conversions/{urllib.parse.quote(cid, safe='')}")


def credits(answer: dict) -> int | None:
    """The credits left in an answer of ``/account`` or of a conversion (the shape of ``/account`` is not
    documented in detail: the usual names are looked for, also under ``data``); None when not found."""
    if not isinstance(answer, dict):
        return None
    for source in (answer, answer.get("data") if isinstance(answer.get("data"), dict) else {}):
        for name in ("credits_remaining", "credits", "balance", "credit_balance", "remaining_credits"):
            value = source.get(name)
            if isinstance(value, dict):
                value = value.get("total", value.get("remaining"))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    return None


def _check(progress) -> None:
    if progress is None:
        return
    check = getattr(progress, "check", None)
    if check is not None:
        check()


def wait(client: Client, cid: str, progress=None, interval: float = POLL, limit: float = WAIT_LIMIT) -> dict:
    """Ask on the conversion every ``interval`` seconds until it is complete; its answer. ``progress(fraction,
    message)`` (it raises ``Cancelled`` when the user cancels: the conversion is then deleted on the server
    and the cancellation goes on). A failed conversion (refunded by PhotoCAD) raises :class:`PhotoCADError`."""
    waited = 0.0
    try:
        while True:
            _check(progress)
            answer = client.conversion(cid)
            status = str(answer.get("status") or "")
            if status == "complete":
                return answer
            if status == "failed":
                err = answer.get("error") or {}
                detail = err.get("detail") or err.get("title") or err.get("code") or "" if isinstance(err, dict) \
                    else str(err)
                raise PhotoCADError(t("PhotoCAD non è riuscito a vettorializzare l'immagine: {0}. Il credito è stato "
                                      "rimborsato.", "PhotoCAD could not vectorise the image: {0}. The credit was "
                                                     "refunded.").format(detail or "?"), 0, "failed")
            if waited >= limit:
                raise PhotoCADError(t("PhotoCAD non ha finito in {0:.0f} minuti.", "PhotoCAD did not finish in {0:.0f} "
                                      "minutes.").format(limit / 60), 0, "timeout")
            if progress is not None:
                progress(min(0.9, 0.1 + 0.8 * waited / 120.0),
                         t("PhotoCAD sta vettorializzando ({0:.0f} s)", "PhotoCAD is vectorising ({0:.0f} s)")
                         .format(waited))
            step = 0.0
            while step < interval:  # short naps: a cancel is seen within half a second
                _sleep(min(0.5, interval - step) if interval > 0.5 else interval)
                step += min(0.5, interval) if interval > 0.5 else interval
                _check(progress)
            waited += interval
    except BaseException as exc:
        if type(exc).__name__ == "Cancelled" or isinstance(exc, PhotoCADError) and exc.code == "timeout":
            try:
                client.delete(cid)
                log.info("PhotoCAD conversion %s deleted", cid)
            except PhotoCADError as err:
                log.warning("PhotoCAD conversion %s not deleted: %s", cid, err)
        raise
