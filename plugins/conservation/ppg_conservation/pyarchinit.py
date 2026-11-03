"""A campaign into PyArchInit, only when the user asks (*Salva in PyArchInit*: :func:`save` is the plugin's only writer
into the database, called by the pane's handler and by ``ppg plugin conservation pyarchinit``).

* the plate (SVG, with its PNG as the thumbnail) and the DXF of the campaign registered as media of a "Prospetto"
  document named after the elevation and the campaign (the program's ``register_elevation_media``; the ids are kept
  in the campaign's record, so a second save writes the same media again and the program's own media of the
  elevation are never touched), the PDF beside them when the bridge is the program's; the media linked to the US
  forms the features are linked to (``links.us``);
* the US forms linked: ``stato_di_conservazione`` = the word of the campaign's condition class (buono / discreto /
  mediocre / cattivo, the database's thesaurus words when it has four of them, or the words the user chose) and a
  line ``[conservazione <date>] <terms with their extents>`` in ``osservazioni`` (replaced, not repeated, when the
  campaign is saved again)."""
from __future__ import annotations

from pathlib import Path

from . import records
from .texts import num, t

FIELD_LENGTH = 20  # us_table.stato_di_conservazione is a VARCHAR(20) in PyArchInit 5
TAG = "conservazione"


def words_of(words=None) -> list[str] | None:
    """The four words of CC 0-3 the user chose (ValueError unless four), None when not given."""
    if words:
        words = [str(w) for w in words]
        if len(words) != 4:
            raise ValueError(t("Servono quattro parole, una per CC 0-3.", "Four words are needed, one per CC 0-3."))
        return words
    return None


def database_words(pa) -> list:
    """The values of the database's thesaurus for ``stato_di_conservazione`` of the US form (empty when none)."""
    try:
        return list(pa.thesaurus("stato_di_conservazione", table="us_table") or [])
    except Exception:  # noqa: BLE001 - an older database without the thesaurus
        return []


def condition_value(pa, cc, words=None, lang: str = "it") -> str | None:
    """What goes into ``stato_di_conservazione`` for the class ``cc`` (None: no class chosen)."""
    from .assess import condition_word

    if cc is None or cc == "":
        return None
    given = words_of(words)
    word = given[int(cc)] if given else condition_word(cc, lang, thesaurus=database_words(pa))
    return str(word)[:FIELD_LENGTH]


def _links(feats) -> list[tuple]:
    found = []
    for f in feats:
        for link in (f.get("links") or {}).get("us") or []:
            area, us = str(link[0]), str(link[1])
            tipo = str(link[2]) if len(link) > 2 and link[2] else "US"
            if (area, us, tipo) not in found:
                found.append((area, us, tipo))
    return found


def observation(vocab, c: dict, feats) -> str:
    """The line of the campaign for ``osservazioni``: the terms of the features with their extents."""
    parts = {}
    for f in feats:
        name = f["code"]
        if f["code"] in vocab:  # the UNI 11182 name of a main term, the plugin's Italian one of a sub-type
            term = vocab.term(f["code"])
            uni = vocab.term(term.uni) if term.uni and term.uni in vocab and term.parent is None else None
            name = (uni or term).names["it"]
        key = (name, f["kind"])
        parts.setdefault(key, [0.0, None])
        parts[key][0] += float(f.get("extent") or 0.0)
        if f.get("width_mm") is not None:
            parts[key][1] = max(parts[key][1] or 0.0, float(f["width_mm"]))
    out = []
    for (name, kind), (extent, width) in parts.items():
        if kind == "note":
            out.append(name)
        elif kind == "crack":
            out.append(f"{name} {num(extent, 2)} m" + (f" (apertura {num(width, 1)} mm)" if width else ""))
        else:
            out.append(f"{name} {num(extent, 3)} m²" if extent else name)
    return f"[{TAG} {c['date']}] " + "; ".join(out)


def _with_line(existing: str | None, line: str, date: str) -> str:
    """``existing`` with the campaign's line: the line of that date replaced, else added at the end."""
    text = (existing or "").rstrip("\n")
    head = f"[{TAG} {date}]"
    kept = [x for x in text.split("\n") if not x.startswith(head)] if text else []
    kept = [x for x in kept if x or kept]
    return "\n".join([*kept, line]) if kept else line


def _register_pdf(pa, chunk, entry: dict, c: dict, pdf: Path, sito: str, link_us, preview) -> int | None:
    """The plate's PDF as a media of the site (the program's bridge only: its database and media folders)."""
    if getattr(pa, "db", None) is None or not hasattr(pa, "_links"):
        return None
    from pyarchinit_photogrammetry.pyarchinit.media import register_media

    dirs = pa._media_dirs() if hasattr(pa, "_media_dirs") else {}
    return register_media(pa.db, pdf, mediatype="image",
                          descrizione=f"Prospetto: {entry.get('name') or entry['id']} - {TAG} {c['date']} (PDF)",
                          tags=f"['fotogrammetria', 'prospetto', '{TAG}']", links=pa._links(sito, link_us),
                          copy_to=pa._media_folder(sito, "prospetti"), overwrite=True, preview=preview, **dirs)


def save(pa, chunk, cid: str, sito: str | None = None, words=None, folder=None) -> dict:
    """The campaign ``cid`` into PyArchInit through the bridge ``pa`` (see the module). ``sito``: the project's
    site by default (NotReady without one); ``words``: the four words of CC 0-3; ``folder``: where the files sent go
    (default ``<campaign folder>/pyarchinit``). Returns ``{"media" (ids), "us" [(area, us, tipo, id_us)], "missing"
    [(area, us, tipo)] (forms not in the database), "word", "unnamed"}``."""
    from pyarchinit_photogrammetry.engine.elevation import _safe_stem

    from . import outputs
    from .vocabulary import UNNAMED, load

    c = records.campaign(chunk, cid)
    entry = records._elevation(chunk, c["elevation"])
    if not sito:
        doc = getattr(chunk, "document", None)
        sito = str(((getattr(doc, "meta", None) or {}).get("sito")) or "") if doc is not None else ""
    if not sito:
        raise records.NotReady(t("Il sito: sceglilo nel pannello PyArchInit (sito del progetto).",
                                 "The site: choose it in the PyArchInit pane (site of the project)."))
    folder = Path(folder) if folder else records.campaign_folder(chunk, entry["id"], cid) / "pyarchinit"
    stem = _safe_stem(f"{entry.get('name') or entry['id']}_{entry['id']}_{cid}_{TAG}")
    made = outputs.plate(chunk, cid, folder, stem=f"{stem}_tavola", formats=("svg", "pdf", "png"))
    dxf = outputs.dxf(chunk, cid, folder / f"{stem}.dxf")
    feats = [f for f in records.features(chunk, cid, status="accepted") if f["code"] != UNNAMED]
    link_us = _links(feats)
    paths = {"dxf": dxf}
    for key in ("svg", "png"):
        if made.get(key) is not None:
            paths[key] = made[key]
    shadow = dict(entry, name=f"{entry.get('name') or entry['id']} - {TAG} {c['date']}",
                  media=dict(c.get("media") or {}))  # the program's media of the elevation stay its own
    ids = list(pa.register_elevation_media(chunk, shadow, paths, sito, [list(x) for x in link_us]) or [])
    media = dict(shadow.get("media") or {})
    if made.get("pdf") is not None:
        preview = None
        if made.get("png") is not None:
            from PIL import Image

            with Image.open(made["png"]) as im:
                preview = im.convert("RGB")
        pid = _register_pdf(pa, chunk, entry, c, made["pdf"], sito, link_us, preview)
        if pid is not None:
            media["pdf"] = pid
            ids.append(pid)
    c["media"] = media
    vocab = load(chunk)
    word = condition_value(pa, (c.get("classes") or {}).get("cc"), words)
    updated, missing = [], []
    for area, us, tipo in link_us:
        rec = pa.us_record(sito, area, us, tipo)
        if rec is None:
            missing.append((area, us, tipo))
            continue
        mine = [f for f in feats if any([str(x) for x in link[:2]] == [area, us]
                                        for link in (f.get("links") or {}).get("us") or [])]
        fields = {"osservazioni": _with_line(rec.get("osservazioni"), observation(vocab, c, mine), c["date"])}
        if word is not None:
            fields["stato_di_conservazione"] = word
        pa.update_us(rec["id_us"], **fields)
        updated.append((area, us, tipo, rec["id_us"]))
    return {"media": ids, "us": updated, "missing": missing, "word": word,
            "unnamed": outputs.unnamed_count(chunk, cid)}


__all__ = ["save", "condition_value", "database_words", "observation", "FIELD_LENGTH"]

