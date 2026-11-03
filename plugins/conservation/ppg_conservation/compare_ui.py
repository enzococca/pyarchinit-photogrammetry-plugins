"""The comparison of two campaigns in the window (the *Confronta* button, the overlays of the pane): the plugin's
handlers live here, as its forms in :mod:`.forms`; the engine is :mod:`.compare`. Each function takes the plugin."""
from __future__ import annotations

from pathlib import Path

from .texts import t


def document(plugin):
    window = plugin.context.window
    return getattr(window, "doc", None) or plugin.context.app.document


def chunk_by_key(plugin, key: str):
    from .records import NotReady

    for ch in getattr(document(plugin), "chunks", None) or []:
        if str(ch.key) == key:
            return ch
    raise NotReady(t("Il chunk scelto non è più nel progetto.", "The chunk chosen is no longer in the project."))


def choices(plugin, chunk, entry: dict, c: dict) -> list[tuple[str, str]]:
    """What campaign A can be compared with: ``c:<chunk key>:<campaign id>`` for the other campaigns of the face
    (this elevation, or another chunk's elevation on the same frame), ``b:<chunk key>:`` to build that elevation
    in a chunk with a model that has none."""
    from . import compare, records

    out = [(f"c:{chunk.key}:{x['id']}", t(f"{x['date']}  {x['name']} (stesso prospetto)",
                                         f"{x['date']}  {x['name']} (same elevation)"))
           for x in records.campaigns(chunk, entry["id"]) if x["id"] != c["id"]]
    for other in getattr(document(plugin), "chunks", None) or []:
        if other is chunk:
            continue
        found = compare.find_elevation(other, entry)
        if found is not None:
            out += [(f"c:{other.key}:{x['id']}", f"«{other.label}»: {x['date']}  {x['name']}")
                    for x in records.campaigns(other, found["id"])]
        elif other.product_path("model") is not None:
            out.append((f"b:{other.key}:", t(f"«{other.label}»: costruisci il prospetto sullo stesso piano",
                                             f"«{other.label}»: build the elevation on the same plane")))
    return out


def do_compare(plugin) -> None:
    """*Confronta*: campaign A (the one chosen) against another campaign of the same face - of this elevation, or
    of another chunk's elevation on the same frame (built here when the chunk has none) - after the check that
    the two chunks are aligned; the tables, the PDF page and the overlays."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import compare, records

    chunk, entry, c = plugin._need_campaign()
    options = choices(plugin, chunk, entry, c)
    if not options:
        raise records.NotReady(t(
            "Nessun'altra campagna di questa faccia e nessun altro chunk con un modello: fai una seconda campagna, "
            "o aggiungi il rilievo successivo come chunk allineato (Allinea chunk).",
            "No other campaign of this face and no other chunk with a model: make a second campaign, or add the "
            "later survey as an aligned chunk (Align chunks)."))
    v = plugin._ask(t("Confronta due campagne", "Compare two campaigns"), [
        Field("other", t("Campagna B", "Campaign B"), options[0][0], choices=options),
        Field("folder", t("Cartella (vuota: quella del confronto)", "Folder (empty: the comparison's own)"), "",
              kind="dir")],
        intro=t(f"A: {c['date']}  {c['name']}. Un rilievo in un altro chunk deve essere già allineato a questo "
                f"(Flusso di lavoro > Allinea chunk): il plugin lo verifica con i marker sulla faccia (entro 1 cm) "
                f"o con lo spostamento delle ortofoto (sotto 2 px) e rifiuta altrimenti.",
                f"A: {c['date']}  {c['name']}. A survey in another chunk must already be aligned with this one "
                f"(Workflow > Align chunks): the plugin checks it with the markers on the face (within 1 cm) or "
                f"with the shift of the orthophotos (under 2 px) and refuses otherwise."))
    if v is None:
        return
    kind, key, cid_b = str(v["other"]).split(":", 2)
    other = chunk_by_key(plugin, key)
    if kind == "b":
        def built(found):
            plugin.context.mark_modified()
            plugin.context.refresh_tree()
            plugin._message("information", t(
                f"Prospetto «{found.get('name')}» costruito in «{other.label}» sullo stesso piano: fai lì la "
                f"campagna B (scegli quel chunk), poi Confronta.",
                f"Elevation «{found.get('name')}» built in «{other.label}» on the same plane: make campaign B "
                f"there (choose that chunk), then Compare."))

        plugin.context.run(t("Conservazione: prospetto sullo stesso piano", "Conservation: elevation on the same "
                           "plane"), lambda progress: compare.ensure_elevation(other, entry, progress), built)
        return
    folder = Path(v["folder"]) if v.get("folder") else None

    def work(progress):
        try:
            return compare.compare(chunk, c["id"], cid_b, folder=folder, progress=progress, other=other)
        except records.NotReady as exc:  # the alignment refused, the grids differ: said, nothing written
            return {"refused": str(exc)}

    plugin.context.run(t("Conservazione: confronto", "Conservation: comparison"), work,
                     lambda r: plugin._guarded(compared, plugin, chunk, entry, r))


def compared(plugin, chunk, entry: dict, r: dict) -> None:
    if r.get("refused"):
        plugin._message("warning", r["refused"])
        return
    plugin.comparison = {"chunk": str(chunk.key), "eid": entry["id"], "colour": r["change"],
                       "relief": r.get("relief_change"), "folder": r["folder"]}
    available = ["colour"] + (["relief"] if r.get("relief_change") else [])
    if plugin.pane is not None:
        plugin.pane.set_comparison(available, "colour")
    plugin._show(entry)
    show_overlay(plugin, "colour")
    files = [r[k] for k in ("xlsx", "pdf", "png", "change", "relief_change", "json") if r.get(k)]
    plugin._done_files(t("Confronto:", "Comparison:"), files, "\n".join(r.get("lines") or []))


def comparison_shown(plugin) -> bool:
    chunk = plugin._chunk()
    return bool(plugin.comparison) and chunk is not None and str(chunk.key) == plugin.comparison["chunk"] and \
        plugin._shown_eid() == plugin.comparison["eid"]


def show_overlay(plugin, kind: str) -> None:
    """The raster ``kind`` (``colour``, ``relief``) of the last comparison over its elevation; none for ``""``."""
    from .overlay import Overlay
    from .records import NotReady

    window = plugin.context.window
    view = getattr(window, "ortho_view", None) if window is not None else None
    if view is None:
        return
    if plugin.overlay is None or plugin.overlay.view is not view:
        plugin.overlay = Overlay(view)
    if not kind:
        plugin.overlay.remove()
        return
    if not plugin.comparison or not plugin.comparison.get(kind):
        raise NotReady(t("Nessun confronto con questo strato: Confronta...",
                         "No comparison with this layer: Compare..."))
    if not comparison_shown(plugin):
        chunk = plugin._chunk()
        if chunk is None or str(chunk.key) != plugin.comparison["chunk"]:
            raise NotReady(t("Il confronto è di un altro chunk: torna a quel chunk.",
                             "The comparison is of another chunk: go back to that chunk."))
        plugin._show(plugin._entry(chunk, plugin.comparison["eid"]))  # its raster_loaded shows the overlay
    opacity = plugin.pane.opacity_value() if plugin.pane is not None else 0.6
    plugin.overlay.show(plugin.comparison[kind], f"{plugin.comparison['eid']}:{kind}", kind, opacity=opacity)
