"""The outputs of a campaign in the window (the *Tavola*, *Tabelle*, *Disegni*, *QGIS*, *IFC 4.3*, *PyArchInit* and
*Cartella* buttons, the exporters of *File > Esporta*, the plan's QGIS project): the plugin's handlers live here, as
its forms in :mod:`.forms`; the files are written by :mod:`.outputs`, :mod:`.ifc43` and :mod:`.pyarchinit`. Each
function takes the plugin."""
from __future__ import annotations

from pathlib import Path

from .texts import t


def ifc_done(plugin, out: dict) -> None:
    from .texts import plural

    walls = plural(out["hosts"], ("parete", "pareti"), ("wall", "walls"))
    feats = plural(out["features"], ("elemento", "elementi"), ("feature", "features"))
    lines = [str(out["path"]), t(f"{walls} con {feats}", f"{walls} with {feats}")]
    lines += out.get("warnings") or []
    plugin._message("information", t("IFC 4.3:", "IFC 4.3:") + "\n" + "\n".join(lines))


def do_ifc(plugin) -> None:
    """*IFC 4.3*: the campaigns of the elevation chosen, the thickness of the stand-in wall, the file."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import ifc43, records
    from .outputs import _context, _stem

    chunk, entry, c = plugin._need_campaign()
    if entry.get("kind") == "section":
        raise records.NotReady(t(f"IFC: {ifc43.SECTION[0]}: i suoi elementi vanno nei GeoPackage e nel DXF.",
                                 f"IFC: {ifc43.SECTION[1]}: its features go to the GeoPackages and the DXF."))
    ifc43.need_ifcopenshell()
    found = records.campaigns(chunk, entry["id"])
    default = str(records.campaign_folder(chunk, entry["id"], c["id"]) / f"{_stem(_context(chunk, c['id']))}.ifc")
    v = plugin._ask(t("IFC 4.3 della campagna", "IFC 4.3 of the campaign"), [
        Field("campaigns", t("Campagne", "Campaigns"), [c["id"]],
              choices=[(x["id"], f"{x['name']} ({x['date']})") for x in found], kind="checklist"),
        Field("thickness", t("Spessore della parete di appoggio (m)", "Thickness of the host wall (m)"), 0.5,
              minimum=0.01, maximum=10.0, decimals=2,
              tip=t("Senza un muro del plugin murature, gli elementi aderiscono a una lastra di questo spessore "
                    "dietro la faccia.", "Without a wall of the masonry plugin, the features adhere to a slab "
                    "this thick behind the face.")),
        Field("path", t("File", "File"), default, kind="file_save", filter="IFC (*.ifc)")],
        intro=t("Gli elementi della campagna come IfcSurfaceFeature (IFC 4.3) aderenti al muro; niente "
                "ortofoto né texture.", "The campaign's features as IfcSurfaceFeature (IFC 4.3) adhering to "
                "the wall; no orthophoto or texture."))
    if v is None:
        return
    cids = [x for x in (v["campaigns"] or []) if x] or [c["id"]]
    path = Path(v["path"] or default)
    if path.suffix.lower() != ".ifc":
        path = path.with_suffix(".ifc")
    thickness = float(v["thickness"] or 0.5)
    doc = getattr(chunk, "document", None)
    site = (doc.meta.get("sito") if doc is not None else None) or None
    plugin.context.run(t("Conservazione: IFC 4.3", "Conservation: IFC 4.3"),
                       lambda _p: ifc43.export(chunk, cids, path, thickness, site=site),
                       lambda out: ifc_done(plugin, out))


def output_folder(plugin, chunk, entry: dict, c: dict, title: str):
    """The folder of an output: the campaign's by default, or the one the user chooses; None when cancelled."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from .records import campaign_folder

    default = str(campaign_folder(chunk, entry["id"], c["id"]))
    v = plugin._ask(title, [Field("folder", t("Cartella", "Folder"), default, kind="dir")])
    if v is None:
        return None
    return Path(v["folder"] or default)


def done_files(plugin, title: str, paths, note: str = "") -> None:
    lines = [str(p) for p in paths if p]
    if note:
        lines.append(note)
    plugin._message("information", title + "\n" + "\n".join(lines))


PLATE_FRAMES = (("features", ("l'area degli elementi accettati", "the area of the accepted features")),
                ("drawn", ("l'area scelta (da disegnare sulla vista)", "the chosen area (to draw on the view)")),
                ("face", ("tutta la faccia", "the whole face")))


def do_plate(plugin) -> None:
    """*Tavola*: the scale, the frame (the features' extent with half a metre around, a window drawn on the view, or
    the whole face) and the folder; the plate in the background."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import outputs, records

    chunk, entry, c = plugin._need_campaign()
    scales = [t("automatica", "automatic")] + [f"1:{n}" for n in outputs.SCALES]
    default = str(records.campaign_folder(chunk, entry["id"], c["id"]))
    v = plugin._ask(t("Tavola della campagna", "Plate of the campaign"),
                    [Field("scale", t("Scala", "Scale"), scales[0], choices=scales),
                     Field("frame", t("Cosa inquadrare", "What to frame"), PLATE_FRAMES[0][0],
                           choices=[(k, t(*words)) for k, words in PLATE_FRAMES]),
                     Field("folder", t("Cartella", "Folder"), default, kind="dir")])
    if v is None:
        return
    scale = None if v["scale"] == scales[0] else int(str(v["scale"]).split(":")[1])
    folder = Path(v["folder"] or default)
    stem = f"{outputs._stem(outputs._context(chunk, c['id']))}_{t('tavola', 'plate')}"

    def go(bounds):
        plugin.context.run(t("Conservazione: tavola", "Conservation: plate"),
                           lambda _p: outputs.plate(chunk, c["id"], folder, scale, stem=stem, bounds=bounds),
                           lambda out: done_files(plugin, t(f"Tavola in scala 1:{out['scale']}:",
                                                            f"Plate at 1:{out['scale']}:"),
                                                  [out.get(k) for k in ("pdf", "svg", "png")],
                                                  outputs.unnamed_text(chunk, c["id"])))

    frame = v.get("frame") or "features"
    if frame == "drawn":
        plugin._request("polygon", entry, lambda xy: go(plugin._geometry("polygon", xy).bounds),
                        t("Area della tavola: clic per i vertici (o trascina un riquadro), doppio clic o Invio per "
                          "chiudere; Esc annulla", "Area of the plate: click the vertices (or drag a box), double "
                          "click or Enter to close; Esc cancels"))
        return
    go("face" if frame == "face" else None)


def do_tables(plugin) -> None:
    from . import outputs

    chunk, entry, c = plugin._need_campaign()
    folder = output_folder(plugin, chunk, entry, c, t("Tabelle della campagna", "Tables of the campaign"))
    if folder is None:
        return
    stem = f"{outputs._stem(outputs._context(chunk, c['id']))}_{t('tabelle', 'tables')}"
    plugin.context.run(t("Conservazione: tabelle", "Conservation: tables"),
                       lambda _p: outputs.tables(chunk, c["id"], folder, stem),
                       lambda out: done_files(plugin, t("Tabelle:", "Tables:"), out.values(),
                                              outputs.unnamed_text(chunk, c["id"])))


def do_drawings(plugin) -> None:
    """*Disegni*: DXF, GeoPackages and QGIS styles in the folder chosen; the folder kept in ``campaign["exports"]``
    (the plan's QGIS project reads the 3D tables there) as one entry of the undo."""
    from . import outputs

    chunk, entry, c = plugin._need_campaign()
    folder = output_folder(plugin, chunk, entry, c, t("Disegni della campagna", "Drawings of the campaign"))
    if folder is None:
        return
    stem = outputs._stem(outputs._context(chunk, c["id"]))

    def work(progress):
        progress(0.1, "DXF")
        paths = [outputs.dxf(chunk, c["id"], folder / f"{stem}.dxf")]
        progress(0.5, "GeoPackage")
        made = outputs.geopackages(chunk, c["id"], folder, f"{stem}_conservazione")
        rows = outputs.feature_rows(chunk, c["id"])
        codes = {k: list(dict.fromkeys(v["code"] for _g, v in rows[k])) for k in outputs.TABLES}
        styles = outputs.qgis_styles(folder, codes, outputs._context(chunk, c["id"])["legend"], chunk=chunk)
        return paths + list(made.values()) + list(styles.values())

    def done(paths):
        before = plugin._snapshot(chunk)
        c.setdefault("exports", {})["drawings"] = str(folder)  # the plan's QGIS project reads them there
        plugin._changed(chunk, before, t("cartella dei disegni", "folder of the drawings"))
        done_files(plugin, t("Disegni (DXF, GeoPackage spiegato e 3D, stili QGIS):",
                             "Drawings (DXF, unfolded and 3D GeoPackages, QGIS styles):"), paths,
                   outputs.unnamed_text(chunk, c["id"]))

    plugin.context.run(t("Conservazione: disegni", "Conservation: drawings"), work, done)


def do_qgis(plugin) -> None:
    from pyarchinit_photogrammetry.io import qgis_link

    from . import outputs
    from .records import NotReady, campaign_folder

    chunk, entry, c = plugin._need_campaign()
    if qgis_link.find_qgis() is None:
        raise NotReady(t("QGIS non trovato: installalo da qgis.org (o indica il percorso con PPG_QGIS).",
                         "QGIS not found: install it from qgis.org (or give its path with PPG_QGIS)."))
    plugin.context.store_drawings()  # the program's drawings of the elevation go into the project too
    folder = campaign_folder(chunk, entry["id"], c["id"]) / "qgis"
    plugin.context.run(t("Conservazione: apertura in QGIS", "Conservation: opening in QGIS"),
                       lambda _p: outputs.qgis_project(chunk, c["id"], folder),
                       lambda project: plugin._message("information", t(
                           f"Progetto QGIS del prospetto con la campagna: {project} (coordinate del prospetto, in "
                           "metri; i retini della legenda come stili dei layer).", f"QGIS project of the elevation "
                           f"with the campaign: {project} (the elevation's coordinates, metres; the legend's "
                           "hatches as the layers' styles).")))


def do_pyarchinit(plugin) -> None:
    """*Salva in PyArchInit*: the only step that writes into the database (:func:`.pyarchinit.save`, in the
    background; the media ids it gives back written into the campaign here, in the interface thread)."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import assess, pyarchinit, records

    chunk, entry, c = plugin._need_campaign()
    window = plugin.context.window
    pa = getattr(window, "pa", None)
    if pa is None:
        raise records.NotReady(t("Collegati prima a PyArchInit (pannello PyArchInit).",
                                 "Connect to PyArchInit first (PyArchInit pane)."))
    site = ""
    if hasattr(window, "project_site"):
        try:
            site = window.project_site() or ""
        except Exception:  # noqa: BLE001 - no site chosen yet
            site = ""
    found = pyarchinit.database_words(pa)
    words = [assess.condition_word(n, "it", thesaurus=found) for n in range(4)]
    links = pyarchinit._links([f for f in records.features(chunk, c["id"], status="accepted")])
    cc = (c.get("classes") or {}).get("cc")
    units = ", ".join(f"{tipo} {us}" for _a, us, tipo in links) or t("nessuna", "none")
    ccn = "-" if cc is None else cc
    intro = t(f"La tavola (SVG, PDF) e il DXF diventano media di un documento «Prospetto»; le US collegate "
              f"({units}) ricevono lo stato di conservazione della CC {ccn} e i termini nelle osservazioni.",
              f"The plate (SVG, PDF) and the DXF become media of a “Prospetto” document; the linked units "
              f"({units}) get the condition of CC {ccn} and the terms in their observations.")
    fields = [Field("sito", t("Sito", "Site"), site, kind="text")]
    fields += [Field(f"w{n}", t(f"Parola per CC {n}", f"Word for CC {n}"), words[n], kind="text")
               for n in range(4)]
    v = plugin._ask(t("Salva in PyArchInit", "Save in PyArchInit"), fields, intro=intro)
    if v is None:
        return
    chosen = [str(v[f"w{n}"]).strip() or words[n] for n in range(4)]
    before = plugin._snapshot(chunk)
    plugin.context.store_drawings()

    def done(out):
        c["media"] = dict(out.get("campaign_media") or {})
        plugin._changed(chunk, before, "PyArchInit")
        lines = [t(f"Media registrati: {', '.join(map(str, out['media'])) or '-'}",
                   f"Media registered: {', '.join(map(str, out['media'])) or '-'}")]
        lines += [t(f"{tipo} {us}: aggiornata", f"{tipo} {us}: updated") for _a, us, tipo, _i in out["us"]]
        lines += [t(f"{tipo} {us}: scheda non trovata", f"{tipo} {us}: no form found")
                  for _a, us, tipo in out["missing"]]
        plugin._message("information", "\n".join(lines))

    plugin.context.run(t("Conservazione: PyArchInit", "Conservation: PyArchInit"),
                       lambda _p: pyarchinit.save(pa, chunk, c["id"], sito=v["sito"] or None, words=chosen,
                                                  keep=False),
                       lambda out: plugin._guarded(done, out))


def do_folder(plugin) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    from .records import campaign_folder

    chunk, entry, c = plugin._need_campaign()
    folder = campaign_folder(chunk, entry["id"], c["id"])
    folder.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def qgis_entries(plugin):
    """For the QGIS project of the plan: the shown campaign's tables in 3D in the chunk's system (the GeoPackage
    the campaign's *Disegni* wrote last - in the folder the user chose, kept in ``campaign["exports"]`` - else the
    one in the campaign's folder)."""
    from . import outputs, records

    chunk = plugin._chunk()
    cid = plugin.pane.campaign_id() if plugin.pane is not None else None
    if chunk is None or cid is None:
        return [], []
    try:
        ctx = outputs._context(chunk, cid)
    except records.NotReady:
        return [], []
    name = f"{outputs._stem(ctx)}_conservazione_mondo.gpkg"
    folders = [(ctx["campaign"].get("exports") or {}).get("drawings"),
               records.campaign_folder(chunk, ctx["entry"]["id"], cid)]
    path = next((Path(f) / name for f in folders if f and (Path(f) / name).exists()), None)
    if path is None:
        return [], []
    vectors = []
    for table, (gtype, _style, _kinds) in outputs.TABLES.items():
        first = next((f["code"] for f in ctx["features"] if f["kind"] in outputs.TABLES[table][2]), None)
        colour = (ctx["legend"].get(first) or {}).get("colour") if first else None
        vectors.append({"name": f"{t(*outputs.TABLE_TITLES[table])} - {ctx['campaign']['name']} (3D)",
                        "path": str(path), "table": table, "geometry": gtype, "color": colour or "#808080",
                        "label": ""})
    return [], vectors


def export_plate(plugin, chunk, path, progress=None):
    """*File > Esporta > Tavola della campagna (PDF)*: the campaign chosen in the pane, else the chunk's latest."""
    from . import outputs, records

    cid = plugin.pane.campaign_id() if plugin.pane is not None else None
    if cid is None or cid not in {c["id"] for c in records.campaigns(chunk)}:
        found = records.campaigns(chunk)
        if not found:
            raise records.NotReady(t("Nessuna campagna di conservazione in questo chunk.",
                                     "No conservation campaign in this chunk."))
        cid = found[-1]["id"]
    path = Path(path)
    out = outputs.plate(chunk, cid, path.parent, stem=path.stem, formats=("pdf",))
    return out.get("pdf") or out.get("png")


def export_ifc(plugin, chunk, path, progress=None):
    """*File > Esporta > Conservazione (IFC 4.3)*: the campaigns of the elevation chosen in the pane, else every
    campaign of the chunk on a face (sections left out), the stand-in walls 0.5 m thick."""
    from . import ifc43, records

    eid = plugin.pane.elevation_id() if plugin.pane is not None else None
    found = records.campaigns(chunk, eid) if eid else []
    if not found:
        faces = {e["id"] for e in plugin._elevations(chunk) if e.get("kind") != "section"}
        found = [c for c in records.campaigns(chunk) if c["elevation"] in faces] or records.campaigns(chunk)
    if not found:
        raise records.NotReady(t("Nessuna campagna di conservazione in questo chunk.",
                                 "No conservation campaign in this chunk."))
    return ifc43.export(chunk, [c["id"] for c in found], path)["path"]


__all__ = ["ifc_done", "do_ifc", "output_folder", "done_files", "do_plate", "do_tables", "do_drawings", "do_qgis",
           "do_pyarchinit", "do_folder", "qgis_entries", "export_plate", "export_ifc", "PLATE_FRAMES"]
