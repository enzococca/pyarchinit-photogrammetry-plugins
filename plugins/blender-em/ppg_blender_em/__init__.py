"""Blender ed EMtools - a plugin of pyarchinit-photogrammetry: the survey and the proxies of the units
in Blender and in the Extended Matrix of EMtools, kept up to date while the work goes on.

* **The matrix is the em.json** that PyArchInit (*Extended Matrix > Esporta sito in em.json*),
  EMStudio or EMtools 1.6 wrote (:mod:`.emjson`): the plugin reads it and never writes it. Its units are paired with the proxies
  of the chunk (*Strumenti > Proxy 3D delle US*) by the ``node_uuid`` of the US form, or by area,
  type and number, or by name.
* **The EM project** (:mod:`.scene`): each proxy into ``EM/proxies/<unit>.glb`` named as its node -
  the name EMtools pairs a proxy object with - and the survey into ``RB/<chunk>/<chunk>.glb``,
  shifted by whole metres, with the manifest ``ppg_scene.json``.
* **Blender, live** (:mod:`.blender`, :mod:`.blender_live`): *Apri in Blender* starts Blender with a
  script that imports the manifest and follows it, as QGIS follows the drawings of the QGIS link;
  with *Aggiornamento dal vivo* every proxy built again and every model changed is written again,
  and Blender reloads it within a second. What was changed in Blender is never overwritten.
* **EMtools, live** (:mod:`.sync`): with *Sincronizza con EMtools* the program joins the live sync
  of EMtools 1.6 (the WebSocket EMStudio uses): a proxy picked in Blender selects its unit in the
  pane, a unit picked in the pane selects and frames its proxy in Blender; *Proxy in Blender* has
  EMtools start the proxy of a unit the program has none of (its command ``create_proxy_for_unit``).
* **Back from Blender** (:mod:`.inbox`): what is modelled in Blender and sent with *Invia al
  programma* comes into the chunk as the proxy of its unit, with its volume, area and elevations.
* **Cameras** (:mod:`.cameras`): the aligned cameras in Blender, each with its photo undistorted
  behind it, to model a reconstruction where the photos show it.

``ppg plugin blender-em export|open|take|status|blender`` from the command line.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import plural, t

__version__ = "0.1.2"
META = "blender_em"  # chunk.meta: the em.json linked and the EM project of the chunk
LIVE_MS = 2000  # how often the live update looks at the chunk


def _number(value, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return ""


def rows_for(chunk, matrix, node_uuids=None) -> list[dict]:
    """The rows of the pane: the units of ``matrix`` with the proxy of the chunk paired to each, then
    the proxies the matrix does not have."""
    from .emjson import match

    entries = list((chunk.meta or {}).get("proxies") or []) if chunk is not None else []
    matches = match(entries, matrix, node_uuids)
    by_unit = {m.unit.id: m for m in matches if m.unit is not None}
    rows = []

    def proxy_text(entry):
        vol = _number(entry.get("volume_m3"))
        return f"{vol} m³" if vol else t("sì", "yes")

    for u in (matrix.units if matrix is not None else []):
        m = by_unit.get(u.id)
        info = [u.description] if u.description else []
        if m is not None:
            e = m.entry
            info.append(t("proxy: {0} m³, {1} m²; tetto {2}-{3} m, fondo {4} m",
                          "proxy: {0} m³, {1} m²; top {2}-{3} m, bottom {4} m").format(
                _number(e.get("volume_m3")), _number(e.get("area_m2"), 2), _number((e.get("z_top") or [None])[0]),
                _number((e.get("z_top") or [None, None])[1]), _number((e.get("z_bottom") or [None])[0])))
        rows.append({"name": u.name, "tipo": u.tipo, "epoch": u.epoch.name if u.epoch else "",
                     "proxy": proxy_text(m.entry) if m is not None else "", "node_id": u.id,
                     "color": u.color, "info": "\n".join(info)})
    for m in matches:
        if m.unit is None:
            rows.append({"name": m.name, "tipo": m.entry.get("tipo") or "", "epoch": "",
                         "proxy": proxy_text(m.entry) + (t(" - non nella matrice", " - not in the matrix")
                                                         if matrix is not None else ""),
                         "node_id": None, "info": t("Candidati: {0}", "Candidates: {0}").format(m.problem)
                         if m.problem else ""})
    return rows


def report_text(report: dict) -> str:
    """What an export did, in a few lines."""
    lines = []

    def line(key, it, en):
        items = report.get(key) or []
        if items:
            lines.append(t(it, en).format(len(items), ", ".join(map(str, items[:12])) + (" ..." if len(items) > 12 else "")))

    line("written", "Scritti {0}: {1}", "Written {0}: {1}")
    line("kept", "Invariati {0}", "Unchanged {0}")
    line("removed", "Tolti {0}: {1}", "Removed {0}: {1}")
    line("foreign", "Lasciati com'erano {0} (file di un altro programma, per esempio EMtools): {1}",
         "Left as they were {0} (files of another program, EMtools for one): {1}")
    line("not_in_matrix", "Non nella matrice {0} (nome del programma): {1}",
         "Not in the matrix {0} (the program's name): {1}")
    line("unnamed", "Nomi che non possono essere file {0}: {1}", "Names that cannot be files {0}: {1}")
    for name, candidates in (report.get("ambiguous") or {}).items():
        lines.append(t("{0}: più unità possibili ({1})", "{0}: several possible units ({1})").format(name, candidates))
    cams = report.get("cameras") or {}
    if cams.get("written") or cams.get("kept"):
        lines.append(t("Camere: {0} foto raddrizzate, {1} invariate", "Cameras: {0} photos undistorted, {1} unchanged").format(
            len(cams.get("written") or []), len(cams.get("kept") or [])))
    if cams.get("missing"):
        lines.append(t("Foto non trovate {0}: {1}", "Photos not found {0}: {1}").format(
            len(cams["missing"]), ", ".join(cams["missing"][:8])))
    shift = report.get("shift")
    if shift:
        lines.append(t("Spostamento delle coordinate: {0}", "Shift of the coordinates: {0}").format(
            ", ".join(f"{v:.0f}" for v in shift)))
    if report.get("manifest"):
        lines.append(t("Manifest: {0} (revisione {1})", "Manifest: {0} (revision {1})").format(
            report["manifest"], report.get("revision")))
    return "\n".join(lines)


class BlenderEmPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Blender ed EMtools", "Blender and EMtools")
        self.description = t("Proxy delle US e rilievo in Blender e nell'Extended Matrix di EMtools (em.json), "
                             "aggiornati dal vivo; selezione condivisa con EMtools",
                             "Unit proxies and the survey in Blender and in EMtools' Extended Matrix (em.json), kept up "
                             "to date; selection shared with EMtools")
        self.context = None
        self.pane = None
        self.client = None
        self.snapshot = None  # the matrix EMtools sent (when no em.json is linked)
        self.host = {}  # what EMtools said of itself
        self._timer = None
        self._live_busy = False
        self._live_signature = None
        self._matrices: dict[str, tuple] = {}  # path -> (mtime, Matrix)
        self._busy_signal = None
        self._commands: dict[str, str] = {}  # cmd_id -> the unit EMtools is asked to make a proxy of

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Pannello Blender ed EMtools", "Blender and EMtools pane"), self.show_pane)
        context.add_action(t("Collega l'em.json del sito...", "Link the em.json of the site..."), self.link_emjson)
        context.add_action(t("Esporta nel progetto EM", "Export into the EM project"), self.export, processing=True,
                           tip=t("I proxy in EM/proxies e il rilievo in RB, con i nomi dei nodi dell'em.json",
                                 "The proxies into EM/proxies and the survey into RB, named after the nodes of the em.json"))
        context.add_action(t("Apri in Blender (aggiornato dal vivo)", "Open in Blender (kept up to date)"),
                           self.open_blender, processing=True)
        context.add_action(t("Importa i proxy da Blender", "Import the proxies from Blender"), self.take_from_blender,
                           processing=True,
                           tip=t("I proxy mandati da Blender con Invia al programma, nel chunk con il loro volume",
                                 "The proxies sent from Blender with Invia al programma, into the chunk with their volume"))
        context.add_action(t("Scegli il programma Blender...", "Choose the Blender program..."), self.choose_blender)
        context.add_command("blender-em", self.command,
                            help="the proxies and the survey into an Extended Matrix project for Blender and EMtools: "
                                 "export, open, take, status, blender")
        if context.window is not None:
            self._install_pane()
            context.on_refresh(self.refresh_pane)
            context.add_tree_items(self._tree_items)
            from PySide6.QtCore import QTimer

            self._timer = QTimer()
            self._timer.setInterval(LIVE_MS)
            self._timer.timeout.connect(self._live_tick)
            if self.context.settings.get("live"):
                self._timer.start()
            if self.context.settings.get("sync"):
                self.set_sync(True)

    def deactivate(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        signal, self._busy_signal = self._busy_signal, None
        if signal is not None:
            try:
                signal.disconnect(self._busy)
            except (RuntimeError, TypeError):
                pass
        self.pane = None

    def _install_pane(self) -> None:
        from .pane import BlenderEmPane

        self.pane = BlenderEmPane()
        self.context.add_pane(self.pane, t("Blender ed EMtools", "Blender and EMtools"), "model", area="right")
        self.pane.link_requested.connect(self.link_emjson)
        self.pane.export_requested.connect(self.export)
        self.pane.open_requested.connect(self.open_blender)
        self.pane.live_toggled.connect(self.set_live)
        self.pane.sync_toggled.connect(self.set_sync)
        self.pane.cameras_toggled.connect(self.set_cameras)
        self.pane.take_requested.connect(self.take_from_blender)
        self.pane.proxy_requested.connect(self.proxies_in_blender)
        self.pane.selected.connect(self._picked)
        self.pane.set_checks(self.context.settings.get("live", False), self.context.settings.get("sync", False),
                             self.context.settings.get("cameras", False))
        signal = getattr(getattr(self.context.window, "runner", None), "busy_changed", None)
        if signal is not None:  # one step at a time: the buttons wait while the program works
            signal.connect(self._busy)
            self._busy_signal = signal
        self.refresh_pane()

    def _busy(self, busy: bool) -> None:
        if self.pane is not None:
            self.pane.set_busy(bool(busy))

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ the chunk and its matrix
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        window = self.context.window
        chunk = getattr(window, "chunk", None) if window is not None else None
        return chunk if chunk is not None else getattr(doc, "chunk", None)

    @staticmethod
    def linked(chunk) -> dict:
        return dict((chunk.meta or {}).get(META) or {}) if chunk is not None else {}

    @staticmethod
    def _resolve(chunk, stored: dict, key: str) -> Path | None:
        """A path kept for the chunk: as written, else relative to the project (a project moved with its
        EM project)."""
        value = stored.get(key)
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return path
        doc = getattr(chunk, "document", None)
        rel = stored.get(key + "_rel")
        if rel and doc is not None and getattr(doc, "path", None):
            other = (Path(doc.path).parent / rel).resolve()
            if other.exists():
                return other
        return path

    def _store(self, chunk, **paths) -> None:
        stored = self.linked(chunk)
        doc = getattr(chunk, "document", None)
        for key, value in paths.items():
            if value is None:
                stored.pop(key, None)
                stored.pop(key + "_rel", None)
                continue
            stored[key] = str(Path(value).resolve())
            if doc is not None and getattr(doc, "path", None):
                try:
                    stored[key + "_rel"] = os.path.relpath(Path(value).resolve(), Path(doc.path).parent)
                except ValueError:  # another drive
                    stored.pop(key + "_rel", None)
        chunk.meta[META] = stored
        self.context.mark_modified()

    def matrix(self, chunk):
        """The matrix of the chunk: its em.json (read again when it changes), else the snapshot EMtools
        sent; None without either."""
        from .emjson import EmJsonError, load

        path = self._resolve(chunk, self.linked(chunk), "emjson") if chunk is not None else None
        if path is not None and path.exists():
            mtime = path.stat().st_mtime_ns
            cached = self._matrices.get(str(path))
            if cached is None or cached[0] != mtime:
                try:
                    cached = (mtime, load(path))
                except EmJsonError as exc:
                    self.context.log.warning("em.json not readable: %s", exc)
                    cached = (mtime, None)
                self._matrices[str(path)] = cached
            return cached[1]
        return self.snapshot

    def folders(self, chunk, ask: bool = False):
        """Where the EM project of the chunk is: from its em.json, else the folder chosen for it (asked
        when ``ask`` and never chosen)."""
        from . import scene

        stored = self.linked(chunk)
        root = self._resolve(chunk, stored, "root")
        emjson = self._resolve(chunk, stored, "emjson")
        if root is not None:
            return scene.project_folders(root=root)
        if emjson is not None and emjson.exists():
            return scene.project_folders(emjson=emjson)
        if not ask or self.context.window is None:
            return None
        from PySide6.QtWidgets import QFileDialog

        doc = getattr(chunk, "document", None)
        start = str(Path(doc.path).parent) if doc is not None and getattr(doc, "path", None) else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self.context.window, t("La cartella del progetto EM (EM, RB, RM...)",
                                                                         "The folder of the EM project (EM, RB, RM...)"),
                                                  start)
        if not folder:
            return None
        self._store(chunk, root=folder)
        return scene.project_folders(root=folder)

    def form_uuids(self, chunk) -> dict:
        """``{(area, tipo, us): node_uuid}`` of the US forms of the sites of the proxies, when a PyArchInit
        database is connected in the window: the id the em.json of PyArchInit gives the units."""
        window = self.context.window if self.context is not None else None
        pa = getattr(window, "pa", None) if window is not None else None
        if pa is None or chunk is None:
            return {}
        sites = {e.get("sito") for e in (chunk.meta or {}).get("proxies") or [] if e.get("sito")}
        if not sites:
            current = getattr(getattr(window, "pa_pane", None), "current_site", None)
            site = current() if callable(current) else None
            sites = {site} if site else set()
        out = {}
        for site in sites:
            try:
                rows = pa.stratigraphic_units(site)
            except Exception as exc:  # noqa: BLE001 - the pairing goes on by the fields of the forms
                self.context.log.warning("US forms of %s not read: %s", site, exc)
                continue
            for r in rows:
                if r.get("node_uuid"):
                    key = (str(r.get("area") or ""), str(r.get("unita_tipo") or "US"), str(r.get("us") or ""))
                    out[key] = str(r["node_uuid"])
        return out

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self) -> None:
        if self.pane is None:
            return
        chunk = self._chunk()
        matrix = self.matrix(chunk)
        stored = self.linked(chunk)
        lines = []
        emjson = self._resolve(chunk, stored, "emjson") if chunk is not None else None
        if emjson is not None:
            if matrix is not None and matrix.path is not None:
                lines.append(t("em.json: {0} - grafo «{1}», {2} {3}, {4} {5}",
                               "em.json: {0} - graph '{1}', {2} {3}, {4} {5}").format(
                    emjson.name, matrix.name, len(matrix.units), plural(len(matrix.units), t("unità", "unit"),
                                                                         t("unità", "units")),
                    len(matrix.epochs), plural(len(matrix.epochs), t("epoca", "epoch"), t("epoche", "epochs"))))
            else:
                lines.append(t("em.json non leggibile: {0}", "em.json not readable: {0}").format(emjson))
        elif matrix is not None:
            lines.append(t("Matrice di EMtools (sync): grafo «{0}», {1} unità",
                           "EMtools' matrix (sync): graph '{0}', {1} units").format(matrix.name, len(matrix.units)))
        else:
            lines.append(t("Nessun em.json collegato: i proxy prendono il nome del programma (US12, USM3). "
                           "PyArchInit lo scrive con Extended Matrix > Esporta sito in em.json.",
                           "No em.json linked: the proxies take the program's name (US12, USM3). PyArchInit writes "
                           "it with Extended Matrix > Esporta sito in em.json."))
        folders = self.folders(chunk) if chunk is not None else None
        if folders is not None:
            lines.append(t("Progetto EM: {0}", "EM project: {0}").format(folders.base))
        self.pane.set_matrix("\n".join(lines))
        self.pane.set_rows(rows_for(chunk, matrix, self.form_uuids(chunk)) if chunk is not None else [])
        self._show_status()

    def _show_status(self, extra: str = "") -> None:
        if self.pane is None:
            return
        from . import blender

        exe = blender.find_blender(self.context.settings.get("blender"))
        parts = [t("Blender: {0}", "Blender: {0}").format(exe) if exe else
                 t("Blender non trovato: Scegli il programma Blender...", "Blender not found: Choose the Blender program...")]
        if self.client is not None:
            state = {"connected": t("EMtools collegato", "EMtools connected"),
                     "connecting": t("collegamento a EMtools...", "connecting to EMtools..."),
                     "waiting": t("in attesa di EMtools (in Blender: EM > Sync)", "waiting for EMtools (in Blender: EM > "
                                                                                 "Sync)")}.get(self.client.state, "")
            if state:
                parts.append(f"{state} ({self.client.host}:{self.client.port})")
            if self.host.get("tool"):
                parts.append(str(self.host["tool"]))
        if extra:
            parts.append(extra)
        self.pane.set_status("\n".join(parts))
        connected = self.client is not None and self.client.connected
        self.pane.set_commands(connected and self.host.get("accepts_commands") is not False)

    def _tree_items(self, chunk) -> list:
        stored = self.linked(chunk)
        if not stored and not (chunk.meta or {}).get("proxies"):
            return []
        emjson = self._resolve(chunk, stored, "emjson")
        tip = t("em.json: {0}", "em.json: {0}").format(emjson) if emjson else t("Nessun em.json collegato",
                                                                                "No em.json linked")
        return [{"label": t("Progetto EM (Blender)", "EM project (Blender)"), "icon": "plugin", "tooltip": tip,
                 "activate": self.show_pane,
                 "actions": [(t("Esporta nel progetto EM", "Export into the EM project"), self.export),
                             (t("Apri in Blender", "Open in Blender"), self.open_blender),
                             (t("Collega l'em.json del sito...", "Link the em.json of the site..."), self.link_emjson)]}]

    # ------------------------------------------------------------------ actions
    def _message(self, kind: str, text: str, details: str = "") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text, details)

    def link_emjson(self) -> None:
        chunk = self._chunk()
        window = self.context.window
        if chunk is None or window is None:
            return
        from PySide6.QtWidgets import QFileDialog

        from .emjson import EmJsonError, load

        old = self._resolve(chunk, self.linked(chunk), "emjson")
        start = str(old.parent) if old is not None else (str(Path(chunk.document.path).parent)
                                                         if chunk.document and chunk.document.path else "")
        path, _f = QFileDialog.getOpenFileName(window, t("L'em.json del sito", "The em.json of the site"), start,
                                               "Extended Matrix (*.em.json *.emj *.json)")
        if not path:
            return
        try:
            matrix = load(path)
        except EmJsonError as exc:
            self._message("warning", t("Non è un em.json leggibile: {0}", "Not a readable em.json: {0}").format(exc))
            return
        self._store(chunk, emjson=path, root=None)
        self.refresh_pane()
        self.show_pane()
        missing = [r["name"] for r in rows_for(chunk, matrix, self.form_uuids(chunk)) if r["node_id"] is None]
        if missing:
            self._message("info", t("{0} proxy del chunk non hanno un'unità nella matrice: {1}",
                                    "{0} proxies of the chunk have no unit in the matrix: {1}").format(
                len(missing), ", ".join(missing)))

    def export(self, then=None) -> None:
        from . import scene

        chunk = self._chunk()
        if chunk is None:
            return
        if not (chunk.meta or {}).get("proxies") and (chunk.product_path("model") is None
                                                      or not chunk.product_path("model").exists()):
            self._message("warning", t("Il chunk non ha proxy delle US né un modello: crea i proxy (Strumenti > Proxy 3D "
                                       "delle US) o costruisci il modello.",
                                       "The chunk has no unit proxies and no model: make the proxies (Tools > 3D proxies "
                                       "of the units) or build the model."))
            return
        folders = self.folders(chunk, ask=True)
        if folders is None:
            return
        matrix = self.matrix(chunk)
        uuids = self.form_uuids(chunk)
        survey = bool(self.context.settings.get("survey", True))
        cameras = bool(self.context.settings.get("cameras", False))
        max_side = int(self.context.settings.get("photo_side") or 2048)

        def done(report):
            self._live_signature = self._signature(chunk)
            self.refresh_pane()
            self._show_status(report_text(report))
            if then is not None:
                then(report)
            elif self.context.window is not None:
                self._message("info", t("Progetto EM aggiornato.", "EM project updated."), report_text(report))

        self.context.run(t("Progetto EM", "EM project"),
                         lambda p: scene.export_project(chunk, folders, matrix, survey=survey, node_uuids=uuids,
                                                        progress=p, cameras=cameras, max_side=max_side), done)

    def open_blender(self) -> None:
        from . import blender

        def go(report):
            folders = self.folders(self._chunk())
            blends = sorted(Path(folders.base).glob("*.blend"), key=lambda p: p.stat().st_mtime, reverse=True) \
                if folders is not None else []
            emtools = bool(self.matrix(self._chunk()) is not None and not blends)
            try:
                blender.launch(report["manifest"], self.context.settings.get("blender"),
                               blends[0] if blends else None, emtools=emtools)
            except (RuntimeError, OSError) as exc:
                self._message("warning", str(exc))
                return
            self._show_status(t("Blender avviato su {0}", "Blender started on {0}").format(
                blends[0].name if blends else report["manifest"]))
            if not self.context.settings.get("live"):
                self.set_live(True)  # Blender follows: so does the program

        self.export(then=go)

    def choose_blender(self) -> None:
        window = self.context.window
        if window is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _f = QFileDialog.getOpenFileName(window, t("Il programma Blender", "The Blender program"),
                                               self.context.settings.get("blender") or "")
        if path:
            self.context.settings["blender"] = path
            self.context.save_settings()
            self._show_status()

    def set_cameras(self, on: bool) -> None:
        """The cameras with their photos in the next export (and taken away when off)."""
        self.context.settings["cameras"] = bool(on)
        self.context.save_settings()

    def _proxies_changed(self, chunk) -> None:
        """The proxies of the chunk changed here: the window's pane of the proxies and the project follow."""
        pane = getattr(self.context.window, "proxies_pane", None) if self.context.window is not None else None
        if pane is not None and chunk is self._chunk():
            try:
                pane.set_proxies(chunk.meta.get("proxies", []))
            except Exception as exc:  # noqa: BLE001 - an older window: the proxies are in the project anyway
                self.context.log.warning("proxies pane not refreshed: %s", exc)
        self.context.mark_modified()
        self.refresh_pane()

    def take_from_blender(self, quiet: bool = False) -> dict | None:
        """The proxies sent from Blender (*Invia al programma*) into the chunk, with their measures."""
        from . import inbox

        chunk = self._chunk()
        folders = self.folders(chunk) if chunk is not None else None
        if folders is None:
            if not quiet:
                self._message("warning", t("Il chunk non ha un progetto EM: esportalo prima.",
                                           "The chunk has no EM project: export it first."))
            return None
        if not inbox.pending(folders):
            if not quiet:
                self._message("info", t("Nessun proxy mandato da Blender (in Blender: pannello PPG, Invia al programma).",
                                        "No proxy sent from Blender (in Blender: PPG panel, Invia al programma)."))
            return None
        report = inbox.take(chunk, folders, self.matrix(chunk))
        if report["taken"]:
            self._proxies_changed(chunk)
        text = self._take_text(report)
        self._show_status(text)
        if not quiet or report["errors"]:
            self._message("info" if not report["errors"] else "warning", t("Proxy da Blender", "Proxies from Blender"),
                          text)
        return report

    @staticmethod
    def _take_text(report: dict) -> str:
        lines = []
        if report.get("taken"):
            lines.append(t("Da Blender {0}: {1}", "From Blender {0}: {1}").format(len(report["taken"]),
                                                                                   ", ".join(report["taken"])))
        if report.get("replaced"):
            lines.append(t("Sostituiti: {0}", "Replaced: {0}").format(", ".join(report["replaced"])))
        if report.get("open"):
            lines.append(t("Non chiusi, senza volume: {0}", "Not closed, no volume: {0}").format(", ".join(report["open"])))
        if report.get("republish"):
            lines.append(t("Da ripubblicare in PyArchInit: {0}", "To publish again in PyArchInit: {0}").format(
                ", ".join(report["republish"])))
        for name, reason in (report.get("errors") or {}).items():
            lines.append(t("{0} non letto: {1}", "{0} not read: {1}").format(name, reason))
        return "\n".join(lines)

    def proxies_in_blender(self) -> None:
        """EMtools starts the proxy of each unit picked in the pane that has none in the chunk."""
        from . import scene, sync

        chunk = self._chunk()
        matrix = self.matrix(chunk) if chunk is not None else None
        if self.client is None or not self.client.connected or matrix is None:
            self._message("warning", t("Serve la sincronizzazione con EMtools 1.6 e la sua matrice (Sincronizza con "
                                       "EMtools).", "The sync with EMtools 1.6 and its matrix are needed (Sync with "
                                       "EMtools)."))
            return
        if self.host.get("accepts_commands") is False:
            self._message("warning", t("EMtools non accetta comandi: in Blender abilita Accept commands from EMStudio "
                                       "(pannello EM, Sync).", "EMtools does not accept commands: in Blender enable "
                                       "Accept commands from EMStudio (EM panel, Sync)."))
            return
        if self.context.window is not None:
            self.context.store_drawings()  # the units drawn, for the place of each proxy
        with_proxy = {r["node_id"] for r in rows_for(chunk, matrix, self.form_uuids(chunk)) if r["node_id"] and r["proxy"]}
        units = [u for u in (matrix.unit(n) for n in self.pane.node_ids()) if u is not None and u.id not in with_proxy] \
            if self.pane is not None else []
        if not units:
            self._message("info", t("Scegli nella tabella le unità senza proxy.", "Pick in the table the units without a proxy."))
            return
        manifest = scene.read_manifest(self.folders(chunk).base) if self.folders(chunk) is not None else None
        shift = (manifest or {}).get("shift") or (list(matrix.shift) if matrix.shift else [0.0, 0.0, 0.0])
        for u in units:
            params = scene.proxy_start(chunk, u, shift)
            cmd_id = self.client.command(sync.PROXY_VERB, u.id, params)
            if cmd_id:
                self._commands[cmd_id] = u.name
        self._show_status(t("Chiesti a EMtools i proxy di: {0}", "EMtools asked for the proxies of: {0}").format(
            ", ".join(u.name for u in units)))

    # ------------------------------------------------------------------ live update
    def set_live(self, on: bool) -> None:
        self.context.settings["live"] = bool(on)
        self.context.save_settings()
        if self.pane is not None:
            self.pane.set_checks(bool(on), self.client is not None)
        if self._timer is not None:
            (self._timer.start if on else self._timer.stop)()

    def _signature(self, chunk):
        if chunk is None:
            return None
        sig = []
        for e in (chunk.meta or {}).get("proxies") or []:
            f = chunk.data_dir / str(e.get("file") or "")
            st = f.stat() if e.get("file") and f.exists() else None
            sig.append((e.get("name"), str(f), st.st_mtime_ns if st else 0, st.st_size if st else 0))
        model = chunk.product_path("model")
        sig.append(("model", str(model), model.stat().st_mtime_ns if model is not None and model.exists() else 0))
        emjson = self._resolve(chunk, self.linked(chunk), "emjson")
        sig.append(("emjson", str(emjson), emjson.stat().st_mtime_ns if emjson is not None and emjson.exists() else 0))
        sig.append(("snapshot", id(self.snapshot)))
        return tuple(sig)

    def _live_tick(self) -> None:
        """Every two seconds: when the proxies, the model or the em.json changed, the EM project is written
        again in a thread (no progress window: Blender shows the result)."""
        if self._live_busy:
            return
        runner = getattr(self.context.window, "runner", None)
        if runner is not None and getattr(runner, "busy", False):
            return  # a step is running: its result comes at the next look
        chunk = self._chunk()
        folders = self.folders(chunk) if chunk is not None else None
        if folders is not None:
            from . import inbox

            if inbox.pending(folders):  # proxies sent from Blender: into the chunk, then written again below
                self.take_from_blender(quiet=True)
        signature = self._signature(chunk)
        if signature is None or signature == self._live_signature:
            return
        folders = self.folders(chunk)
        if folders is None:
            return
        self._live_signature = signature
        from . import scene

        matrix, uuids = self.matrix(chunk), self.form_uuids(chunk)
        survey = bool(self.context.settings.get("survey", True))
        self._live_busy = True

        def work():
            try:
                result = scene.export_project(chunk, folders, matrix, survey=survey, node_uuids=uuids)
            except Exception as exc:  # noqa: BLE001 - said in the pane, the timer goes on
                result = exc
            self.context.call_in_gui(self._live_done, result)

        threading.Thread(target=work, name="ppg-blender-em-live", daemon=True).start()

    def _live_done(self, result) -> None:
        self._live_busy = False
        if isinstance(result, Exception):
            self.context.log.warning("live update of the EM project failed: %s", result)
            self._show_status(t("Aggiornamento non riuscito: {0}", "Update failed: {0}").format(result))
            return
        if result.get("written") or result.get("removed"):
            self.refresh_pane()
            self._show_status(report_text(result))

    # ------------------------------------------------------------------ the sync with EMtools
    def set_sync(self, on: bool) -> None:
        from . import sync

        self.context.settings["sync"] = bool(on)
        self.context.save_settings()
        if on and self.client is None:
            gui = self.context.call_in_gui
            self.client = sync.SyncClient(port=int(self.context.settings.get("port") or sync.PORT),
                                          on_message=lambda kind, payload: gui(self._on_sync, kind, payload),
                                          on_state=lambda state, text: gui(self._show_status),
                                          info=self._client_info)
            self.client.start()
        elif not on and self.client is not None:
            self.client.stop()
            self.client = None
            self.host = {}
        if self.pane is not None:
            self.pane.set_checks(bool(self.context.settings.get("live")), self.client is not None)
        self._show_status()

    def _client_info(self) -> dict:
        """What the program says of itself to EMtools: the matrix it has open."""
        chunk = self._chunk()
        matrix = self.matrix(chunk) if chunk is not None else None
        info = {"tool": "pyarchinit-photogrammetry", "document": str(matrix.path) if matrix and matrix.path else None,
                "graph_id": matrix.graph_id if matrix else None, "graph_name": matrix.name if matrix else None}
        return {k: v for k, v in info.items() if v}

    def _on_sync(self, kind: str, payload: dict) -> None:
        """A message of EMtools, in the interface thread."""
        from .emjson import EmJsonError, read

        if kind == "select":
            ids = payload.get("node_ids") or ([payload["node_id"]] if payload.get("node_id") else [])
            if self.pane is None:
                return
            names = self.pane.select_nodes(ids)
            chunk = self._chunk()
            matrix = self.matrix(chunk)
            unit = matrix.unit(str(payload.get("node_id") or "")) if matrix is not None else None
            if unit is not None:
                row = next((r for r in rows_for(chunk, matrix) if r["node_id"] == unit.id), {})
                self.pane.set_unit(t("In Blender: {0} ({1}{2})", "In Blender: {0} ({1}{2})").format(
                    unit.name, unit.tipo, f", {unit.epoch.name}" if unit.epoch else "") +
                    (("\n" + row["info"]) if row.get("info") else ""))
            elif names:
                self.pane.set_unit(t("In Blender: {0}", "In Blender: {0}").format(", ".join(names)))
        elif kind == "snapshot":
            doc = payload.get("doc")
            if isinstance(payload.get("host"), dict):
                self.host = payload["host"]
            try:
                self.snapshot = read(doc) if isinstance(doc, dict) else None
            except EmJsonError:
                self.snapshot = None
            self.refresh_pane()
        elif kind == "host_info":
            self.host = dict(payload)
            self._show_status()
        elif kind == "command_result":
            name = self._commands.pop(str(payload.get("cmd_id") or ""), None)
            if name is None:
                return  # the answer to a command of another client (EMStudio)
            if not payload.get("ok"):
                self._message("warning", t("EMtools non ha creato il proxy di {0}: {1}",
                                           "EMtools did not make the proxy of {0}: {1}").format(
                    name, payload.get("error") or "?"))
            elif (payload.get("info") or {}).get("already"):
                self._show_status(t("{0} ha già un proxy in Blender", "{0} already has a proxy in Blender").format(name))
            else:
                self._show_status(t("Proxy di {0} creato in Blender: modellalo, poi Invia al programma (pannello PPG)",
                                    "Proxy of {0} made in Blender: model it, then Invia al programma (PPG panel)").format(name))
        elif kind == "snapshot_unavailable":
            self._show_status(t("EMtools non ha un grafo caricato", "EMtools has no graph loaded"))

    def _picked(self, node_ids: list) -> None:
        """Units picked in the pane: their proxies selected and framed in Blender."""
        if self.client is not None:
            self.client.select(node_ids, active=node_ids[0] if node_ids else None)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        import pyarchinit_photogrammetry as ppg

        from . import blender, scene
        from .emjson import EmJsonError, load

        parser = argparse.ArgumentParser(prog="ppg plugin blender-em",
                                         description="The proxies and the survey into an Extended Matrix project.")
        sub = parser.add_subparsers(dest="what", required=True)
        for name, text in (("export", "write the EM project (EM/proxies, RB) and its manifest"),
                           ("open", "write the EM project and open it in Blender, kept up to date"),
                           ("take", "the proxies sent from Blender (Invia al programma) into the chunk, saved"),
                           ("status", "the units of the em.json and the proxies of the chunk paired to them")):
            p = sub.add_parser(name, help=text)
            p.add_argument("project", help="the .ppg project")
            p.add_argument("--chunk", help="label of the chunk (the active one by default)")
            p.add_argument("--emjson", help="the em.json of the site (the one linked to the chunk by default)")
            if name != "status":
                p.add_argument("--out", help="the folder of the EM project (else the project of the em.json)")
            if name in ("export", "open"):
                p.add_argument("--no-survey", action="store_true", help="the proxies only, not the model")
                p.add_argument("--cameras", action="store_true",
                               help="the aligned cameras with their photos undistorted (RB/<chunk>/foto)")
                p.add_argument("--photo-side", type=int, default=2048, help="long side of the photos (pixels)")
            if name == "open":
                p.add_argument("--blender", help="the Blender program (else the one found)")
        sub.add_parser("blender", help="the Blender programs found")
        a = parser.parse_args(list(args))
        if a.what == "blender":
            found = blender.candidates()
            print("\n".join(found) if found else "no Blender found (PPG_BLENDER names one)")
            return 0 if found else 1
        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}")
            return 2
        emjson = Path(a.emjson) if a.emjson else self._resolve(chunk, self.linked(chunk), "emjson")
        try:
            matrix = load(emjson) if emjson is not None and emjson.exists() else None
        except EmJsonError as exc:
            print(f"em.json not readable: {exc}")
            return 2
        if a.what == "status":
            for row in rows_for(chunk, matrix):
                print("\t".join(str(row.get(k) or "-") for k in ("name", "tipo", "epoch", "proxy")))
            return 0
        if a.out:
            folders = scene.project_folders(root=a.out)
        elif emjson is not None and emjson.exists():
            folders = scene.project_folders(emjson=emjson)
        else:
            stored = self._resolve(chunk, self.linked(chunk), "root")
            if stored is None:
                print("say where the EM project is: --out FOLDER or --emjson FILE")
                return 2
            folders = scene.project_folders(root=stored)
        if a.what == "take":
            from . import inbox

            taken = inbox.take(chunk, folders, matrix)
            print(self._take_text(taken) or "nothing sent from Blender")
            if taken["taken"]:
                doc.save()
            return 0 if not taken["errors"] else 1
        report = scene.export_project(chunk, folders, matrix, survey=not a.no_survey,
                                      cameras=True if a.cameras else None, max_side=a.photo_side)
        print(report_text(report))
        if a.what == "open":
            try:
                blender.launch(report["manifest"], a.blender, emtools=matrix is not None)
            except RuntimeError as exc:
                print(exc)
                return 1
        return 0
