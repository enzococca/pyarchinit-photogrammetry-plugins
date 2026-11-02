"""Architecture - a plugin of pyarchinit-photogrammetry: parametric architectural elements made from the
photogrammetric model, with CadQuery in a process of its own.

* columns by revolution (the axis and the profile measured on the model, a spline through K knots), mouldings and
  cornices swept along a path (the section cut from the model), arches and barrel vaults from a section (circle,
  pointed, ellipse or as measured, chosen by the fit);
* every element as STEP (CadQuery, OpenCascade) and as a parametric IFC solid (``IfcRevolvedAreaSolid``,
  ``IfcExtrudedAreaSolid``, ``IfcFixedReferenceSweptAreaSolid``) beside the mesh, the measured profile kept and the
  fit always given: how far the model is from the solid (median, 90th percentile, maximum);
* the *Elementi architettonici* pane (on the right), the elements in the workspace tree, *File > Esporta >
  Elementi architettonici (IFC)*, ``ppg plugin architecture column|moulding|arch|list|show|export|remove|install|
  status``.

The program's process never imports CadQuery: it runs in a Python of its own (see :mod:`.environment`). The chunk's
model, transform and products are never changed; nothing is written outside ``<chunk data>/architecture/`` but what
the user exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, arch_title, kind_title, metres, num, t

__version__ = "0.1.0"
NEGATIVE_VALUES = ("--window", "--pick", "--out-side", "--points", "--radii")


class ArchitecturePlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Colonne, modanature, archi e volte come solidi parametrici (STEP, IFC) dal modello "
                             "fotogrammetrico, con lo scarto dal modello", "Columns, mouldings, arches and vaults as "
                             "parametric solids (STEP, IFC) from the photogrammetric model, with their fit")
        self.context = None
        self.pane = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        from . import environment

        self.context = context
        environment.configure(context)
        context.add_action(t("Elementi architettonici...", "Architectural elements..."), self.show_pane,
                           tip=t("Il pannello degli elementi architettonici", "The pane of the architectural elements"))
        context.add_command("architecture", self.command,
                            help="parametric architectural elements: column, moulding, arch, list, show, export, "
                                 "remove, install, status")
        context.add_exporter(t("Elementi architettonici (IFC)", "Architectural elements (IFC)"), ".ifc", self._export)
        if context.window is not None:
            from .pane import ArchitecturePane

            self.pane = ArchitecturePane()
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            self.pane.requested.connect(self._requested)
            context.on_refresh(self.refresh_pane)
            context.add_tree_items(self._tree_items)
            self.refresh_pane()

    def deactivate(self) -> None:
        self.pane = None

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        window = self.context.window if self.context else None
        chunk = getattr(window, "chunk", None) if window is not None else None
        if chunk is not None:
            return chunk
        doc = self.context.app.document if self.context else None
        return None if doc is None else doc.chunk

    def _need_chunk(self):
        from .elements import NotReady

        chunk = self._chunk()
        if chunk is None:
            raise NotReady(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _message(self, kind: str, text: str, details: str = "") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text, details)

    def _status(self, text: str) -> None:
        window = self.context.window
        if window is None:
            print(text)
        elif hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 15000)

    def _changed(self, eid: str | None = None) -> None:
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane()
        if eid and self.pane is not None:
            self.pane.select(eid)

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        from .elements import NotReady
        from .environment import CadQueryMissing

        try:
            function(*args)
        except (NotReady, CadQueryMissing, ValueError) as exc:
            self._message("warning", str(exc))

    def _run(self, title: str, job, done) -> None:
        window = self.context.window
        if window is not None and hasattr(window, "run_task"):
            window.run_task(title, job, done, refresh=False, save=True)
        else:
            self.context.run(title, job, done)

    def _ask(self, title: str, fields: list, intro: str = "") -> dict | None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        return ParamDialog.ask(self.context.window, title, fields, intro=intro)

    def _need_view(self) -> None:
        """The plan to draw on: the chunk's own orthophoto or DEM, shown again even when the view has something else
        (an elevation shown for a path or a point: its coordinates are (u, v), not plan x, y); without either, an
        empty frame around the model."""
        if self.context.show_raster(None):
            return
        from . import elements

        chunk = self._need_chunk()
        v = elements._world_vertices(chunk)
        lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
        self.context.show_frame((lo[0], lo[1], hi[0], hi[1]), max(hi - lo) / 1500.0, t(*NAME))

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self) -> None:
        if self.pane is None:
            return
        from . import elements, environment

        problem = environment.unsupported()
        kind, _version = environment.where()
        self.pane.set_environment(environment.status_text(), problem is None and kind is None, problem is None and
                                  kind is not None)
        chunk = self._chunk()
        self.pane.set_rows(elements.elements(chunk) if chunk is not None else [])

    def _requested(self, name: str) -> None:
        self._guarded(getattr(self, "_do_" + name))

    def _selected(self, one: bool = False) -> list[str]:
        from .elements import NotReady

        ids = self.pane.selected_ids() if self.pane is not None else []
        if not ids:
            raise NotReady(t("Scegli un elemento nella tabella.", "Choose an element in the table."))
        return ids[:1] if one else ids

    # ------------------------------------------------------------------ install
    def _do_install(self) -> None:
        from . import environment

        def done(_env):
            self._status(environment.status_text())
            self.refresh_pane()

        self._run(t("Installazione di CadQuery", "Installing CadQuery"), lambda p: environment.install(p), done)

    # ------------------------------------------------------------------ column
    def _do_column(self) -> None:
        self._need_chunk()
        self._need_view()
        self.context.request_polygon(lambda xy: self._guarded(self._column_dialog, xy),
                                     hint=t("Disegna in pianta la zona della colonna (o della sua parte); doppio "
                                            "clic o "
                                            "Invio per finire", "Draw in plan the region of the column (or of its "
                                            "part); double click or Enter ends"))

    def _column_dialog(self, xy) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements, engine

        chunk = self._need_chunk()
        poly = np.asarray(xy, float)[:, :2]
        v = elements._world_vertices(chunk)
        inside = v[engine.points_in_polygon(v[:, :2], poly)]
        if not len(inside):
            raise elements.NotReady(t("Nella zona disegnata non c'è il modello.",
                                      "The model is not inside the region drawn."))
        z0, z1 = float(inside[:, 2].min()), float(inside[:, 2].max())
        columns = [("", t("(misurato su questa parte)", "(measured on this part)"))] + \
            [(e["id"], e["name"]) for e in elements.elements(chunk) if e["kind"] == "column"]
        val = self._ask(t("Colonna", "Column"), [
            Field("name", t("Nome", "Name"), elements._new_name(chunk, "column"), kind="text"),
            Field("zmin", t("Quota minima (m)", "Lowest height (m)"), z0, minimum=-1e5, maximum=1e5, decimals=3),
            Field("zmax", t("Quota massima (m)", "Highest height (m)"), z1, minimum=-1e5, maximum=1e5, decimals=3),
            Field("knots", t("Nodi del profilo", "Knots of the profile"), 12, minimum=2, maximum=60, decimals=0,
                  tip=t("Il profilo è una spline per questi nodi, adattata ai minimi quadrati al profilo misurato",
                        "The profile is a spline through these knots, fitted by least squares to the measured "
                        "profile")),
            Field("measured", t("Come misurato (tutti i punti del profilo)",
                                "As measured (every point of the profile)"),
                  False),
            Field("axis", t("Stesso asse di", "Same axis as"), "", choices=columns)],
            intro=t("Base, fusto e capitello si fanno come elementi separati sullo stesso asse.",
                    "Base, shaft and capital are made as separate elements on the same axis."))
        if not val:
            return
        self.column(chunk, polygon=poly, zmin=val["zmin"], zmax=val["zmax"], knots=int(val["knots"]),
                    as_measured=val["measured"], same_axis=val["axis"] or None, name=val["name"])

    def column(self, chunk, **kwargs) -> None:
        from . import elements

        def done(e):
            self._status(elements.summary(e))
            self._changed(e["id"])
            self._show(e)

        self._run(t("Colonna", "Column"), lambda p: elements.column(chunk, progress=p, **kwargs), done)

    # ------------------------------------------------------------------ moulding
    def _do_moulding(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements

        chunk = self._need_chunk()
        names = [(m.label, m.label) for m in chunk.markers if m.position is not None]
        views = [(x["id"], x.get("name") or x["id"]) for x in elements.elevation_entries(chunk)]
        how = [("trace", t("tracciato sulla pianta", "traced on the plan"))]
        if views:
            how.append(("elevation", t("tracciato su un prospetto o una sezione",
                                       "traced on an elevation or a section")))
        if len(names) >= 2:
            how.append(("markers", t("dritto, da due marker", "straight, from two markers")))
        if len(how) > 1:
            fields = [Field("how", t("Percorso", "Path"), how[0][0], choices=how)]
            if views:
                fields.append(Field("view", t("Prospetto o sezione", "Elevation or section"), views[0][0],
                                    choices=views))
            if len(names) >= 2:
                fields += [Field("a", t("Primo marker", "First marker"), names[0][0], choices=names),
                           Field("b", t("Secondo marker", "Second marker"), names[1][0], choices=names)]
            val = self._ask(t("Percorso della modanatura", "Path of the moulding"), fields)
            if not val:
                return
            if val["how"] == "markers":
                self._moulding_dialog(elements.marker_points(chunk, [val["a"], val["b"]]))
                return
            if val["how"] == "elevation":
                eid = val["view"]
                if not self.context.show_raster("ortho", path=elements.elevation_raster(chunk, eid)):
                    raise elements.NotReady(t("Il prospetto non si può mostrare.", "The elevation cannot be shown."))
                self.context.request_line(
                    lambda uv: self._guarded(self._moulding_dialog, elements.from_elevation(chunk, eid, uv), True),
                    hint=t("Traccia il percorso della modanatura sul prospetto; doppio clic o Invio per finire",
                           "Trace the path of the moulding on the elevation; double click or Enter ends"))
                return
        self._need_view()
        self.context.request_line(lambda xy: self._guarded(self._moulding_dialog, xy),
                                  hint=t("Traccia il percorso della modanatura lungo il muro; doppio clic o Invio per "
                                         "finire", "Trace the path of the moulding along the wall; double click or "
                                         "Enter ends"))

    def _moulding_dialog(self, xy, in_space: bool = False) -> None:
        """The parameters of a moulding along ``xy`` (plan points, or world points with ``in_space``: a path drawn on
        an elevation keeps its heights, a raking cornice too)."""
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements, engine

        chunk = self._need_chunk()
        xy = np.asarray(xy, float)
        path = xy if in_space else xy[:, :2]
        v, f = elements.local_mesh(chunk)
        o = elements.origin(chunk)
        if xy.shape[1] >= 3:
            z = float(xy[:, 2].mean()) - o[2]
        else:
            z = engine.model_height_at(v, f, path[0, :2] - o[:2], radius=0.2)
        fields = [Field("name", t("Nome", "Name"), elements._new_name(chunk, "moulding"), kind="text")]
        if not in_space:
            fields.append(Field("height", t("Quota della modanatura (m)", "Height of the moulding (m)"),
                                float(z + o[2]) if np.isfinite(z) else float(o[2]), minimum=-1e5, maximum=1e5,
                                decimals=3, tip=t("La quota del percorso: la sezione si cerca intorno a questa quota",
                                                  "The height of the path: the section is looked for around it")))
        fields += [
            Field("station", t("Sezione a (m dall'inizio)", "Section at (m from the start)"),
                  round(0.5 * float(np.linalg.norm(path[1] - path[0])), 3), minimum=0.0, maximum=1e5, decimals=3),
            Field("below", t("Finestra: sotto (m, 0 = automatica)", "Window: below (m, 0 = automatic)"), 0.0,
                  minimum=0.0, maximum=100.0, decimals=3,
                  tip=t("Automatica: la parte del taglio che sporge dal muro più vicina al percorso, con 10 cm "
                        "intorno", "Automatic: the part of the cut standing out of the wall nearest the path, with "
                        "10 cm around it")),
            Field("above", t("Finestra: sopra (m, 0 = automatica)", "Window: above (m, 0 = automatic)"), 0.0,
                  minimum=0.0, maximum=100.0, decimals=3),
            Field("reach", t("Finestra: sporgenza massima (m, 0 = automatica)",
                             "Window: largest projection (m, 0 = automatic)"), 0.0, minimum=0.0, maximum=100.0,
                  decimals=3),
            Field("side", t("Sporge verso", "Projects towards"), "auto",
                  choices=[("auto", t("le foto", "the photos")),
                           ("left", t("sinistra del percorso", "left of the path")),
                           ("right", t("destra del percorso", "right of the path"))]),
            Field("depth", t("Profondità nel muro (m)", "Depth into the wall (m)"), 0.1, minimum=0.0, maximum=10.0,
                  decimals=3),
            Field("simplify", t("Semplificazione (m)", "Simplification (m)"), 0.002, minimum=0.0, maximum=1.0,
                  decimals=4)]
        val = self._ask(t("Modanatura", "Moulding"), fields)
        if not val:
            return
        out = {"auto": None, "left": 1.0, "right": -1.0}[val["side"]]
        if val["reach"] or val["below"] or val["above"]:
            reach = val["reach"] or 1.0
            win = (-reach, -(val["below"] or 1.0), reach, val["above"] or 1.0)
        else:
            win = None  # found from the cut
        self.moulding(chunk, path=path, height=None if in_space else val["height"], station=val["station"], window=win,
                      out=out, depth=val["depth"], simplify=val["simplify"], name=val["name"])

    def moulding(self, chunk, **kwargs) -> None:
        from . import elements

        def done(e):
            self._status(elements.summary(e))
            self._changed(e["id"])
            self._show(e)

        self._run(t("Modanatura", "Moulding"), lambda p: elements.moulding(chunk, progress=p, **kwargs), done)

    # ------------------------------------------------------------------ arch
    def _do_arch(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements

        chunk = self._need_chunk()
        sections = elements.section_planes(chunk)
        choices = [(s["id"], s.get("name") or s["id"]) for s in sections] + \
            [("", t("Un piano per due punti sulla pianta...", "A plane through two points on the plan..."))]
        fields = [Field("section", t("Sezione", "Section"), choices[0][0], choices=choices)]
        if sections:
            fields.append(Field("pick", t("Punto nell'apertura", "Point in the opening"), "click", choices=[
                ("click", t("cliccato sulla sezione", "clicked on the section")),
                ("type", t("scritto (u, v)", "typed (u, v)"))]))
        val = self._ask(t("Arco / volta", "Arch / vault"), fields,
                        intro=t("L'arco si legge su una sezione verticale del modello (Strumenti > Prospetti e "
                                "sezioni) "
                                "o su un piano verticale tracciato ora in pianta.", "The arch is read on a vertical "
                                "section of the model (Tools > Elevations and sections) or on a vertical plane traced "
                                "now in plan."))
        if not val:
            return
        if val["section"]:
            if val.get("pick") == "click":
                if not self.context.show_raster("ortho", path=elements.elevation_raster(chunk, val["section"])):
                    raise elements.NotReady(t("La sezione non si può mostrare.", "The section cannot be shown."))
                self.context.request_point(
                    lambda uv: self._guarded(self._arch_dialog, chunk, val["section"], None,
                                             np.asarray(uv, float)[0, :2]),
                    hint=t("Clic dentro l'apertura, sotto l'arco", "Click inside the opening, under the arch"))
                return
            self._arch_dialog(chunk, section=val["section"])
            return
        self._need_view()
        self.context.request_line(lambda xy: self._guarded(self._arch_dialog, chunk, None, np.asarray(xy, float)),
                                  hint=t("Due punti: la traccia del piano dell'arco in pianta", "Two points: the trace "
                                         "of the plane of the arch in plan"))

    def _arch_dialog(self, chunk, section=None, xy=None, pick=None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements, engine

        p1 = p2 = None
        if xy is not None:
            p1, p2 = xy[0, :2], xy[-1, :2]
        po, u, vv, _what = elements.plane_of(chunk, section, p1, p2)
        v, f = elements.local_mesh(chunk)
        lines = engine.mesh_section(v, f, po, u, vv)
        if not lines:
            raise elements.NotReady(t("Il piano non taglia il modello.", "The plane does not cut the model."))
        guess = tuple(map(float, pick)) if pick is not None else engine.guess_pick(lines)
        val = self._ask(t("Arco / volta", "Arch / vault"), [
            Field("name", t("Nome", "Name"), elements._new_name(chunk, "arch"), kind="text"),
            Field("pick_u", t("Punto nell'apertura: u (m)", "Point in the opening: u (m)"), guess[0], minimum=-1e5,
                  maximum=1e5, decimals=3, tip=t("Un punto dentro l'apertura, sotto l'arco, nelle coordinate della "
                                                 "sezione", "A point inside the opening, under the arch, in the "
                                                 "coordinates of the section")),
            Field("pick_v", t("Punto nell'apertura: v (m)", "Point in the opening: v (m)"), guess[1], minimum=-1e5,
                  maximum=1e5, decimals=3),
            Field("kind", t("Tipo", "Type"), "auto", choices=[("auto", t("il migliore (scarto)", "the best (fit)"))] +
                  [(k, arch_title(k)) for k in ("semicircular", "segmental", "pointed", "elliptical", "measured")]),
            Field("thickness", t("Spessore (m, 0 = misurato)", "Thickness (m, 0 = measured)"), 0.0, minimum=0.0,
                  maximum=100.0, decimals=3),
            Field("depth", t("Profondità (m, 0 = misurata)", "Depth (m, 0 = measured)"), 0.0, minimum=0.0,
                  maximum=1000.0, decimals=3),
            Field("vault", t("Volta a botte", "Barrel vault"), False)])
        if not val:
            return
        self.arch(chunk, pick=(val["pick_u"], val["pick_v"]), section=section, p1=p1, p2=p2, kind=val["kind"],
                  thickness=val["thickness"] or None, depth=val["depth"] or None, vault=val["vault"], name=val["name"])

    def arch(self, chunk, **kwargs) -> None:
        from . import elements

        def done(e):
            self._status(elements.summary(e))
            self._changed(e["id"])
            self._show(e)

        self._run(t("Arco", "Arch"), lambda p: elements.arch(chunk, progress=p, **kwargs), done)

    # ------------------------------------------------------------------ the chosen elements
    def _show(self, e: dict) -> None:
        from . import elements

        chunk = self._chunk()
        if chunk is not None:
            elements.check_frame(chunk)
        path = elements.path_of(chunk, e, "ply") if chunk is not None else None
        if path is not None and path.exists():
            self.context.show_model("model", path=path, title=e["name"])

    def _do_show(self) -> None:
        from . import elements

        self._show(elements.need(self._need_chunk(), self._selected(one=True)[0]))

    def _do_edit(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import elements

        chunk = self._need_chunk()
        e = elements.need(chunk, self._selected(one=True)[0])
        p = e["params"]
        if e["kind"] == "column":
            fields = [Field("measured", t("Come misurato", "As measured"), p["method"] == "measured"),
                      Field("knots", t("Nodi", "Knots"), p["knots"], minimum=2, maximum=60, decimals=0)]
            fields += [Field(f"r{k}", t(f"Raggio al nodo {k + 1}, h {num(z, 3)} m",
                                        f"Radius at knot {k + 1}, h {num(z, 3)} m"),
                             r, minimum=0.001, maximum=100.0, decimals=4)
                       for k, (z, r) in enumerate(zip(p["knot_heights"], p["knot_radii"]))] \
                if p["method"] == "spline" else []
            val = self._ask(t("Parametri della colonna", "Parameters of the column"), fields,
                            intro=t("Il profilo misurato resta com'è: cambia solo il solido.",
                                    "The measured profile stays as it is: only the solid changes."))
            if not val:
                return
            if val["measured"] != (p["method"] == "measured") or int(val["knots"]) != p["knots"]:
                job = lambda prog: elements.edit_column(chunk, e["id"], knots=int(val["knots"]),  # noqa: E731
                                                        as_measured=val["measured"], progress=prog)
            else:
                radii = [val[f"r{k}"] for k in range(len(p["knot_radii"]))] if p["method"] == "spline" else None
                job = lambda prog: elements.edit_column(chunk, e["id"], knot_radii=radii, progress=prog)  # noqa: E731
        elif e["kind"] == "moulding":
            src = e["source"]
            val = self._ask(t("Parametri della modanatura", "Parameters of the moulding"), [
                Field("simplify", t("Semplificazione (m)", "Simplification (m)"), src["simplify"], minimum=0.0,
                      maximum=1.0,
                      decimals=4),
                Field("depth", t("Profondità nel muro (m)", "Depth into the wall (m)"), src["depth"], minimum=0.0,
                      maximum=10.0, decimals=3)])
            if not val:
                return
            job = lambda prog: elements.edit_moulding(chunk, e["id"], simplify=val["simplify"],  # noqa: E731
                                                      depth=val["depth"], progress=prog)
        else:
            val = self._ask(t("Parametri dell'arco", "Parameters of the arch"), [
                Field("kind", t("Tipo", "Type"), p["type"],
                      choices=[(k, arch_title(k)) for k in ("semicircular", "segmental", "pointed", "elliptical",
                                                            "measured")]),
                Field("thickness", t("Spessore (m)", "Thickness (m)"), p["thickness"], minimum=0.001, maximum=100.0,
                      decimals=3),
                Field("depth", t("Profondità (m)", "Depth (m)"), p["depth"], minimum=0.001, maximum=1000.0,
                      decimals=3)])
            if not val:
                return
            kind = val["kind"] if val["kind"] != p["type"] else None
            job = lambda prog: elements.edit_arch(chunk, e["id"], kind=kind,  # noqa: E731
                                                  thickness=val["thickness"], depth=val["depth"], progress=prog)

        def done(new):
            self._status(elements.summary(new))
            self._changed(new["id"])
            self._show(new)

        self._run(t("Parametri dell'elemento", "Parameters of the element"), job, done)

    def _do_step(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import elements

        chunk = self._need_chunk()
        ids = self._selected()
        if len(ids) == 1:
            e = elements.need(chunk, ids[0])
            path, _ = QFileDialog.getSaveFileName(self.context.window, t("Esporta STEP", "Export STEP"),
                                                  f"{elements._safe(e['name'])}.step", "STEP (*.step *.stp)")
        else:
            path = QFileDialog.getExistingDirectory(self.context.window,
                                                    t("Cartella degli STEP", "Folder of the STEP files"))
        if path:
            out = elements.export_step(chunk, ids, path)
            where = Path(out[0]).parent
            self._status(t(f"{len(out)} file STEP in {where}", f"{len(out)} STEP files in {where}"))

    def _do_ifc(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ifc

        chunk = self._need_chunk()
        ids = self.pane.selected_ids() if self.pane is not None else []
        val = self._ask(t("Esporta IFC", "Export IFC"), [
            Field("which", t("Elementi", "Elements"), "selected" if ids else "all",
                  choices=[("all", t("tutti", "all")), ("selected", t("quelli scelti", "the chosen ones"))]),
            Field("mesh", t("Con il modello del rilievo", "With the survey model"), False)])
        if not val:
            return
        path, _ = QFileDialog.getSaveFileName(self.context.window, t("Esporta IFC", "Export IFC"), "elementi.ifc",
                                              "IFC (*.ifc)")
        if not path:
            return
        chosen = ids if val["which"] == "selected" else None
        self._run(t("Esportazione IFC", "IFC export"),
                  lambda p: ifc.export(chunk, path, chosen, with_mesh=val["mesh"], progress=p),
                  lambda out: self._status(t(f"IFC scritto: {out}", f"IFC written: {out}")))

    def _do_dxf(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import drawings, elements

        chunk = self._need_chunk()
        e = elements.need(chunk, self._selected(one=True)[0])
        path, _ = QFileDialog.getSaveFileName(self.context.window, "DXF", f"{elements._safe(e['name'])}.dxf",
                                              "DXF (*.dxf)")
        if path:
            self._status(str(drawings.write_dxf(e, path)))

    def _do_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from . import elements

        chunk = self._need_chunk()
        ids = self.pane.selected_ids() if self.pane is not None else []
        target = elements.element_dir(chunk, elements.need(chunk, ids[0])) if ids else elements.folder(chunk)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _do_rebuild(self) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import elements

        chunk = self._need_chunk()
        if elements.frame_matches(chunk):
            self._status(t("Il sistema del chunk non è cambiato: niente da ricostruire.",
                           "The chunk's system did not change: nothing to rebuild."))
            return
        if dialogs.choice(self.context.window, "question", self.title,
                          t("Il sistema del chunk è cambiato. Ricostruire tutti gli elementi nel sistema nuovo, "
                            "misurandoli di nuovo sul modello?", "The chunk's system changed. Rebuild every element in "
                            "the new system, measuring them again on the model?"),
                          [t("Ricostruisci", "Rebuild")]) != 0:
            return

        def done(made):
            self._status(t(f"{len(made)} elementi ricostruiti", f"{len(made)} elements rebuilt"))
            self._changed()

        self._run(t("Ricostruzione degli elementi", "Rebuilding the elements"),
                  lambda p: elements.rebuild(chunk, progress=p), done)

    def _do_remove(self) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import elements

        chunk = self._need_chunk()
        ids = self._selected()
        names = ", ".join(elements.need(chunk, i)["name"] for i in ids)
        if dialogs.choice(self.context.window, "question", self.title,
                          t(f"Rimuovere {names} e i suoi file?", f"Remove {names} and its files?"),
                          [t("Rimuovi", "Remove")]) != 0:
            return
        for i in ids:
            elements.remove(chunk, i)
        self._changed()

    def _export(self, chunk, path, progress=None):
        from . import ifc

        return ifc.export(chunk, path, progress=progress)

    # ------------------------------------------------------------------ the workspace tree
    def _tree_items(self, chunk) -> list:
        from . import elements

        items = elements.elements(chunk)
        if not items:
            return []
        kids = []
        for e in items:
            fit = e.get("fit") or {}
            label = f"{e['name']} ({kind_title(e['kind']).lower()}, {t('scarto', 'fit')} {metres(fit.get('median'))})"
            kids.append({"label": label, "icon": "plugin", "activate": lambda x=e: self._show(x)})
        return [{"label": t(*NAME), "icon": "plugin", "children": kids}]

    # ------------------------------------------------------------------ the command line
    def command(self, *args) -> int:
        from shapely import wkt

        from . import drawings, elements, environment, ifc

        parser = argparse.ArgumentParser(prog="ppg plugin architecture",
                                         description="parametric architectural elements from the photogrammetric model")
        parser.add_argument("what", choices=["column", "moulding", "arch", "list", "show", "export", "remove",
                                             "rebuild", "install", "status"])
        parser.add_argument("project", nargs="?", help="the .ppg file")
        parser.add_argument("element", nargs="?", help="show, remove: the id or the name of an element")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--name", help="the name of the new element")
        parser.add_argument("--polygon", help="column: the region in plan, WKT POLYGON (world x y)")
        parser.add_argument("--zmin", type=float, help="column: the lowest height (world)")
        parser.add_argument("--zmax", type=float, help="column: the highest height (world)")
        parser.add_argument("--box", help="column: xmin,ymin,zmin,xmax,ymax,zmax (world) instead of the polygon")
        parser.add_argument("--knots", type=int, default=12, help="column: knots of the spline (default 12)")
        parser.add_argument("--measured", action="store_true",
                            help="column: the profile as measured; arch: --type measured")
        parser.add_argument("--same-axis", help="column: the id or name of a column whose axis this part shares")
        parser.add_argument("--path", help="moulding: WKT LINESTRING (x y, or x y z) world")
        parser.add_argument("--markers", help="moulding: two markers A,B: a straight path between them")
        parser.add_argument("--elevation",
                            help="moulding: --path is in (u v) of this elevation or section (id or name)")
        parser.add_argument("--height", type=float, help="moulding: the height of a 2D path (world)")
        parser.add_argument("--station", type=float, help="moulding: the section at this distance from the start (m)")
        parser.add_argument("--window", help="moulding: u0,v0,u1,v1 around the path point (default: found from the "
                                             "cut, the moulding's extent and 10 cm)")
        parser.add_argument("--out-side", type=float,
                            help="moulding: +1 the moulding projects to the left of the path, "
                                                            "-1 to the right (default: the side of the photos)")
        parser.add_argument("--depth", type=float, help="moulding: into the wall (default 0.1 m); arch: of the vault")
        parser.add_argument("--simplify", type=float, default=0.002, help="moulding: Douglas-Peucker tolerance (m)")
        parser.add_argument("--section", help="arch: the id or name of a section of the project")
        parser.add_argument("--points", help="arch: x1,y1,x2,y2 world, the trace of a vertical plane")
        parser.add_argument("--pick", help="arch: u,v of a point inside the opening (default: a guess)")
        parser.add_argument("--type", default="auto", help="arch: auto, semicircular, segmental, pointed, elliptical, "
                                                           "measured")
        parser.add_argument("--thickness", type=float, help="arch: thickness of the ring (default: measured)")
        parser.add_argument("--vault", action="store_true", help="arch: a barrel vault")
        parser.add_argument("--out", help="export: the file (.ifc, .step, .dxf) or a folder for several STEP")
        parser.add_argument("--ids", help="export: ids or names separated by commas (default: all)")
        parser.add_argument("--mesh", action="store_true", help="export IFC: the survey model beside the elements")
        ns = parser.parse_args(joined_values([str(a) for a in args]))
        if ns.what == "status":
            print(environment.status_text())
            if not ns.project:
                return 0
        if ns.what == "install":
            from pyarchinit_photogrammetry.core.progress import print_progress

            try:
                print(environment.install(print_progress))
            except environment.CadQueryMissing as exc:
                print(exc)
                return 1
            return 0
        if not ns.project:
            print("the project (.ppg) is needed")
            return 1
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        changed = False
        try:
            if ns.what == "column":
                polygon = np.asarray(wkt.loads(ns.polygon).exterior.coords)[:, :2] if ns.polygon else None
                box = [float(x) for x in ns.box.split(",")] if ns.box else None
                if polygon is None and box is None:
                    raise elements.NotReady("--polygon or --box")
                e = elements.column(chunk, polygon=polygon, zmin=ns.zmin, zmax=ns.zmax, box=box, knots=ns.knots,
                                    as_measured=ns.measured, same_axis=ns.same_axis, name=ns.name,
                                    progress=print_progress)
                print(elements.summary(e))
                changed = True
            elif ns.what == "moulding":
                if ns.markers:
                    path = elements.marker_points(chunk, [x.strip() for x in ns.markers.split(",")])
                elif ns.path and ns.elevation:
                    path = elements.from_elevation(chunk, ns.elevation, np.asarray(wkt.loads(ns.path).coords))
                elif ns.path:
                    path = np.asarray(wkt.loads(ns.path).coords)
                else:
                    raise elements.NotReady("--path or --markers")
                win = [float(x) for x in ns.window.split(",")] if ns.window else None
                e = elements.moulding(chunk, path, station=ns.station, window=win, out=ns.out_side,
                                      depth=0.1 if ns.depth is None else ns.depth, simplify=ns.simplify,
                                      height=ns.height, name=ns.name, progress=print_progress)
                print(elements.summary(e))
                changed = True
            elif ns.what == "arch":
                p1 = p2 = None
                if ns.points:
                    x1, y1, x2, y2 = (float(x) for x in ns.points.split(","))
                    p1, p2 = np.array([x1, y1]), np.array([x2, y2])
                elif not ns.section:
                    raise elements.NotReady("--section or --points")
                if ns.pick:
                    pick = tuple(float(x) for x in ns.pick.split(","))
                else:
                    from . import engine

                    po, u, vv, _w = elements.plane_of(chunk, ns.section, p1, p2)
                    v, f = elements.local_mesh(chunk)
                    pick = engine.guess_pick(engine.mesh_section(v, f, po, u, vv))
                    print(f"pick u,v = {pick[0]:.3f},{pick[1]:.3f}")
                e = elements.arch(chunk, pick, section=ns.section, p1=p1, p2=p2,
                                  kind="measured" if ns.measured else ns.type, thickness=ns.thickness, depth=ns.depth,
                                  vault=ns.vault, name=ns.name, progress=print_progress)
                print(elements.summary(e))
                changed = True
            elif ns.what == "list":
                for e in elements.elements(chunk):
                    print(f"{e['id']}  {e['name']}  {elements.summary(e)}")
            elif ns.what == "show":
                if not ns.element:
                    raise elements.NotReady("the id or the name of the element")
                print(json.dumps(elements.need(chunk, ns.element), indent=1, ensure_ascii=False))
            elif ns.what == "remove":
                if not ns.element or not elements.remove(chunk, ns.element):
                    raise elements.NotReady(f"no element {ns.element!r}")
                changed = True
            elif ns.what == "rebuild":
                if elements.frame_matches(chunk):
                    print("the chunk's system did not change: nothing to rebuild")
                else:
                    for e in elements.rebuild(chunk, progress=print_progress):
                        print(elements.summary(e))
                    changed = True
            elif ns.what == "export":
                if not ns.out:
                    raise elements.NotReady("--out FILE (.ifc, .step, .dxf) or a folder")
                ids = [x.strip() for x in ns.ids.split(",")] if ns.ids else None
                suffix = Path(ns.out).suffix.lower()
                if suffix == ".ifc":
                    print(ifc.export(chunk, ns.out, ids, with_mesh=ns.mesh, progress=print_progress))
                elif suffix == ".dxf":
                    chosen = ids or [e["id"] for e in elements.elements(chunk)]
                    if len(chosen) != 1:
                        raise elements.NotReady("a DXF is made for one element: --ids")
                    print(drawings.write_dxf(elements.need(chunk, chosen[0]), ns.out))
                else:
                    for p in elements.export_step(chunk, ids, ns.out):
                        print(p)
            else:
                for e in elements.elements(chunk):
                    print(elements.summary(e))
        except (elements.NotReady, environment.CadQueryMissing, ValueError) as exc:
            print(exc)
            return 1
        if changed:
            doc.save()
        return 0


def joined_values(args: list[str]) -> list[str]:
    """``--window -0.5,-0.5,0.5,0.5`` as ``--window=-0.5,...``: argparse takes for an option anything that begins
    with a minus and is not one negative number."""
    out: list[str] = []
    for a in args:
        if out and out[-1] in NEGATIVE_VALUES and a[:1] == "-" and a[1:2] in tuple("0123456789."):
            out[-1] = f"{out[-1]}={a}"
        else:
            out.append(a)
    return out
