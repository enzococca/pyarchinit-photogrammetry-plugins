"""The elements as parametric IFC4 solids, with the conventions of the program's ``bim.ifc.export_ifc``: project
"Rilievo" and site "Sito", lengths in metres, coordinates relative to a local origin georeferenced with
``IfcMapConversion`` and ``IfcProjectedCRS`` (the chunk's export origin: the same as the STEP files), each element
contained in the site.

* a column: ``IfcColumn`` with an ``IfcRevolvedAreaSolid``, the half-section ``(r, z)`` as an
  ``IfcArbitraryClosedProfileDef`` revolved about the element's z axis;
* a moulding: ``IfcMember`` (MOULDING... IFC has no such type: ``USERDEFINED`` with object type "Moulding") with an
  ``IfcExtrudedAreaSolid`` when the path is straight, else an ``IfcFixedReferenceSweptAreaSolid`` on an
  ``IfcIndexedPolyCurve`` directrix with the fixed reference up (mitred corners, as the STEP);
* an arch or a vault: ``IfcBuildingElementProxy`` (object type "Arch" / "Vault") with the ring's section (lines
  and arcs: ``IfcIndexedPolyCurve`` with ``IfcLineIndex`` and ``IfcArcIndex``) extruded along the plane's normal;
* an opening: ``IfcOpeningElement`` (OPENING) with the plan of the void (``IfcArbitraryClosedProfileDef``) as an
  ``IfcExtrudedAreaSolid`` from the sill up, and the void under its arch (lines and arcs) extruded across the wall
  when an arch closes it; a door or a window also as ``IfcDoor`` / ``IfcWindow`` (DOOR / WINDOW) with the same solid
  and ``OverallWidth`` / ``OverallHeight``. Not related to a wall (the program has no wall element).

``Pset_PyArchInit_Element``: kind, method, name, the parameters (metres and degrees), the fit (median, p90, max in
metres, points), the source and a note that the solid is an interpretation of the measured model. The chunk's mesh
may be written beside the elements as the program's IFC export writes it (``IfcGeographicElement``), so the HBIM
model holds both.
"""
from __future__ import annotations

import contextlib
import json
import locale
from pathlib import Path

import numpy as np

from .texts import t

PSET = "Pset_PyArchInit_Element"
FRAMES = ("Lengths in metres. P_world_*, P_axis_point and P_axis_direction: the CRS of the map conversion (world). "
          "P_springing, P_centre_u, P_centre_v, P_span, P_rise and the other arch values: the section plane, u "
          "along it and v up, from the plane's origin (Source.plane, world); P_wall_u, P_v_range, P_window_used of a "
          "moulding: the plane normal to its path at the station, u = up x tangent, v up, from the path's point. "
          "P_sill, P_lintel, P_springing, P_ground of an opening: world heights; P_splay_left/right: degrees, positive "
          "when the reveal opens out towards the inner face; P_world_axis: across the wall towards the outer face. The "
          "geometry: the IFC's local coordinates, the map conversion's origin added.")
QTO = "Qto_PyArchInit_Element"
METHOD = "pyarchinit-photogrammetry: solido CadQuery (OpenCascade)"


def _require():
    try:
        import ifcopenshell.api  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(t("L'IFC richiede ifcopenshell: pip install 'pyarchinit-photogrammetry[bim]'.",
                            "IFC needs ifcopenshell: pip install 'pyarchinit-photogrammetry[bim]'.")) from exc


@contextlib.contextmanager
def c_numbers():
    """Numbers read and written with a decimal point while IfcOpenShell works: Qt sets the system's ``LC_NUMERIC``
    (a decimal comma in Italian) and IfcOpenShell then reads "1.5" as 1 (the program's launcher puts "C" back at
    start; a window opened later, or a script, may not have)."""
    try:
        old = locale.setlocale(locale.LC_NUMERIC)
    except locale.Error:  # pragma: no cover
        old = None
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
        yield
    finally:
        if old is not None:
            try:
                locale.setlocale(locale.LC_NUMERIC, old)
            except locale.Error:  # pragma: no cover
                pass


def _plain(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def properties(e: dict) -> dict:
    """The property set of an element: every scalar parameter, the fit, the source."""
    params = e.get("params") or {}
    method = params.get("method") or params.get("type") or params.get("opening_type") or ""
    props = {"Kind": e["kind"], "Method": str(method),
             "Name": e.get("name") or "",
             "Interpretation": t("Solido idealizzato dal modello fotogrammetrico misurato: lo scarto dice quanto il "
                                 "modello se ne discosta.", "An idealised solid from the measured photogrammetric "
                                 "model: the fit says how far the model is from it.")}
    for k, v in (e.get("params") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and np.isfinite(v):
            props["P_" + k] = float(v) if isinstance(v, float) else int(v)
        elif isinstance(v, str):
            props["P_" + k] = v
        elif v is not None and not isinstance(v, bool):
            props["P_" + k] = _plain(v)
    fit = e.get("fit") or {}
    for k in ("median", "p90", "max"):
        if fit.get(k) is not None:
            props[f"Fit_{k}_m"] = float(fit[k])
    props["Fit_points"] = int(fit.get("points") or 0)
    if e["kind"] in ("arch", "vault"):
        from .elements import candidates_text

        props["Fit_candidates"] = candidates_text(e.get("params") or {})
    if e.get("warnings"):
        props["Warnings"] = " ".join(e["warnings"])
    props["Frames"] = FRAMES
    # parametric for every kind: a raking or turning moulding's fixed-reference sweep is read back with the volume and
    # the bounding box of CadQuery's solid (tested), so nothing is written as a tessellation
    props["Representation"] = "parametric"
    if e["kind"] == "opening":
        props["Host"] = t("un muro sostitutivo «Muro (porzione misurata)», non un muro rilevato",
                          "a stand-in wall 'Wall (measured stretch)', not a surveyed wall")
    props["Source"] = _plain(e.get("source") or {})
    if e.get("notes"):
        props["Notes"] = e["notes"]
    return props


class _Writer:
    def __init__(self, model, body):
        self.m, self.body = model, body

    def p2(self, x, y):
        return self.m.createIfcCartesianPoint((float(x), float(y)))

    def p3(self, x, y, z):
        return self.m.createIfcCartesianPoint((float(x), float(y), float(z)))

    def d3(self, x, y, z):
        return self.m.createIfcDirection((float(x), float(y), float(z)))

    def axes(self, origin, z_axis=None, x_axis=None):
        return self.m.createIfcAxis2Placement3D(self.p3(*origin), None if z_axis is None else self.d3(*z_axis),
                                                None if x_axis is None else self.d3(*x_axis))

    def closed_polygon(self, pts):
        """A closed profile of straight lines (the first point repeated at the end)."""
        pts = [tuple(map(float, p)) for p in pts]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        curve = self.m.createIfcIndexedPolyCurve(self.m.createIfcCartesianPointList2D(pts), None, False)
        return self.m.createIfcArbitraryClosedProfileDef("AREA", None, curve)

    def closed_curve(self, outline: dict):
        """A closed profile of lines and three-point arcs (the engine's outline: indices from 0)."""
        pts = [tuple(map(float, p)) for p in outline["points"]]
        segs = []
        for s in outline["segments"]:
            idx = tuple(int(i) + 1 for i in s[1:])
            segs.append(self.m.createIfcLineIndex(idx) if s[0] == "L" else self.m.createIfcArcIndex(idx))
        curve = self.m.createIfcIndexedPolyCurve(self.m.createIfcCartesianPointList2D(pts), segs, False)
        return self.m.createIfcArbitraryClosedProfileDef("AREA", None, curve)

    def representation(self, items, kind="SweptSolid"):
        return self.m.createIfcShapeRepresentation(self.body, "Body", kind, items)


def _placement_axes(w: _Writer, matrix):
    m = np.asarray(matrix, float)
    return w.axes(m[:3, 3], m[:3, 2], m[:3, 0])


def solid_for(w: _Writer, e: dict):
    """``(representation, local placement axes)`` of an element's solid."""
    g = e["geometry"]
    if e["kind"] == "column":
        prof = np.asarray(g["profile"], float)
        pts = [(0.0, prof[0, 1])] + [tuple(p) for p in prof] + [(0.0, prof[-1, 1])]
        area = w.closed_polygon(pts)
        # the profile plane: x = r (local x), y = z (local z); the axis is the profile's y axis
        solid = w.m.createIfcRevolvedAreaSolid(area, w.axes((0, 0, 0), (0, -1, 0), (1, 0, 0)),
                                               w.m.createIfcAxis1Placement(w.p3(0, 0, 0), w.d3(0, 1, 0)), 2 * np.pi)
        return w.representation([solid]), _placement_axes(w, g["placement"])
    if e["kind"] == "moulding":
        path = np.asarray(g["path"], float)
        sec = np.asarray(g["section"], float)
        t0 = path[1] - path[0]
        t0 /= np.linalg.norm(t0)
        up = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
        u0 = np.cross(up, t0)
        u0 /= np.linalg.norm(u0)
        v0 = np.cross(t0, u0)
        if len(path) == 2:
            area = w.closed_polygon(sec)
            solid = w.m.createIfcExtrudedAreaSolid(area, w.axes((0, 0, 0)), w.d3(0, 0, 1),
                                                   float(np.linalg.norm(path[1] - path[0])))
            return w.representation([solid]), w.axes(path[0], t0, u0)
        # IfcFixedReferenceSweptAreaSolid: the profile's x follows the fixed reference (up), its y = tangent x up
        # = -u: a section point (u, v) is (v, -u) of the profile
        area = w.closed_polygon(np.column_stack([sec[:, 1], -sec[:, 0]]))
        rel = path - path[0]
        points = w.m.createIfcCartesianPointList3D([tuple(map(float, p)) for p in rel])
        directrix = w.m.createIfcIndexedPolyCurve(points, None, False)
        solid = w.m.createIfcFixedReferenceSweptAreaSolid(area, w.axes((0, 0, 0)), directrix, None, None,
                                                          w.d3(*(v0 if abs(t0[2]) > 0.999 else up)))
        return w.representation([solid], "AdvancedSweptSolid"), w.axes(path[0])
    if e["kind"] == "opening":
        bottom, top = float(g["bottom"]), float(g["top"])
        items = [w.m.createIfcExtrudedAreaSolid(w.closed_polygon(g["plan"]), w.axes((0, 0, bottom)), w.d3(0, 0, 1),
                                                top - bottom)]
        if g.get("void"):
            vd = g["void"]
            items.append(w.m.createIfcExtrudedAreaSolid(w.closed_curve(vd["outline"]),
                                                        _placement_axes(w, vd["placement"]), w.d3(0, 0, 1),
                                                        float(vd["depth"])))
        return w.representation(items), w.axes((0, 0, 0))
    area = w.closed_curve(g["outline"])
    solid = w.m.createIfcExtrudedAreaSolid(area, w.axes((0, 0, 0)), w.d3(0, 0, 1), float(g["depth"]))
    return w.representation([solid]), _placement_axes(w, g["placement"])


IFC_CLASS = {"column": ("IfcColumn", "COLUMN", None), "moulding": ("IfcMember", "USERDEFINED", "Moulding"),
             "arch": ("IfcBuildingElementProxy", "USERDEFINED", "Arch"),
             "vault": ("IfcBuildingElementProxy", "USERDEFINED", "Vault"),
             "opening": ("IfcOpeningElement", "OPENING", None)}
FILLING = {"door": ("IfcDoor", "DOOR"), "window": ("IfcWindow", "WINDOW")}


HOST_NAME = ("Muro (porzione misurata)", "Wall (measured stretch)")
HOST_NOTE = ("Muro sostitutivo: la porzione del muro misurata intorno alle aperture (spessore misurato o dato, mezzo "
             "metro oltre ogni stipite, dal suolo a mezzo metro sopra l'architrave), perché l'IFC vuole che "
             "un'apertura sia in un muro. Non è un elemento rilevato: il muro vero lo farà il plugin delle murature.",
             "A stand-in wall: the stretch of wall measured around the openings (its thickness measured or given, half "
             "a metre beyond each reveal, from the ground to half a metre over the lintel), because IFC wants an "
             "opening in a wall. Not a surveyed element: the real wall will come from the masonry plugin.")


def host_slab(e: dict, origin) -> dict:
    """The stand-in wall around an opening (export frame): its plan (the two faces, half a metre beyond the reveals
    at the wider face), heights (the ground of the DEM at the threshold, else half a metre under the sill, to half a
    metre over the lintel), its axis and thickness."""
    from shapely.geometry import Polygon

    from . import engine

    g, p = e["geometry"], e["params"]
    lines = g["lines"]
    a = np.asarray(lines["axis"], float)
    outer = (np.asarray(lines["outer"][0], float), np.asarray(lines["outer"][1], float))
    inner = (np.asarray(lines["inner"][0], float), np.asarray(lines["inner"][1], float))
    mid = 0.5 * (engine.intersect_lines(*map(np.asarray, lines["left"]), *outer)
                 + engine.intersect_lines(*map(np.asarray, lines["right"]), *outer))
    half = max(p["width_outer"], p["width_inner"]) / 2 + 0.5
    d = outer[1] / np.linalg.norm(outer[1])
    corners = []
    for s_ in (-half, half):
        q = mid + s_ * d
        corners.append((q, engine.intersect_lines(q, a, *inner)))
    poly = np.array([corners[0][0], corners[1][0], corners[1][1], corners[0][1]])
    o = np.asarray(origin, float)
    bottom = (p["ground"] if p.get("ground") is not None else p["sill"] - 0.5) - o[2]
    top = p["lintel"] + 0.5 - o[2]
    return {"polygon": Polygon(poly).buffer(0), "z": (float(min(bottom, p["sill"] - o[2])), float(top)), "axis": a,
            "thickness": float(p["reveal_depth"])}


def _host_groups(slabs: list[dict]) -> list[list[int]]:
    """The openings that share a stand-in wall: slabs that overlap, across the same wall (axes parallel, thickness
    within 2 cm)."""
    groups = [[k] for k in range(len(slabs))]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = slabs[groups[i][0]], slabs[groups[j][0]]
                if abs(a["axis"] @ b["axis"]) > 0.99 and abs(a["thickness"] - b["thickness"]) < 0.02 and \
                        any(slabs[x]["polygon"].intersects(slabs[y]["polygon"]) for x in groups[i] for y in groups[j]):
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    return groups


def _hosts(model, w: "_Writer", site_e, openings, origin, style) -> None:
    """A stand-in ``IfcWall`` for the openings (shared by the openings of one stretch of wall), voided by each
    ``IfcOpeningElement`` and filled by its ``IfcDoor`` / ``IfcWindow``."""
    import ifcopenshell.api.feature
    import ifcopenshell.api.geometry
    import ifcopenshell.api.pset
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    from shapely.ops import unary_union

    slabs = [host_slab(e, origin) for e, _o, _f in openings]
    for group in _host_groups(slabs):
        shape = unary_union([slabs[k]["polygon"] for k in group]).convex_hull
        z0 = min(slabs[k]["z"][0] for k in group)
        z1 = max(slabs[k]["z"][1] for k in group)
        wall = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=t(*HOST_NAME),
                                                   predefined_type="NOTDEFINED")
        wall.Description = t(*HOST_NOTE)
        ifcopenshell.api.spatial.assign_container(model, products=[wall], relating_structure=site_e)
        solid = model.createIfcExtrudedAreaSolid(w.closed_polygon(np.asarray(shape.exterior.coords)[:-1]),
                                                 w.axes((0, 0, z0)), w.d3(0, 0, 1), z1 - z0)
        rep = w.representation([solid])
        ifcopenshell.api.geometry.assign_representation(model, product=wall, representation=rep)
        wall.ObjectPlacement = model.createIfcLocalPlacement(None, w.axes((0, 0, 0)))
        if style is not None:
            style(rep, wall.Name, (0.75, 0.75, 0.72))
        names = [openings[k][0]["name"] for k in group]
        pset = ifcopenshell.api.pset.add_pset(model, product=wall, name=PSET)
        ifcopenshell.api.pset.edit_pset(model, pset=pset, properties={
            "Kind": "host", "Name": wall.Name, "Interpretation": t(*HOST_NOTE), "Openings": ", ".join(names),
            "P_thickness": float(np.mean([slabs[k]["thickness"] for k in group])), "Frames": FRAMES})
        for k in group:
            _e, opening, filling = openings[k]
            opening.ObjectPlacement = model.createIfcLocalPlacement(wall.ObjectPlacement, w.axes((0, 0, 0)))
            ifcopenshell.api.feature.add_feature(model, feature=opening, element=wall)
            if filling is not None:
                ifcopenshell.api.feature.add_filling(model, opening=opening, element=filling)


def write(path, entries: list[dict], origin, crs: str | None, mesh=None, project: str = "Rilievo", site: str = "Sito",
          colours=None, progress=None) -> Path:
    """Write the elements ``entries`` (in the export frame of ``origin``) as IFC4 in ``path``; ``mesh`` (a trimesh in
    the export frame) is written beside them as ``IfcGeographicElement``."""
    with c_numbers():
        return _write(path, entries, origin, crs, mesh, project, site, colours, progress)


def _write(path, entries, origin, crs, mesh, project, site, colours, progress) -> Path:
    from pyarchinit_photogrammetry.core import crs as crs_mod
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    _require()
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.georeference
    import ifcopenshell.api.pset
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.style
    import ifcopenshell.api.unit
    import ifcopenshell.guid

    progress(0.05, t("IFC: progetto, unità e georeferenziazione", "IFC: project, units and georeference"))
    origin = np.asarray(origin, float)
    model = ifcopenshell.file(schema="IFC4")
    proj = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=project)
    ifcopenshell.api.unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"},
                                      area={"is_metric": True, "raw": "METERS"},
                                      volume={"is_metric": True, "raw": "METERS"})
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=ctx)
    if not crs_mod.is_local(crs):
        ifcopenshell.api.georeference.add_georeferencing(model)
        # a compound system (heights above a geoid): its plan and its vertical datum, as the program writes them
        # (programs before 1.6 have neither function: the system as it is)
        epsg = crs_mod.epsg_code(crs)
        vertical = getattr(crs_mod, "vertical_part", lambda c: None)(crs)
        plan = getattr(crs_mod, "horizontal", lambda c: c)(crs)
        ifcopenshell.api.georeference.edit_georeferencing(
            model,
            projected_crs={"Name": f"EPSG:{epsg}" if epsg else str(plan),
                           **({"VerticalDatum": vertical} if vertical else {})},
            coordinate_operation={"Eastings": float(origin[0]), "Northings": float(origin[1]),
                                  "OrthogonalHeight": float(origin[2]), "XAxisAbscissa": 1.0, "XAxisOrdinate": 0.0,
                                  "Scale": 1.0})
    site_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name=site)
    ifcopenshell.api.aggregate.assign_object(model, products=[site_e], relating_object=proj)
    w = _Writer(model, body)

    def style(rep, name, colour):
        st = ifcopenshell.api.style.add_style(model, name=f"{name}_style")
        ifcopenshell.api.style.add_surface_style(model, style=st, ifc_class="IfcSurfaceStyleShading",
                                                 attributes={"SurfaceColour": {"Name": None, "Red": colour[0],
                                                                               "Green": colour[1], "Blue": colour[2]}})
        ifcopenshell.api.style.assign_representation_styles(model, shape_representation=rep, styles=[st])

    openings = []  # (entry, the opening element, its door or window or None)
    for k, e in enumerate(entries):
        progress(0.1 + 0.8 * k / max(len(entries), 1), t(f"IFC: elemento {k + 1}/{len(entries)} - {e['name']}",
                                                         f"IFC: element {k + 1}/{len(entries)} - {e['name']}"))
        cls, ptype, otype = IFC_CLASS[e["kind"]]
        classes = [(cls, ptype, otype)]
        kind = (e.get("params") or {}).get("opening_type") if e["kind"] == "opening" else None
        if kind in FILLING:  # the door or the window itself, with the same solid, beside the opening
            classes.append((*FILLING[kind], None))
        made = []
        for cls, ptype, otype in classes:
            prod = ifcopenshell.api.root.create_entity(model, ifc_class=cls, name=e["name"], predefined_type=ptype)
            made.append(prod)
            if otype:
                prod.ObjectType = otype
            prod.Description = e.get("notes") or None
            if cls in ("IfcDoor", "IfcWindow"):
                # the clear width: where a leaf passes (the narrowest, between the innermost runs of the reveals)
                prod.OverallWidth = float(e["params"].get("clear_width") or e["params"]["width"])
                prod.OverallHeight = float(e["params"]["height"])
            if cls != "IfcOpeningElement":  # an opening is in the wall it voids, not in the site
                ifcopenshell.api.spatial.assign_container(model, products=[prod], relating_structure=site_e)
            rep, axes = solid_for(w, e)
            ifcopenshell.api.geometry.assign_representation(model, product=prod, representation=rep)
            prod.ObjectPlacement = model.createIfcLocalPlacement(None, axes)
            if colours:
                style(rep, e["name"], colours[k % len(colours)])
            pset = ifcopenshell.api.pset.add_pset(model, product=prod, name=PSET)
            ifcopenshell.api.pset.edit_pset(model, pset=pset, properties=properties(e))
            volume = (e.get("solid") or {}).get("volume")
            if volume:
                qs = [model.create_entity("IfcQuantityVolume", Name="NetVolume", VolumeValue=float(volume))]
                area = (e.get("solid") or {}).get("area")
                if area:
                    qs.append(model.create_entity("IfcQuantityArea", Name="NetSurfaceArea", AreaValue=float(area)))
                qto = model.create_entity("IfcElementQuantity", GlobalId=ifcopenshell.guid.new(), Name=QTO,
                                          MethodOfMeasurement=METHOD, Quantities=qs)
                model.create_entity("IfcRelDefinesByProperties", GlobalId=ifcopenshell.guid.new(),
                                    RelatedObjects=[prod], RelatingPropertyDefinition=qto)
        if e["kind"] == "opening":
            openings.append((e, made[0], made[1] if len(made) > 1 else None))
    if openings:
        progress(0.9, t("IFC: i muri che ospitano le aperture", "IFC: the walls that host the openings"))
        _hosts(model, w, site_e, openings, origin, style if colours else None)
    if mesh is not None:
        progress(0.92, t("IFC: il modello del rilievo", "IFC: the survey model"))
        prod = ifcopenshell.api.root.create_entity(model, ifc_class="IfcGeographicElement",
                                                   name=t("Modello del rilievo", "Survey model"),
                                                   predefined_type="TERRAIN")
        ifcopenshell.api.spatial.assign_container(model, products=[prod], relating_structure=site_e)
        pts = model.createIfcCartesianPointList3D([tuple(map(float, p)) for p in np.asarray(mesh.vertices, float)])
        faces = model.createIfcTriangulatedFaceSet(pts, None, None,
                                                   [tuple(int(i) + 1 for i in f) for f in np.asarray(mesh.faces)], None)
        rep = model.createIfcShapeRepresentation(body, "Body", "Tessellation", [faces])
        ifcopenshell.api.geometry.assign_representation(model, product=prod, representation=rep)
        prod.ObjectPlacement = model.createIfcLocalPlacement(None, w.axes((0, 0, 0)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))
    progress(1.0, t("IFC scritto", "IFC written"))
    return path


def export(chunk, path, ids=None, with_mesh: bool = False, progress=None) -> Path:
    """The elements ``ids`` of the chunk (all when None) as IFC4 in ``path``, the chunk's model beside them with
    ``with_mesh``."""
    from . import elements

    elements.check_frame(chunk)
    chosen = [elements.need(chunk, i) for i in ids] if ids else elements.elements(chunk)
    if not chosen:
        raise elements.NotReady(t("Nessun elemento da esportare.", "No element to export."))
    elements.check_fresh(chosen)
    mesh = None
    if with_mesh:
        import trimesh

        v, f = elements.local_mesh(chunk)
        mesh = trimesh.Trimesh(v, f, process=False)
    project = getattr(getattr(chunk, "document", None), "path", None)
    project = Path(project).stem if project else "Rilievo"
    return write(path, chosen, elements.origin(chunk), chunk.crs, mesh=mesh, project=project,
                 site=chunk.meta.get("sito") or (chunk.document.meta.get("sito") if chunk.document else None) or "Sito",
                 colours=elements.COLOURS, progress=progress)


def read_volumes(path) -> dict:
    """Volume and bounding box (world-of-file coordinates) of every element of an IFC as ``ifcopenshell.geom``
    tessellates it: ``{name: (volume, bbox)}`` (the round trip of the tests); also under ``"IfcClass:name"`` (an
    opening and its door share the name)."""
    with c_numbers():
        return _read_volumes(path)


def _read_volumes(path) -> dict:
    import ifcopenshell
    import ifcopenshell.geom
    import trimesh

    f = ifcopenshell.open(str(path))
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    out = {}
    for prod in f.by_type("IfcProduct"):
        if not prod.Representation or prod.is_a("IfcGeographicElement"):
            continue
        sh = ifcopenshell.geom.create_shape(s, prod)
        v = np.asarray(sh.geometry.verts, float).reshape(-1, 3)
        fa = np.asarray(sh.geometry.faces, int).reshape(-1, 3)
        m = trimesh.Trimesh(v, fa, process=False)
        value = (abs(float(m.volume)), np.r_[v.min(axis=0), v.max(axis=0)])
        out.setdefault(prod.Name, value)
        out[f"{prod.is_a()}:{prod.Name}"] = value
    return out
