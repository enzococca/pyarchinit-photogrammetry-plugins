"""The campaigns in IFC 4.3 (``IFC4X3_ADD2``, ISO 16739-1:2024): each mapped feature an ``IfcSurfaceFeature``
(``PredefinedType`` DEFECT for decay, cracks and notes, TREATMENT for interventions) adhering to its host wall
(``IfcRelAdheresToElement``, 4.3 only), with the program's conventions (metres, ``IfcMapConversion`` on the floor of
the minimum when the chunk has a CRS, the walls in an ``IfcSite``).

The host of an elevation is the masonry plugin's wall when the face belongs to one and that plugin is installed (its
slab rebuilt by the same rules, with the same GlobalId); else a stand-in ``IfcWall``: behind a plane a slab of the
thickness given (``IfcExtrudedAreaSolid``), on a cylinder a faceted band of the surface over the face's bounds, as
thick (``IfcFacetedBrep``). A section is a cut, not a face: refused. A feature's shape is the documented shell
concept: ``Body`` / ``SurfaceModel``, an ``IfcShellBasedSurfaceModel`` of one ``IfcOpenShell`` of the triangles of
its polygon lifted on the surface through the relief (a crack the band of its opening, a note a disc), placed
relative to the host. No orthophoto, no texture.

The GlobalIds follow the rule of the plugin suite (:func:`stable_global_id`, the same constant in the masonry
plugin): the host ``(chunk key, wall id or "face", elevation id)``, a feature ``(chunk key, elevation id, feature
id)``."""
from __future__ import annotations

import locale
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import records
from .records import NotReady
from .texts import lang, num, t

SUITE_NAMESPACE = uuid.UUID("6f1c2b0e-5a1d-4c8e-9b7a-3d2e1f0c9b8a")  # the plugin suite's: the same in every plugin
STEP = 0.10  # metres: the grid of inner points of a lifted polygon
SAGITTA = 0.002  # metres: the most a facet of a cylinder's band leaves its surface
SECTION = ("una sezione è un taglio, non una faccia", "a section is a cut, not a face")


def stable_global_id(*parts) -> str:
    """The IFC GlobalId of the parts joined by ``/`` (a UUID 5 in the suite's namespace, compressed)."""
    import ifcopenshell.guid

    return ifcopenshell.guid.compress(uuid.uuid5(SUITE_NAMESPACE, "/".join(str(p) for p in parts)).hex)


@contextmanager
def c_numbers():
    """Numbers read and written with a decimal point while IfcOpenShell works: Qt sets the system's ``LC_NUMERIC``
    (a decimal comma in Italian) and IfcOpenShell then reads "1.5" as 1."""
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


# ---------------------------------------------------------------- the hosts
def _bounds(chunk, entry) -> tuple[float, float, float, float]:
    """``(u0, v0, u1, v1)`` of the face: its bounds, else its orthophoto's extent."""
    if entry.get("bounds"):
        return tuple(float(x) for x in entry["bounds"])
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    with rasterio.open(elevation_paths(chunk, entry)[0]) as src:
        b = src.bounds
    return float(b.left), float(b.bottom), float(b.right), float(b.top)


def _slab(gid, name, wall_id, outline, origin, u, into, thickness) -> dict:
    """A host slab: the ``outline`` (``(u, v)``) extruded ``thickness`` along ``into`` from ``origin``."""
    corners = [origin + x * u + y * np.cross(into, u) * -1.0 for x, y in np.asarray(outline.exterior.coords)]
    corners = np.asarray(corners)
    return {"gid": gid, "name": name, "wall": wall_id, "kind": "slab", "outline": outline,
            "origin": np.asarray(origin, np.float64), "u": np.asarray(u, np.float64),
            "into": np.asarray(into, np.float64), "thickness": float(thickness),
            "points": np.vstack([corners, corners + thickness * into])}


def _masonry_host(chunk, entry, thickness, warnings) -> dict | None:
    """The masonry plugin's slab holding the face, when the plugin is installed and the face is one of a wall's."""
    try:
        from ppg_masonry import ifc as mifc
        from ppg_masonry import walls as mwalls
    except ImportError:
        return None
    w = mwalls.wall_of_face(chunk, entry["id"])
    if w is None or (entry.get("frame") or {}).get("kind") != "plane":
        return None
    try:
        solids = mifc.wall_solids(chunk, w["id"], thickness)
    except Exception as exc:  # noqa: BLE001 - the masonry plugin says why in its own words
        warnings.append(t(f"«{entry.get('name')}»: il muro «{w['name']}» del plugin murature non dà la sua lastra "
                          f"({exc}): parete di appoggio", f"«{entry.get('name')}»: the masonry plugin's wall "
                          f"«{w['name']}» gives no slab ({exc}): host wall stand-in"))
        return None
    for solid in solids:
        if entry["id"] in (solid["entry"]["id"], (solid.get("other") or {}).get("id")):
            name = w["name"] if len(solids) == 1 else f"{w['name']} - {solid['entry'].get('name', '')}"
            outline = solid["outline"]
            if outline.geom_type != "Polygon":
                outline = max(outline.geoms, key=lambda g: g.area)
            host = _slab(mifc.solid_global_id(chunk, w["id"], solid, len(solids)), name, w["id"], outline,
                         solid["origin"], solid["u"], solid["into"], solid["thickness"])
            host["outline"] = solid["outline"]
            return host
    return None


def _stand_in_name(entry) -> str:
    return t(f"{entry.get('name') or entry['id']} (parete di appoggio)", f"{entry.get('name') or entry['id']} "
             "(host wall)")


def _cylinder_band(chunk, entry, frame, gid, thickness, warnings) -> dict:
    """The stand-in of a cylinder: its surface over the face's bounds and as much behind it, faceted so that no
    facet leaves the surface by more than ``SAGITTA``; ``points`` (n+1, 2, 2, 3) indexed by u, v, depth."""
    u0, v0, u1, v1 = _bounds(chunk, entry)
    if not frame.inside and thickness >= 0.9 * frame.radius:
        thickness = 0.9 * frame.radius
        warnings.append(t(f"«{entry.get('name')}»: spessore ridotto a {num(thickness)} m, meno del raggio",
                          f"«{entry.get('name')}»: thickness reduced to {num(thickness)} m, less than the radius"))
    outer = frame.radius + (thickness if frame.inside else 0.0)
    step = 2.0 * np.arccos(1.0 - SAGITTA / outer)
    t0, t1 = float(frame._turn(u0)), float(frame._turn(u1))
    n = max(2, int(np.ceil(abs(t1 - t0) / step)))
    us = frame._u(np.linspace(t0, t1, n + 1))
    uu, vv, dd = np.meshgrid(us, [v0, v1], [0.0, -thickness], indexing="ij")
    pts = frame.point(uu.ravel(), vv.ravel(), dd.ravel()).reshape(n + 1, 2, 2, 3)
    return {"gid": gid, "name": _stand_in_name(entry), "wall": None, "kind": "band", "grid": pts,
            "thickness": float(thickness), "points": pts.reshape(-1, 3)}


def _host_of(chunk, entry, thickness, warnings) -> dict:
    from .geometry import frame_of

    host = _masonry_host(chunk, entry, thickness, warnings)
    if host is not None:
        return host
    wall_id = "face"
    try:
        from ppg_masonry import walls as mwalls

        w = mwalls.wall_of_face(chunk, entry["id"])
        if w is not None:  # a face of a masonry wall the masonry plugin does not make a slab of (a cylinder)
            wall_id = w["id"]
    except ImportError:
        pass
    gid = stable_global_id(chunk.key, wall_id, entry["id"])
    frame = frame_of(chunk, entry)
    if frame.kind == "cylinder":
        return _cylinder_band(chunk, entry, frame, gid, thickness, warnings)
    from shapely.geometry import box

    return _slab(gid, _stand_in_name(entry), None, box(*_bounds(chunk, entry)), frame.origin, frame.u_axis,
                 -frame.normal, thickness)


# ---------------------------------------------------------------- the features
def _shape_of(f: dict):
    """The polygon a feature's shell is made of: an area as it is, a line the band of its opening, a point a disc."""
    from .geometry import as_geometry, band, disc

    g = as_geometry(f["geometry"])
    if g.geom_type in ("LineString", "MultiLineString"):
        return band(g, (f.get("width_mm") or 0.0) / 1000.0)
    if g.geom_type in ("Point", "MultiPoint"):
        return disc(g)
    return g


def _standard(code: str, names: dict) -> str:
    if names.get("treatment"):
        return "intervention"
    if code.startswith("ISCS."):
        return "ICOMOS-ISCS"
    if code.startswith("UNI."):
        return "UNI 11182"
    return "local"


def _clean(props: dict) -> dict:
    return {k: v for k, v in props.items() if v is not None and v != ""}


def _feature_props(f: dict, c: dict, names: dict) -> dict:
    from .outputs import _links_text, _photos_text

    tr = f.get("treatment") or {}
    line = (f.get("geometry") or "").upper().startswith(("LINESTRING", "MULTILINESTRING"))
    return _clean({
        "Standard": _standard(f["code"], names), "Code": f["code"], "CodeUNI": f.get("code_uni") or names["uni"],
        "Family": names["family"], "TermEN": names["en"], "TermIT": names["it"], "Kind": f["kind"],
        "Intensity": f.get("intensity"),
        ("Length_m" if line else "Extent_m2"): None if f.get("extent") is None else round(float(f["extent"]), 6),
        "Width_mm": f.get("width_mm"), "Depth_mm": f.get("depth_mm"), "Cause": f.get("cause"),
        "UrgencyClass": f.get("urgency"), "Campaign": c.get("name"), "Date": c.get("date"),
        "Surveyor": c.get("surveyor"), "Source": f.get("source"), "Photos": _photos_text(f),
        "Technique": tr.get("technique"), "Materials": tr.get("materials"),
        "Done": bool(tr.get("done")) if f["kind"] == "treatment" else None, "US": _links_text(f)})


def _host_classes(chunk, c: dict) -> tuple[dict, dict]:
    """``(Pset_Condition, Pset_PyArchInit_Conservation)`` of a host from its campaign ``c``."""
    classes = c.get("classes") or {}
    cc, uc = classes.get("cc"), classes.get("uc")
    rc_rec = records.assessment(chunk, c["date"]) or {}
    rc = rc_rec.get("rc")
    condition = ", ".join(x for x in (f"CC {cc}" if cc is not None else "", f"UC {uc}" if uc is not None else "")
                          if x)
    description = "; ".join(x for x in (
        f"CC {cc}" + (f" ({classes['reason']})" if classes.get("reason") else "") if cc is not None else "",
        f"RC {rc}" + (f" ({rc_rec['reason']})" if rc_rec.get("reason") else "") if rc is not None else "") if x)
    cond = _clean({"AssessmentDate": c["date"], "AssessmentCondition": condition, "AssessmentType": "EN 16096",
                   "AssessmentDescription": description})
    mine = _clean({"ConditionClass": cc, "UrgencyClass": uc, "RecommendationClass": rc, "Campaign": c.get("name"),
                   "Date": c.get("date"), "Surveyor": c.get("surveyor"), "Method": c.get("method")})
    return cond, mine


# ---------------------------------------------------------------- the file
def export(chunk, cids: list[str], path, thickness: float = 0.5, project: str = "Rilievo", site=None) -> dict:
    """The campaigns ``cids`` in one IFC 4.3 file at ``path``: a host wall per elevation, the accepted features
    adhering to it (those still to be named left out and counted). Returns ``{"path", "hosts", "features",
    "warnings"}``; NotReady when no campaign is on a face (a section is refused)."""
    from .geometry import lift
    from .outputs import _context, _order, glossary_names

    if thickness is None or float(thickness) <= 0:
        raise ValueError(t("Lo spessore della parete di appoggio è una lunghezza positiva.",
                           "The host wall's thickness is a positive length."))
    thickness = float(thickness)
    warnings: list[str] = []
    by_face: dict[str, list] = {}
    sections = []
    for cid in dict.fromkeys(cids):
        ctx = _context(chunk, cid)
        entry = ctx["entry"]
        if entry.get("kind") == "section":
            sections.append(entry.get("name") or entry["id"])
            continue
        by_face.setdefault(entry["id"], []).append(ctx)
        if ctx["unnamed"]:
            warnings.append(t(f"«{ctx['campaign']['name']}»: {ctx['unnamed']} elementi da nominare esclusi",
                              f"«{ctx['campaign']['name']}»: {ctx['unnamed']} features to be named left out"))
    for name in dict.fromkeys(sections):
        warnings.append(t(f"«{name}»: {SECTION[0]}: i suoi elementi vanno nei GeoPackage e nel DXF",
                          f"«{name}»: {SECTION[1]}: its features go to the GeoPackages and the DXF"))
    if not by_face:
        if sections:
            raise NotReady(t(f"IFC: {SECTION[0]} ({', '.join(dict.fromkeys(sections))}): i suoi elementi vanno nei "
                             "GeoPackage e nel DXF.", f"IFC: {SECTION[1]} ({', '.join(dict.fromkeys(sections))}): its "
                             "features go to the GeoPackages and the DXF."))
        raise NotReady(t("Nessuna campagna da esportare.", "No campaign to export."))

    hosts: dict[str, dict] = {}
    lg = lang()
    for eid, ctxs in by_face.items():
        entry = ctxs[0]["entry"]
        host = _host_of(chunk, entry, thickness, warnings)
        host = hosts.setdefault(host["gid"], dict(host, campaigns=[], features=[]))
        on_plane = 0
        for ctx in ctxs:
            c, vocab = ctx["campaign"], ctx["vocab"]
            host["campaigns"].append(c)
            for f in _order(ctx):
                verts, tris, flat = lift(chunk, entry, _shape_of(f), STEP)
                if not len(tris):
                    warnings.append(t(f"Elemento {f['id']} senza area: escluso", f"Feature {f['id']} with no area: "
                                      "left out"))
                    continue
                on_plane += flat
                names = glossary_names(vocab, f["code"], f.get("code_iscs"))
                cc = f.get("condition") if f.get("condition") is not None else (c.get("classes") or {}).get("cc")
                n = len(host["features"]) + 1
                host["features"].append({
                    "gid": stable_global_id(chunk.key, eid, f["id"]), "record": f, "campaign": c,
                    "name": f"{vocab.name(f['code'], lg) if f['code'] in vocab else f['code']} {n}",
                    "verts": verts, "tris": tris, "names": names, "condition": cc})
        if on_plane:
            where = entry.get("name") or entry["id"]
            warnings.append(t(f"«{where}»: {on_plane} vertici sul piano del prospetto, dove il rilievo non ha valore",
                              f"«{where}»: {on_plane} vertices on the elevation's plane, where the relief has no "
                              "value"))

    from pyarchinit_photogrammetry.core import crs as crs_mod

    local = crs_mod.is_local(chunk.crs)
    every = np.vstack([h["points"] for h in hosts.values()] + [f["verts"] for h in hosts.values()
                                                                  for f in h["features"]])
    origin = np.zeros(3) if local else np.floor(every.min(axis=0))
    with c_numbers():
        n_feat = _write(chunk, Path(path), hosts, origin, local, project, site)
    return {"path": Path(path), "hosts": len(hosts), "features": n_feat, "warnings": warnings}


def _write(chunk, path: Path, hosts: dict, origin, local: bool, project, site) -> int:
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.georeference
    import ifcopenshell.api.pset
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit

    from pyarchinit_photogrammetry.core import crs as crs_mod

    model = ifcopenshell.file(schema="IFC4X3_ADD2")
    proj = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=project or "Rilievo")
    ifcopenshell.api.unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"},
                                      area={"is_metric": True, "raw": "METERS"},
                                      volume={"is_metric": True, "raw": "METERS"})
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=ctx)
    if not local:
        ifcopenshell.api.georeference.add_georeferencing(model)
        epsg, vertical = crs_mod.epsg_code(chunk.crs), crs_mod.vertical_part(chunk.crs)
        ifcopenshell.api.georeference.edit_georeferencing(
            model, projected_crs={"Name": f"EPSG:{epsg}" if epsg else str(crs_mod.horizontal(chunk.crs)),
                                  **({"VerticalDatum": vertical} if vertical else {})},
            coordinate_operation={"Eastings": float(origin[0]), "Northings": float(origin[1]),
                                  "OrthogonalHeight": float(origin[2]), "XAxisAbscissa": 1.0, "XAxisOrdinate": 0.0,
                                  "Scale": 1.0})
    doc = getattr(chunk, "document", None)
    site_name = site or ((doc.meta.get("sito") if doc is not None else None) or "Sito")
    site_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name=site_name)
    ifcopenshell.api.aggregate.assign_object(model, products=[site_e], relating_object=proj)

    def direction(v):
        return model.createIfcDirection(tuple(float(x) for x in v))

    def point(p):
        return model.createIfcCartesianPoint(tuple(float(x) for x in p))

    def faces(loops):
        return [model.createIfcFace([model.createIfcFaceOuterBound(model.createIfcPolyLoop(loop), True)])
                for loop in loops]

    def psets(element, sets: dict):
        for name, props in sets.items():
            if props:
                ps = ifcopenshell.api.pset.add_pset(model, product=element, name=name)
                ifcopenshell.api.pset.edit_pset(model, pset=ps, properties=props)

    count = 0
    for host in hosts.values():
        wall = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=host["name"],
                                                   predefined_type="SOLIDWALL" if host["wall"] else "NOTDEFINED")
        wall.GlobalId = host["gid"]
        ifcopenshell.api.spatial.assign_container(model, products=[wall], relating_structure=site_e)
        if host["kind"] == "slab":
            loc, x_axis, z_axis = host["origin"] - origin, host["u"], host["into"]
            wall.ObjectPlacement = model.createIfcLocalPlacement(None, model.createIfcAxis2Placement3D(
                point(loc), direction(z_axis), direction(x_axis)))
            items = []
            for poly in getattr(host["outline"], "geoms", [host["outline"]]):
                if poly.geom_type != "Polygon" or poly.is_empty:
                    continue
                # local x = u, local z = into the wall, local y = z x x = -v: the outline as (u, -v)
                ring = [(float(x), float(-y)) for x, y in np.asarray(poly.exterior.coords)[:-1]]
                curve = model.createIfcIndexedPolyCurve(model.createIfcCartesianPointList2D(ring + [ring[0]]))
                profile = model.createIfcArbitraryClosedProfileDef("AREA", None, curve)
                items.append(model.createIfcExtrudedAreaSolid(profile, None, direction((0.0, 0.0, 1.0)),
                                                              float(host["thickness"])))
            rep = model.createIfcShapeRepresentation(body, "Body", "SweptSolid", items)
        else:
            loc, x_axis, z_axis = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
            wall.ObjectPlacement = model.createIfcLocalPlacement(None, model.createIfcAxis2Placement3D(
                point(loc), None, None))
            g = host["grid"] - origin
            pts = [[[point(g[i, j, k]) for k in range(2)] for j in range(2)] for i in range(len(g))]
            loops = []
            for i in range(len(g) - 1):
                a, b = pts[i], pts[i + 1]
                loops += [[a[0][0], b[0][0], b[1][0], a[1][0]], [a[0][1], a[1][1], b[1][1], b[0][1]],
                          [a[0][0], a[0][1], b[0][1], b[0][0]], [a[1][0], b[1][0], b[1][1], a[1][1]]]
            first, last = pts[0], pts[-1]
            loops += [[first[0][0], first[1][0], first[1][1], first[0][1]],
                      [last[0][0], last[0][1], last[1][1], last[1][0]]]
            brep = model.createIfcFacetedBrep(model.createIfcClosedShell(faces(loops)))
            rep = model.createIfcShapeRepresentation(body, "Body", "Brep", [brep])
        wall.Representation = model.createIfcProductDefinitionShape(None, None, [rep])
        latest = max(host["campaigns"], key=lambda c: (c.get("date") or "", c.get("made") or ""))
        cond, mine = _host_classes(chunk, latest)
        psets(wall, {"Pset_Condition": cond, "Pset_PyArchInit_Conservation": mine})

        # the features: their shells in the host's axes, placed relative to it
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])
        related = []
        for f in host["features"]:
            rec, c = f["record"], f["campaign"]
            kind = "TREATMENT" if rec["kind"] == "treatment" else "DEFECT"
            feat = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSurfaceFeature", name=f["name"],
                                                       predefined_type=kind)
            feat.GlobalId = f["gid"]
            feat.ObjectType = rec["code"]
            feat.Description = rec.get("notes") or None
            feat.ObjectPlacement = model.createIfcLocalPlacement(wall.ObjectPlacement, model.createIfcAxis2Placement3D(
                point((0.0, 0.0, 0.0)), None, None))
            xyz = (f["verts"] - origin - loc) @ rot  # world -> the host's axes
            vp = [point(p) for p in xyz]
            shell = model.createIfcOpenShell(faces([[vp[a], vp[b], vp[c_]] for a, b, c_ in f["tris"]]))
            rep = model.createIfcShapeRepresentation(body, "Body", "SurfaceModel",
                                                     [model.createIfcShellBasedSurfaceModel([shell])])
            feat.Representation = model.createIfcProductDefinitionShape(None, None, [rep])
            names = f["names"]
            text = " - ".join(x for x in (names["it"] if lang() == "it" else names["en"], rec.get("notes")) if x)
            psets(feat, {
                "Pset_Condition": _clean({"AssessmentDate": c["date"], "AssessmentType": "EN 16096",
                                          "AssessmentCondition": None if f["condition"] is None
                                          else f"CC {f['condition']}", "AssessmentDescription": text}),
                "Pset_PyArchInit_Conservation": _feature_props(rec, c, names)})
            if rec.get("extent") is not None:
                line = (rec.get("geometry") or "").upper().startswith(("LINESTRING", "MULTILINESTRING"))
                qto = ifcopenshell.api.pset.add_qto(model, product=feat, name="Qto_SurfaceFeatureBaseQuantities")
                ifcopenshell.api.pset.edit_qto(model, qto=qto, properties={
                    ("Length" if line else "Area"): float(rec["extent"])})
            related.append(feat)
        if related:
            model.create_entity("IfcRelAdheresToElement", GlobalId=stable_global_id(host["gid"], "adheres"),
                                RelatingElement=wall, RelatedSurfaceFeatures=related)
        count += len(related)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))
    return count


# ---------------------------------------------------------------- reading back
def read_back(path) -> dict:
    """What a file holds (for the tests and the guide): ``{"schema", "origin", "hosts": [{"id", "name", "class",
    "psets", "vertices" (world), "triangles"}], "features": [{"id", "name", "object_type", "predefined_type",
    "description", "host", "psets", "vertices", "triangles"}], "relations": [{"host", "features"}]}``. The walls of
    any file of the suite are hosts (the masonry plugin's too)."""
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element

    with c_numbers():
        model = ifcopenshell.open(str(path))
        conv = model.by_type("IfcMapConversion")
        origin = (np.array([conv[0].Eastings, conv[0].Northings, conv[0].OrthogonalHeight or 0.0], np.float64)
                  if conv else np.zeros(3))
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)

        def shape(el):
            if el.Representation is None:
                return np.zeros((0, 3)), 0
            s = ifcopenshell.geom.create_shape(settings, el)
            verts = np.asarray(s.geometry.verts, np.float64).reshape(-1, 3) + origin
            return verts, len(s.geometry.faces) // 3

        def props(el):
            return {name: {k: v for k, v in p.items() if k != "id"}
                    for name, p in ifcopenshell.util.element.get_psets(el).items()}

        def every(cls):  # an IFC4 file (the masonry plugin's) has no surface features
            try:
                return model.by_type(cls)
            except RuntimeError:
                return []

        relations, host_of = [], {}
        for rel in every("IfcRelAdheresToElement"):
            ids = sorted(f.GlobalId for f in rel.RelatedSurfaceFeatures)
            relations.append({"host": rel.RelatingElement.GlobalId, "features": ids})
            host_of.update({i: rel.RelatingElement.GlobalId for i in ids})
        hosts = []
        walls = {w.id(): w for w in model.by_type("IfcWall")}
        walls.update({r.RelatingElement.id(): r.RelatingElement for r in every("IfcRelAdheresToElement")})
        for w in walls.values():
            v, n = shape(w)
            hosts.append({"id": w.GlobalId, "name": w.Name, "class": w.is_a(), "psets": props(w), "vertices": v,
                          "triangles": n})
        feats = []
        for f in every("IfcSurfaceFeature"):
            v, n = shape(f)
            feats.append({"id": f.GlobalId, "name": f.Name, "object_type": f.ObjectType,
                          "predefined_type": f.PredefinedType, "description": f.Description,
                          "host": host_of.get(f.GlobalId), "psets": props(f), "vertices": v, "triangles": n})
        return {"schema": getattr(model, "schema_identifier", model.schema), "origin": origin, "hosts": hosts,
                "features": feats, "relations": relations}
