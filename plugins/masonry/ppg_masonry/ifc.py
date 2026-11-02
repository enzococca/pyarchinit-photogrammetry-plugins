"""The wall in IFC4, with the program's conventions (metres, ``IfcMapConversion`` on a round origin when the chunk has
a CRS, the elements in an ``IfcSite``): an ``IfcWall`` whose solid is parametric - the outline of a face (the wall's
area drawn on it, else its bounds) extruded into the wall (``IfcExtrudedAreaSolid``):

* between two surveyed faces (two plane faces of the wall looking away from each other): from the first face's fitted
  plane to the second's, the thickness measured between them at the outline's centre;
* behind one face, by a thickness given.

``Pset_PyArchInit_Masonry`` carries the measures (lean, bulge p90, courses, mortar share, the type the measures
suggest, the counts, the thickness and how it was found); the USM drawn on the face are ``IfcBuildingElementPart`` of
the wall (``IfcRelAggregates``), the outline of each extruded across the same thickness, with ``Pset_PyArchInit_US``,
and the relationships accepted by the user as ``IfcRelConnectsElements``. Only plane faces: the developed face of a
round tower is not a slab."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import walls
from .texts import num, t
from .walls import NotReady


def _plane_of(chunk, wid: str, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(point, normal towards the viewer)`` of the face: its fitted plane when measured, else its frame's plane."""
    from .shape import _frame

    res = walls.results(chunk, wid, entry["id"]).get("shape") or {}
    if res.get("plane"):
        return np.asarray(res["plane"]["centre"], float), np.asarray(res["plane"]["normal"], float)
    frame = _frame(entry)
    return frame.origin, frame.normal


def _outline(chunk, wid: str, entry: dict):
    from shapely.geometry import box

    from .stones import drawn_region

    region = drawn_region(chunk, wid, entry["id"])
    return region if region is not None else box(*entry["bounds"])


PAIR_ANGLE_DEG = 15.0  # two faces are the two sides of one wall when they look within this of opposite ways
PAIR_MAX_DISTANCE = 3.0  # and are closer than this (m): a tower's opposite sides, a building's facades are not


def _solid_of(chunk, wid: str, a: dict) -> dict:
    """The slab of the face ``a`` without its thickness: the outline on its fitted plane."""
    from .shape import _frame

    frame = _frame(a)
    outline = _outline(chunk, wid, a)
    c = outline.centroid
    pa, na = _plane_of(chunk, wid, a)
    at = frame.point(np.array([c.x]), np.array([c.y]), np.array([0.0]))[0]
    denom = float(frame.normal @ na)  # the frame point moved along the frame's normal onto the fitted plane
    shift = float((pa - at) @ na / denom) if abs(denom) > 1e-6 else 0.0
    origin = frame.point(np.array([0.0]), np.array([0.0]), np.array([shift]))[0]
    return {"entry": a, "other": None, "outline": outline, "origin": origin, "u": frame.u_axis, "v": frame.v_axis,
            "into": -frame.normal, "start": at + shift * frame.normal, "frame_normal": frame.normal}


def _face_point_at(chunk, wid: str, entry: dict, h: float) -> np.ndarray:
    """The point of the face ``entry`` at its middle column and the world height ``h``, on the wall's surface: on its
    fitted plane when its shape was measured, else at the depth its elevation's relief has there, else on its frame."""
    from .shape import _frame, _relief_at, _v_at_height, read_relief

    frame = _frame(entry)
    u0, _v0, u1, _v1 = entry["bounds"]
    u = 0.5 * (u0 + u1)
    v = _v_at_height(frame, u, h)
    p0 = frame.point(np.array([u]), np.array([v]), np.array([0.0]))[0]
    plane = (walls.results(chunk, wid, entry["id"]).get("shape") or {}).get("plane")
    d = None
    if plane:
        n, c = np.asarray(plane["normal"], float), np.asarray(plane["centre"], float)
        den = float(frame.normal @ n)
        d = float((c - p0) @ n / den) if abs(den) > 1e-6 else None
    if d is None:
        try:
            relief, profile = read_relief(chunk, entry)
            d = _relief_at(relief, profile["transform"], u, v)
        except NotReady:
            d = None
    return frame.point(np.array([u]), np.array([v]), np.array([d or 0.0]))[0]


def _tolerance(chunk, wid: str, entries, length: float) -> float:
    """How far from each face a slice line still belongs to that face: the larger of a tenth of the distance between
    them, 5 cm and the faces' bulge (when measured). The segment's ends are on the faces' surfaces at the slice's
    height, so lean and bulge need no more; the faces' depth bands are not used (with a band of half a metre the outer
    faces of a corridor or of a cavity wall would pass for one wall)."""
    tol = max(0.1 * length, 0.05)
    for e in entries:
        sh = walls.results(chunk, wid, e["id"]).get("shape") or {}
        if sh.get("bulge_min_cm") is not None:
            tol = max(tol, abs(sh["bulge_min_cm"]) / 100.0, abs(sh["bulge_max_cm"]) / 100.0)
    return tol


def _solid_between(chunk, wid: str, a: dict, b: dict) -> bool | None:
    """Whether the space between two faces is the wall itself: the model's horizontal slice at the mid-height of the
    two faces crosses the segment between them - from the surface of one face to that of the other, at that height -
    only near its ends (within :func:`_tolerance` of each face). A slice line crossing the rest is another wall, a
    room, the inside of a building. None when the model cannot be sliced."""
    from shapely.geometry import LineString, MultiLineString

    from .faces import slice_rings
    from .shape import _frame

    heights = []
    for e in (a, b):
        frame = _frame(e)
        u0, v0, u1, v1 = e["bounds"]
        um = 0.5 * (u0 + u1)
        heights += list(frame.point(np.array([um, um]), np.array([v0, v1]), np.zeros(2))[:, 2])
    h = 0.5 * (max(min(heights[:2]), min(heights[2:])) + min(max(heights[:2]), max(heights[2:])))
    try:
        rings = slice_rings(chunk, h)
    except Exception:  # noqa: BLE001 - no model to slice
        return None
    if not rings:
        return None
    pa, pb = _face_point_at(chunk, wid, a, h), _face_point_at(chunk, wid, b, h)
    seg = LineString([pa[:2], pb[:2]])
    tol = _tolerance(chunk, wid, (a, b), seg.length)
    if 2 * tol >= seg.length:
        return True  # nothing between the two faces but their own surfaces
    inner = LineString([seg.interpolate(tol), seg.interpolate(seg.length - tol)])
    return not inner.intersects(MultiLineString([r for r in rings if len(r) >= 2]))


def _pair_thickness(chunk, w: dict, solid: dict, b: dict) -> tuple[float | None, str]:
    """The thickness between the face of ``solid`` and the face ``b`` when they are the two sides of one wall: never in
    a tower nor between faces of one slice (the sides of a polygonal tower); they look within ``PAIR_ANGLE_DEG`` of
    opposite ways, are closer than ``PAIR_MAX_DISTANCE`` and the space between them is solid
    (:func:`_solid_between`). ``(None, why)`` otherwise."""
    from .shape import _frame

    slices = w.get("slices") or {}
    if w.get("kind") == "tower" or (slices.get(solid["entry"]["id"]) is not None
                                     and slices.get(solid["entry"]["id"]) == slices.get(b["id"])):
        return None, "tower"
    cos = float(_frame(b).normal @ solid["frame_normal"])
    if cos > -np.cos(np.radians(PAIR_ANGLE_DEG)):
        return None, "angle"
    pb, nb = _plane_of(chunk, w["id"], b)
    den = float(solid["into"] @ nb)
    if abs(den) < 1e-6:
        return None, "angle"
    thick = float((pb - solid["start"]) @ nb / den)
    if thick <= 0:
        return None, "side"
    if thick > PAIR_MAX_DISTANCE:
        return None, "far"
    solid_between = _solid_between(chunk, w["id"], solid["entry"], b)
    if solid_between is False:
        return None, "hollow"
    return thick, ""


def wall_solids(chunk, wid: str, thickness: float | None = None) -> list[dict]:
    """The slabs of the wall: ``[{"entry", "other", "outline" (u, v), "origin", "u", "v", "into", "thickness",
    "source"}]``. Two plane faces of a wall or a building that are the two sides of one wall (they look within 15
    degrees of opposite ways, lie less than 3 m apart and the model between them is solid) make one slab, its thickness
    measured between their fitted planes; every other plane face (a wall surveyed on one side, each side of a tower)
    makes a slab of the ``thickness`` given - without it the faces that need it are named."""
    w = walls.need(chunk, wid)
    planes = [walls.with_bounds(chunk, e) for e in (walls.elevation(chunk, f) for f in w["faces"])
              if e is not None and (e.get("frame") or {}).get("kind") == "plane"]
    if not planes:
        raise NotReady(t("Il muro non ha facce piane: l'IFC del muro si fa da una faccia su un piano.",
                         "The wall has no plane face: the wall's IFC is made from a face on a plane."))
    out, used, need, refused = [], set(), [], []
    for i, a in enumerate(planes):
        if i in used:
            continue
        solid = _solid_of(chunk, w["id"], a)
        for j in range(i + 1, len(planes)):
            if j in used:
                continue
            thick, why = _pair_thickness(chunk, w, solid, planes[j])
            if thick is not None:
                used.add(j)
                solid.update(other=planes[j], thickness=thick,
                             source=t(f"misurato fra le facce «{a['name']}» e «{planes[j]['name']}»",
                                      f"measured between the faces «{a['name']}» and «{planes[j]['name']}»"))
                break
            if why in ("far", "hollow"):
                refused.append(planes[j]["name"])
        if solid["other"] is None:
            if not thickness or thickness <= 0:
                need.append(a["name"])
                continue
            solid.update(thickness=float(thickness),
                         source=t(f"dato ({a['name']} rilevata da una parte sola)",
                                  f"given ({a['name']} surveyed on one side only)"))
        out.append(solid)
    if need:
        extra = t(" Facce opposte troppo lontane (oltre "
                  f"{num(PAIR_MAX_DISTANCE, 0)} m) o con altro fra loro per essere le due parti di un muro: "
                  + ", ".join(sorted(set(refused))) + ".", " Opposite faces too far apart (over "
                  f"{num(PAIR_MAX_DISTANCE, 0)} m) or with something else between them to be the two sides of one "
                  "wall: " + ", ".join(sorted(set(refused))) + ".") if refused else ""
        raise NotReady(t(f"Dai lo spessore del muro: le facce {', '.join(need)} non hanno la faccia dell'altra parte "
                         "del muro." + extra, f"Give the wall's thickness: the faces {', '.join(need)} have no face on "
                         "the other side of the wall." + extra))
    return out


def wall_solid(chunk, wid: str, thickness: float | None = None) -> dict:
    """The first slab of :func:`wall_solids`."""
    return wall_solids(chunk, wid, thickness)[0]


def _pset(chunk, w: dict, solid: dict) -> dict:
    from .usm import face_units

    eid = solid["entry"]["id"]
    res = walls.results(chunk, w["id"], eid)
    sh, st = res.get("shape") or {}, res.get("stones") or {}
    props = {"Muro": w["name"], "Tipo": w["kind"], "Faccia": solid["entry"].get("name"),
             "Spessore": round(solid["thickness"], 4), "OrigineSpessore": solid["source"],
             "Facce": len(w["faces"]), "USM": len(face_units(chunk, eid))}
    if sh.get("surface") == "plane":
        props.update(StrapiomboAffidabile=bool(sh.get("lean_reliable", True)),
                     StrapiomboNota=None if sh.get("lean_reliable", True) else "non affidabile: la superficie non è "
                     "piana",
                     StrapiomboGradi=round(sh["lean_deg"], 3), StrapiomboCmPerM=round(sh["lean_cm_per_m"], 2),
                     StrapiomboVerso="verso chi guarda" if sh.get("lean_toward") == "out" else "all'indietro",
                     StrapiomboAzimut=None if sh.get("lean_azimuth") is None else round(sh["lean_azimuth"], 1),
                     SpanciamentoP90Cm=round(sh.get("bulge_p90_cm", 0.0), 2),
                     SpanciamentoMinCm=round(sh["bulge_min_cm"], 2), SpanciamentoMaxCm=round(sh["bulge_max_cm"], 2))
    if st.get("typology"):
        props.update(Pietre=st["stones"], Filari=st["courses"], QuotaMalta=round(st["mortar"], 4),
                     TipoSuggerito=st["typology"]["line"])
    return {k: v for k, v in props.items() if v is not None}


def export_wall(chunk, wid: str, path, thickness: float | None = None, project: str | None = None,
                site: str | None = None) -> dict:
    """The wall's IFC in ``path``: an ``IfcWall`` per slab of :func:`wall_solids`. Returns ``{"path", "thickness",
    "source" (of the first slab), "parts", "relations", "walls", "sources"}``."""
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.georeference
    import ifcopenshell.api.pset
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit
    import ifcopenshell.guid

    from pyarchinit_photogrammetry.core import crs as crs_mod

    from .usm import face_units, unit_label

    w = walls.need(chunk, wid)
    solids = wall_solids(chunk, w["id"], thickness)
    solid = solids[0]
    doc = getattr(chunk, "document", None)
    model = ifcopenshell.file(schema="IFC4")
    proj = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=project or "Rilievo")
    ifcopenshell.api.unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"},
                                      area={"is_metric": True, "raw": "METERS"},
                                      volume={"is_metric": True, "raw": "METERS"})
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=ctx)
    origin = np.floor(solid["origin"])
    if not crs_mod.is_local(chunk.crs):
        ifcopenshell.api.georeference.add_georeferencing(model)
        epsg, vertical = crs_mod.epsg_code(chunk.crs), crs_mod.vertical_part(chunk.crs)
        ifcopenshell.api.georeference.edit_georeferencing(
            model, projected_crs={"Name": f"EPSG:{epsg}" if epsg else str(crs_mod.horizontal(chunk.crs)),
                                  **({"VerticalDatum": vertical} if vertical else {})},
            coordinate_operation={"Eastings": float(origin[0]), "Northings": float(origin[1]),
                                  "OrthogonalHeight": float(origin[2]), "XAxisAbscissa": 1.0, "XAxisOrdinate": 0.0,
                                  "Scale": 1.0})
    site_name = site or ((doc.meta.get("sito") if doc is not None else None) or "Sito")
    site_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name=site_name)
    ifcopenshell.api.aggregate.assign_object(model, products=[site_e], relating_object=proj)

    def placement(solid):
        loc = solid["origin"] - origin
        return model.createIfcLocalPlacement(None, model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint(tuple(map(float, loc))),
            model.createIfcDirection(tuple(map(float, solid["into"]))),
            model.createIfcDirection(tuple(map(float, solid["u"])))))

    def extrusion(geom, depth):
        # local x = u, local z = into the wall, local y = z x x = -v: the outline as (u, -v); a wall's area in several
        # parts is one solid per part
        items = []
        for poly in getattr(geom, "geoms", [geom]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            ring = [(float(x), float(-y)) for x, y in np.asarray(poly.exterior.coords)[:-1]]
            pts = model.createIfcCartesianPointList2D(ring + [ring[0]])
            profile = model.createIfcArbitraryClosedProfileDef("AREA", None, model.createIfcIndexedPolyCurve(pts))
            items.append(model.createIfcExtrudedAreaSolid(profile, None, model.createIfcDirection((0.0, 0.0, 1.0)),
                                                          float(depth)))
        rep = model.createIfcShapeRepresentation(body, "Body", "SweptSolid", items)
        return model.createIfcProductDefinitionShape(None, None, [rep])

    parts_n = relations = 0
    for solid in solids:
        name = w["name"] if len(solids) == 1 else f"{w['name']} - {solid['entry'].get('name', '')}"
        wall_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=name,
                                                     predefined_type="SOLIDWALL")
        ifcopenshell.api.spatial.assign_container(model, products=[wall_e], relating_structure=site_e)
        wall_e.ObjectPlacement = placement(solid)
        wall_e.Representation = extrusion(solid["outline"], solid["thickness"])
        pset = ifcopenshell.api.pset.add_pset(model, product=wall_e, name="Pset_PyArchInit_Masonry")
        ifcopenshell.api.pset.edit_pset(model, pset=pset, properties=_pset(chunk, w, solid))
        parts, by_label = [], {}
        for g, rec in face_units(chunk, solid["entry"]["id"]):
            poly = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
            part = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingElementPart",
                                                       name=unit_label(rec))
            part.ObjectPlacement = placement(solid)
            part.Representation = extrusion(poly, solid["thickness"])
            ps = ifcopenshell.api.pset.add_pset(model, product=part, name="Pset_PyArchInit_US")
            attrs = rec.get("attributes") or {}
            ifcopenshell.api.pset.edit_pset(model, pset=ps, properties={k: str(v) for k, v in {
                "us": rec.get("us"), "unita_tipo": rec.get("tipo"), "area": rec.get("area"), "sito": rec.get("sito"),
                "d_stratigrafica": attrs.get("d_stratigrafica")}.items() if v not in (None, "")})
            parts.append(part)
            by_label[unit_label(rec)] = part
        if parts:
            ifcopenshell.api.aggregate.assign_object(model, products=parts, relating_object=wall_e)
        parts_n += len(parts)
        usm_res = walls.results(chunk, w["id"], solid["entry"]["id"]).get("usm") or {}
        for r in usm_res.get("accepted_full") or []:
            a, b = by_label.get(r["unit"]), by_label.get(r["other"])
            if a is not None and b is not None:
                model.create_entity("IfcRelConnectsElements", GlobalId=ifcopenshell.guid.new(), Name=r["relation"],
                                    Description="rapporto stratigrafico proposto dalla geometria e accettato",
                                    RelatingElement=a, RelatedElement=b)
                relations += 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))
    return {"path": path, "thickness": solid["thickness"], "source": solid["source"], "parts": parts_n,
            "relations": relations, "area": float(solid["outline"].area), "walls": len(solids),
            "sources": [s_["source"] for s_ in solids]}
