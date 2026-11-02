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
from .texts import t
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
    from shapely import wkt
    from shapely.geometry import box

    st = walls.results(chunk, wid, entry["id"]).get("stones") or {}
    if st.get("region"):
        return wkt.loads(st["region"])
    return box(*entry["bounds"])


def wall_solid(chunk, wid: str, thickness: float | None = None) -> dict:
    """The slab of the wall: ``{"entry", "outline" (u, v), "origin", "u", "v", "into", "thickness", "source"}`` -
    the outline lies on the first face's fitted plane (moved along the frame's normal to it at the outline's centre),
    ``into`` points into the wall."""
    from .shape import _frame

    w = walls.need(chunk, wid)
    planes = [walls.with_bounds(chunk, e) for e in (walls.elevation(chunk, f) for f in w["faces"])
              if e is not None and (e.get("frame") or {}).get("kind") == "plane"]
    if not planes:
        raise NotReady(t("Il muro non ha facce piane: l'IFC del muro si fa da una faccia su un piano.",
                         "The wall has no plane face: the wall's IFC is made from a face on a plane."))
    a = planes[0]
    frame = _frame(a)
    outline = _outline(chunk, w["id"], a)
    c = outline.centroid
    pa, na = _plane_of(chunk, w["id"], a)
    at = frame.point(np.array([c.x]), np.array([c.y]), np.array([0.0]))[0]
    # the frame point moved along the frame's normal onto the fitted plane
    denom = float(frame.normal @ na)
    shift = float((pa - at) @ na / denom) if abs(denom) > 1e-6 else 0.0
    other = None
    for b in planes[1:]:
        if float(_frame(b).normal @ frame.normal) < -0.9:  # it looks the other way: the other side of the wall
            other = b
            break
    if other is not None:
        pb, nb = _plane_of(chunk, w["id"], other)
        start = at + shift * frame.normal
        into = -frame.normal
        den = float(into @ nb)
        thick = float((pb - start) @ nb / den) if abs(den) > 1e-6 else None
        source = t(f"misurato fra le facce «{a['name']}» e «{other['name']}»",
                   f"measured between the faces «{a['name']}» and «{other['name']}»")
        if thick is None or thick <= 0:
            raise NotReady(t("Le due facce non racchiudono un muro: controlla da che parte guardano.",
                             "The two faces do not enclose a wall: check which way they look."))
    else:
        if not thickness or thickness <= 0:
            raise NotReady(t("Il muro ha una sola faccia rilevata: dai lo spessore del muro.",
                             "The wall has one face surveyed only: give the wall's thickness."))
        thick = float(thickness)
        source = t(f"dato ({a['name']} rilevata da una parte sola)", f"given ({a['name']} surveyed on one side only)")
    origin = frame.point(np.array([0.0]), np.array([0.0]), np.array([shift]))[0]
    return {"entry": a, "other": other, "outline": outline, "origin": origin, "u": frame.u_axis, "v": frame.v_axis,
            "into": -frame.normal, "thickness": thick, "source": source}


def _pset(chunk, w: dict, solid: dict) -> dict:
    from .usm import face_units

    eid = solid["entry"]["id"]
    res = walls.results(chunk, w["id"], eid)
    sh, st = res.get("shape") or {}, res.get("stones") or {}
    props = {"Muro": w["name"], "Tipo": w["kind"], "Faccia": solid["entry"].get("name"),
             "Spessore": round(solid["thickness"], 4), "OrigineSpessore": solid["source"],
             "Facce": len(w["faces"]), "USM": len(face_units(chunk, eid))}
    if sh.get("surface") == "plane":
        props.update(StrapiomboGradi=round(sh["lean_deg"], 3), StrapiomboCmPerM=round(sh["lean_cm_per_m"], 2),
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
    """The wall's IFC in ``path``. Returns ``{"path", "thickness", "source", "parts", "relations"}``."""
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
    solid = wall_solid(chunk, w["id"], thickness)
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

    def placement():
        loc = solid["origin"] - origin
        return model.createIfcLocalPlacement(None, model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint(tuple(map(float, loc))),
            model.createIfcDirection(tuple(map(float, solid["into"]))),
            model.createIfcDirection(tuple(map(float, solid["u"])))))

    def extrusion(poly, depth):
        # local x = u, local z = into the wall, local y = z x x = -v: the outline as (u, -v)
        ring = [(float(x), float(-y)) for x, y in np.asarray(poly.exterior.coords)[:-1]]
        pts = model.createIfcCartesianPointList2D(ring + [ring[0]])
        profile = model.createIfcArbitraryClosedProfileDef("AREA", None, model.createIfcIndexedPolyCurve(pts))
        solid_e = model.createIfcExtrudedAreaSolid(profile, None, model.createIfcDirection((0.0, 0.0, 1.0)),
                                                   float(depth))
        rep = model.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid_e])
        return model.createIfcProductDefinitionShape(None, None, [rep])

    wall_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=w["name"],
                                                 predefined_type="SOLIDWALL")
    ifcopenshell.api.spatial.assign_container(model, products=[wall_e], relating_structure=site_e)
    wall_e.ObjectPlacement = placement()
    wall_e.Representation = extrusion(solid["outline"], solid["thickness"])
    pset = ifcopenshell.api.pset.add_pset(model, product=wall_e, name="Pset_PyArchInit_Masonry")
    ifcopenshell.api.pset.edit_pset(model, pset=pset, properties=_pset(chunk, w, solid))
    parts, by_label = [], {}
    for g, rec in face_units(chunk, solid["entry"]["id"]):
        poly = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
        part = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingElementPart", name=unit_label(rec))
        part.ObjectPlacement = placement()
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
    relations = 0
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
    return {"path": path, "thickness": solid["thickness"], "source": solid["source"], "parts": len(parts),
            "relations": relations, "area": float(solid["outline"].area)}
