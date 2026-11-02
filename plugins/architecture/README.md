# Architectural elements

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 or later):
parametric architectural elements made from the photogrammetric model, for an HBIM model.

* **Column by revolution**: draw the region in plan (and two heights); the axis and the profile are measured on the
  model, the solid is a spline through K knots fitted to the measured profile, or the profile as measured. Base, shaft
  and capital can be separate elements on the same axis.
* **Moulding, cornice, string course**: trace the path on the plan, on an elevation or a section of the project, or
  between two markers; the section is cut from the model by the plane normal to the path and closed against the wall;
  the solid is the section swept along the path with mitred corners (an extrusion when the path is straight).
* **Arch and barrel vault**: on a section of the project (the point inside the opening clicked on it) or a vertical
  plane through two points; the intrados is fitted as a circle (semicircular or segmental), a pointed arch, an ellipse,
  or kept as measured, the type chosen by the fit; the thickness and the depth are measured or given.
* **Door and window** (0.2.0): click inside the opening on the plan; the plan of the void comes from a horizontal cut of
  the model (by default the middle of the opening), the two reveals fitted by lines (a splayed reveal stays splayed),
  closed by the wall faces (or the wall's thickness, given when only one face is on the model); the sill and the
  lintel from the vertical section across the wall; the top flat or the intrados of an arch element. Reported: widths
  at the outer and the inner face and near the sill and the lintel, height, sill and lintel (world heights and above
  the ground of the chunk's DEM), depth, the splay of each reveal, area.

Every element is written as **STEP** (AP214 in metres, from CadQuery and OpenCascade), **STL** (no unit in the
format: its numbers are metres), a parametric **IFC4** solid
(`IfcColumn` with `IfcRevolvedAreaSolid`, `IfcMember` with `IfcExtrudedAreaSolid` or `IfcFixedReferenceSweptAreaSolid`,
`IfcBuildingElementProxy` "Arch"/"Vault" with arcs as `IfcArcIndex`, `IfcOpeningElement` with an `IfcDoor` or an
`IfcWindow` (`OverallWidth`, `OverallHeight`), `Pset_PyArchInit_Element`, the survey mesh
beside on request, `IfcMapConversion` on the chunk's origin) and **DXF** of its profile or section. It always carries
its **fit**: the median, 90th percentile and maximum distance of the model's points from the solid. The solid is an
interpretation of the measured model; the fit says how far.

The coordinates of the files are those of the chunk's system minus a round origin kept per chunk: an exported STEP
gets a `<name>.origin.json` beside it with that origin. When the chunk is georeferenced or moved to another system after
the elements were made, the plugin refuses to export or edit them until *Rebuild* makes them again in the new system.

## The proof

On two synthetic scenes photographed, aligned, densified and meshed by the program (script `proof/make_proof.py`):
a column 2 m high with entasis came out with its axis within 0.001 degrees and 0.1 mm, the knots within 1.6 mm, the
volume +0.5 %; a pointed arch (span 1.6 m) was recognised as pointed, span -4 mm, rise +7 mm, ring thickness +4 mm,
where the program's model itself is 2 mm (median) from the true surface. In the same wall a door 0.9 m wide came out
896 mm wide, its sill and lintel within 1.1 mm; a window splayed 10 degrees came out with splays of 9.9 and 10.0
degrees, sill and lintel within 2.7 mm and its width at the outer face 6 mm narrow (the program's model itself is about
4 mm narrow there). The guide of the program has the details.

## The environment

CadQuery runs in an environment of its own, never in the program's process. *Installa CadQuery* in the pane (or
`ppg plugin architecture install`) makes it with `uv` in the plugin's folder: Python 3.12, CadQuery, OpenCascade
(OCP) and the VTK OpenCascade is linked against, without casadi, numba, scipy, matplotlib and trame. **About 900 MB**,
a few minutes; the first run after the installation takes about a minute. A Python that has CadQuery already is used
as it is (`pip install cadquery`), and `PPG_ARCHITECTURE_ENV` puts the environment elsewhere.

OpenCascade has builds for macOS (Apple Silicon and Intel), Linux (x86-64 and aarch64) and Windows x64, **not for
Windows on ARM**: there the pane says that the elements are not available.

## Command line

```
ppg plugin architecture install | status
ppg plugin architecture column   PROJECT --polygon "POLYGON((x y, ...))" --zmin 0.05 --zmax 1.95 --knots 12
ppg plugin architecture moulding PROJECT --path "LINESTRING(x y, x y)" --height 3.35 [--elevation ID] [--markers A,B]
                                 [--window u0,v0,u1,v1]   (default: found from the cut)
ppg plugin architecture arch     PROJECT --section ID --pick u,v [--type auto|pointed|...] [--thickness T] [--vault]
ppg plugin architecture opening  PROJECT --at x,y [--height H] [--top ARCH] [--thickness T] [--kind door|window|opening]
                                 [--sill S] [--lintel L]
ppg plugin architecture rebuild  PROJECT     (after the chunk's transform or system changed)
ppg plugin architecture list | show ID | remove ID PROJECT
ppg plugin architecture export   PROJECT --out elementi.ifc [--mesh] | --out folder/ | --out arco.dxf --ids Arco
```

## Licences

The plugin: GPL-2.0-or-later. In its environment: CadQuery and OCP (Apache-2.0), OpenCascade (LGPL-2.1 with an
exception), VTK (BSD-3-Clause), ezdxf (MIT), nlopt (MIT/LGPL), numpy (BSD).
