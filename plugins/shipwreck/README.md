# Shipwreck

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/shipwreck/)**: from a 3D model of an excavated hull to its drawings, step by step.

Plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 or later): the
technical drawing of an excavated hull from its 3D model. Where the ship's axis runs, the shape of the hull at each
frame, a plan, and a numbered list of the timbers with their measures. It measures what is there: it reconstructs
nothing, and it does not fair a line away from the model.

| Step | What it does |
|---|---|
| **Ship frame** | x along the keel from a start point (0 at the stem or at a chosen station), y to port, z up: from a line traced on the raster or from two markers. Up is the world's vertical, the keel line is levelled and its slope kept as a number. You say which end is the bow: the model does not show it, and it decides which side is port. |
| **Stations** | regular, from a table `name, x` with the field's names, or from the frames of the catalogue. |
| **Sections** | cross-sections at the stations, longitudinals at given y, horizontals at given z, cut from the mesh in double precision; `<chunk data>/shipwreck/sections.json`. |
| **Water** | a level and an index (1.34): under the level a point is drawn at `level - (level - z) * n`. Off by default. A first approximation for a nadir view: oblique rays and a moving surface are not modelled. The raw and the corrected line are both kept and each drawing says which it shows. |
| **Drawings** | body plan (seen from aft, port on the left), half-breadth plan, profile, plan with the stations over the orthophoto: a scale, automatic or chosen, a scale bar, the north, a title block; SVG, PDF, PNG and DXF with a layer per kind. Plans are never mirror images, whichever way x runs. |
| **Timbers** | keel, keelson, frames, planks, ceiling, stringers, fastenings: drawn on the raster or from a CSV; length along the timber over the model, width across, heights above a datum, room and space; a table, a layer of the raster view, CSV, XLSX and GeoPackage. |

Limits. A chunk in degrees (WGS 84) is refused: a frame needs metres (*GNSS > System and heights* brings the chunk to a
projected system; the import from Metashape does it by itself). No hull reconstruction, mirrored sides, lofted surfaces,
hydrostatics, plank expansion, automatic detection of timbers, or merging of several days. Picks in the 3D view are not
offered directly: place two markers there and use *From two markers*. Heights over the model are good to about one edge of
the mesh.

The transform, the model and the products of the chunk are never changed. Everything is written in
`<chunk data>/shipwreck/`, apart from what you export.

## Command line

```bash
ppg plugin shipwreck frame PROJECT.ppg --start X,Y[,Z] --end X,Y[,Z] [--bow start|end] [--z0 H] [--x0 X]
ppg plugin shipwreck frame PROJECT.ppg --markers --start A --end B
ppg plugin shipwreck frame PROJECT.ppg --flip
ppg plugin shipwreck stations PROJECT.ppg --spacing 1 --count 11 --first 1      # or --table FILE, --from-catalogue
ppg plugin shipwreck sections PROJECT.ppg --longitudinals "0;1;-1" --horizontals "-0.2;-0.5" --half-width 4.5
ppg plugin shipwreck water PROJECT.ppg --level -0.1 --n 1.34 --on
ppg plugin shipwreck drawings PROJECT.ppg [--scale 20] [--out FOLDER]
ppg plugin shipwreck timbers PROJECT.ppg [--table FILE]
ppg plugin shipwreck export PROJECT.ppg --out timbers.xlsx                      # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROJECT.ppg
```

A table of timbers has a header: `kind`, `side`, `name`, `station`, `notes` and either `x, y` (a point) or `x1, y1, x2, y2`
(a line) in ship coordinates, or `wkt` in the coordinates of the chunk.

## License

GPL-3.0-or-later.
