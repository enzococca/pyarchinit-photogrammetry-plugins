# Shipwreck

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/shipwreck/)**: from a 3D model of an excavated hull to its drawings, step by step.

Plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 or later): the
technical drawing of an excavated hull from its 3D model. Where the ship's axis runs, the shape of the hull at each
frame, a plan, and a numbered list of the timbers with their measures. It measures what is there: it reconstructs
nothing, and it does not fair a line away from the model.

| Step | What it does |
|---|---|
| **Ship frame** | x along the keel from a start point (0 at the stem or at a chosen station), y to port, z up: from a line traced on the raster or from two markers. Up is the world's vertical, the keel line is levelled and its slope kept as a number. You say which end is the bow (*The bow is the other end*, which exchanges port and starboard): the model does not show it. *x from the other end* only turns x round: the bow, y and the sides stay. |
| **Stations** | regular, from a table `name, x` with the field's names, or from the frames of the catalogue. Separated by commas a table takes decimal points; by semicolons or tabs also decimal commas. A row with too many cells is refused. |
| **Sections** | cross-sections at the stations, longitudinals at given y, horizontals at given z, cut from the mesh in double precision, as far across the keel as the model goes unless you say otherwise; `<chunk data>/shipwreck/sections.json`. Stale sections and drawings are said: a change of frame, model, georeference or system. |
| **Water** | a level and an index (1.34): under the level a point is drawn at `level - (level - z) * n`. Off by default. A first approximation for a nadir view: oblique rays and a moving surface are not modelled. The raw and the corrected line are both kept and each drawing says which it shows. |
| **Drawings** | body plan (seen from aft, port on the left), half-breadth plan, profile, plan with the stations over the orthophoto: a scale, automatic or chosen, a scale bar, the north, a title block; SVG, PDF, PNG and DXF with a layer per kind. Plans are never mirror images, whichever way x runs. |
| **Timbers** | keel, keelson, frames, planks, ceiling, stringers, fastenings: drawn on the raster or from a CSV; length along the timber over the model, width across, heights above a datum, room and space; a table, a layer of the raster view, CSV, XLSX and GeoPackage. |
| **Strakes (clinker)** | *Find the strakes*: the lands (the steps where a strake laps over the one below) on the raw section of every station and side, from the rabbet out to the sheer, joined into land lines along the hull; stations with another count of lands (a stealer, a drop strake, damage) or whose rabbet or garboard jumps are listed as uncertain; the lines in the drawings and in a layer "Fasce" of the raster view, DXF and GeoPackage with Z. *Strake model*: a strake between two land lines, its width at every station with an uncertainty, length, thickness (the median step) and lap, a mesh per strake (OBJ, PLY in the chunk's coordinates), CSV and XLSX; *Into the catalogue* adds a plank record per strake. *For the Clinker Boat Tool* writes the files for Fröhlich & Nielsen's Houdini asset (Y up); *Planks from Houdini* reads its planks back with their distance from the scan. The lands are measured; the lines and the strake model are a hypothesis, and every file says so. |

Limits. A chunk in degrees (WGS 84) is refused: a frame needs metres (*GNSS > System and heights* brings the chunk to a
projected system; the import from Metashape does it by itself). No hull reconstruction, mirrored sides, lofted surfaces,
hydrostatics, plank expansion, automatic detection of timbers, or merging of several days. Picks in the 3D view are not
offered directly: place two markers there and use *From two markers*. Heights over the model are good to about one edge of
the mesh.

Limits of the strakes. On sections of a mesh noisier than about 0.5 mm the widths are good to about a centimetre; at 1 mm
of noise about one side in fifteen may get a wrong count of lands (flagged as an uncertain station). A jump of 2-2.5 cm of
the rabbet at an end station, three or more adjacent shifted stations, or a gradual step are not flagged; a permanent step
of the keel (a broken or displaced hull, a scarf) flags every station on one side of it; above about 3 mm of scatter on the
rabbet a station may be flagged wrongly. A V of planking steeper than 80° within the keel's half width without a sole is
read as keel until it flattens; a keel side in line with the garboard is a side not read (`keel_unclear`). The strake model
needs at least three stations with a width on each side with lands. A hull seen from inside (planking under the frames,
the keel hidden by the keelson) gives uncertain stations: the next version will trace the lands on the hull's relief.

The Clinker Boat Tool is a Houdini Digital Asset by Fred Fröhlich and Sigurd Solhaug Nielsen (MIT,
[10.5281/zenodo.17488549](https://doi.org/10.5281/zenodo.17488549)), described in *From Photogrammetry to Construction
Knowledge: 3D Documentation and Parametric Reconstruction of Historic Clinker Boats* (CC BY 4.0,
[10.5281/zenodo.23422398](https://doi.org/10.5281/zenodo.23422398)). The plugin writes files that the asset's own steps
take in and reads back the planks it exports; nothing of the asset is copied or used.

The transform, the model and the products of the chunk are never changed. Everything is written in
`<chunk data>/shipwreck/`, apart from what you export.

## Command line

```bash
ppg plugin shipwreck frame PROJECT.ppg --start X,Y[,Z] --end X,Y[,Z] [--bow start|end] [--z0 H] [--x0 X]
ppg plugin shipwreck frame PROJECT.ppg --markers --start A --end B
ppg plugin shipwreck frame PROJECT.ppg --flip        # x from the other end; the sides stay
ppg plugin shipwreck frame PROJECT.ppg --other-bow   # the bow is the other end; port and starboard are exchanged
ppg plugin shipwreck stations PROJECT.ppg --spacing 1 --count 11 --first 1      # or --table FILE, --from-catalogue
ppg plugin shipwreck sections PROJECT.ppg --longitudinals "0;1;-1" --horizontals "-0.2;-0.5" --half-width 4.5
ppg plugin shipwreck water PROJECT.ppg --level -0.1 --n 1.34 --on
ppg plugin shipwreck drawings PROJECT.ppg [--scale 20] [--out FOLDER]
ppg plugin shipwreck timbers PROJECT.ppg [--table FILE]
ppg plugin shipwreck export PROJECT.ppg --out timbers.xlsx                      # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROJECT.ppg
ppg plugin shipwreck clinker detect PROJECT.ppg [--min-step MM] [--min-strake M]   # the lands and the land lines
ppg plugin shipwreck clinker strakes PROJECT.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROJECT.ppg --out FOLDER                        # files for the Clinker Boat Tool
ppg plugin shipwreck clinker import PROJECT.ppg --obj PLANKS.obj [--side-from y|name]
ppg plugin shipwreck clinker status PROJECT.ppg                                     # also says when the lands are stale
```

A table of timbers has a header: `kind`, `side`, `name`, `station`, `notes` and either `x, y` (a point) or `x1, y1, x2, y2`
(a line) in ship coordinates, or `wkt` in the coordinates of the chunk.

## License

GPL-2.0-or-later.
