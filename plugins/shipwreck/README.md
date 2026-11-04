# Shipwreck

*[Italiano](README.it.md)*

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/shipwreck/)**: from a 3D model of an excavated hull to its drawings, step by step.

Plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 or later): the
technical drawing of an excavated hull from its 3D model. Where the ship's axis runs, the shape of the hull at each
frame, a plan, a numbered list of the timbers with their measures, the plan of the wreck with every timber outlined and
named, and every timber a 3D object. It measures what is there and does not fair a line away from the model; what it
adds (a keel of given dimensions, the strake model of the sections, the reconstruction of what is lost) is a stated
hypothesis, hatched and apart.

| Step | What it does |
|---|---|
| **Ship frame** | x along the keel from a start point (0 at the stem or at a chosen station), y to port, z up: from a line traced on the raster or from two markers. Up is the world's vertical, the keel line is levelled and its slope kept as a number. You say which end is the bow (*The bow is the other end*, which exchanges port and starboard): the model does not show it. *x from the other end* only turns x round: the bow, y and the sides stay. |
| **Stations** | regular, from a table `name, x` with the field's names, or from the frames of the catalogue. Separated by commas a table takes decimal points; by semicolons or tabs also decimal commas. A row with too many cells is refused. |
| **Sections** | cross-sections at the stations, longitudinals at given y, horizontals at given z, cut from the mesh in double precision, as far across the keel as the model goes unless you say otherwise; `<chunk data>/shipwreck/sections.json`. Stale sections and drawings are said: a change of frame, model, georeference or system. |
| **Water** | a level and an index (1.34): under the level a point is drawn at `level - (level - z) * n`. Off by default. A first approximation for a nadir view: oblique rays and a moving surface are not modelled. The raw and the corrected line are both kept and each drawing says which it shows. |
| **Drawings** | body plan (seen from aft, port on the left), half-breadth plan, profile, plan with the stations over the orthophoto: a scale, automatic or chosen, a scale bar, the north, a title block; SVG, PDF, PNG and DXF with a layer per kind. Plans are never mirror images, whichever way x runs. |
| **Timbers** | keel, keelson, frames, planks, ceiling, stringers, fastenings: drawn on the raster or from a CSV; length along the timber over the model, width across, heights above a datum, room and space; a table, a layer of the raster view, CSV, XLSX and GeoPackage. |
| **Strakes (clinker)** | *Find the strakes*: the lands (the steps where a strake laps over the one below) on the raw section of every station and side, from the rabbet out to the sheer, joined into land lines along the hull; stations with another count of lands (a stealer, a drop strake, damage) or whose rabbet or garboard jumps are listed as uncertain; the lines in the drawings and in a layer "Fasce" of the raster view, DXF and GeoPackage with Z. *Strake model*: a strake between two land lines, its width at every station with an uncertainty, length, thickness (the median step) and lap, a mesh per strake (OBJ, PLY in the chunk's coordinates), CSV and XLSX; *Into the catalogue* adds a plank record per strake. *For the Clinker Boat Tool* writes the files for Fröhlich & Nielsen's Houdini asset (Y up); *Planks from Houdini* reads its planks back with their distance from the scan. The lands are measured; the lines and the strake model are a hypothesis, and every file says so. |
| **Relief and tracing** | *Relief of the hull*: the hull seen from above in the ship frame (an elevation of the program at 5 mm by default, u = x, v = y to the left of x) with its hillshade and local relief model, to draw on. *Propose*: seams, frames, keelson, stringers and strakes read on it, proposals to *Accept*, *Reject*, *Correct* or *Add* by hand (snap included), named from a table, flagged *Loose*; each change a step of Undo. *Keel (dimensions)*: a keel from a given siding and moulding under the keelson, a hypothesis. *3D timbers*: every traced polygon a closed object (a strake with its thickness, a frame or a longitudinal timber a solid from the planking to its top), one PLY each. *Export 3D*: OBJ, GLB or IFC4 (one element per timber with `Pset_PyArchInit_Timber`) in ship or chunk coordinates, with the tables. *Plan of the wreck*: every timber outlined, coloured by kind and named over the hillshade or the orthophoto, the hypotheses hatched; SVG, PDF, PNG, DXF. |
| **Reconstruction (hypothesis)** | *Stem and stern*: how far the reconstruction may go past the surveyed length. *Reconstruct*: the preserved side mirrored where the other is lost (as the wreck lies; where both are measured, their disagreement listed), the strakes extended along their seams within the surveyed length and to the ends given, the missing frames put back at the measured spacing, the faired hull lofted through the stations' sections where the wreck shows nothing (open at an end not given), over the other chunks of the same wreck too (aligned, or a ship frame from markers of the same names; *Use the ship frame of*); each build a step of Undo. *Table*: measured and reconstructed, timber by timber (CSV, XLSX). *Export 3D* and *Plan with reconstruction*: the reconstruction translucent (`_ric`) and hatched, a hatch per way. |

Limits. A chunk in degrees (WGS 84) is refused: a frame needs metres (*GNSS > System and heights* brings the chunk to a
projected system; the import from Metashape does it by itself). No hydrostatics or plank expansion; the timbers found on the relief are proposals, never
accepted without you. Picks in the 3D view are not
offered directly: place two markers there and use *From two markers*. Heights over the model are good to about one edge of
the mesh.

Limits of the strakes. On sections of a mesh noisier than about 0.5 mm the widths are good to about a centimetre; at 1 mm
of noise about one side in fifteen may get a wrong count of lands (flagged as an uncertain station). A jump of 2-2.5 cm of
the rabbet at an end station, three or more adjacent shifted stations, or a gradual step are not flagged; a permanent step
of the keel (a broken or displaced hull, a scarf) flags every station on one side of it; above about 3 mm of scatter on the
rabbet a station may be flagged wrongly. A V of planking steeper than 80° within the keel's half width without a sole is
read as keel until it flattens; a keel side in line with the garboard is a side not read (`keel_unclear`). The strake model
needs at least three stations with a width on each side with lands. A hull seen from inside (planking under the frames,
the keel hidden by the keelson) gives uncertain stations: trace the seams on the relief of the hull instead.
The jumps of the rabbet and the garboard are looked for with five stations or more on a side: with fewer a true jump is
not caught. A new detection renames the earlier strake model's files `.old` and takes its strakes out of the timber
catalogue; save the planks exported by the Clinker Boat Tool under a name of your own (`planks.obj`), never
`strakes.obj` or `lands.obj` in the export folder.

Limits of the relief and the tracing. Everything read on the relief is a proposal: a loose timber with one edge read gives
a false seam; a timber narrower than 5 cm beside a land may be missed; a model showing the outside of the hull needs the
view "outside"; a long timber flush along almost a whole land renames the lands beyond it; a remnant in a strake of normal
width is proposed as `comento ?` (accepting it splits the strake); a land missing along the whole hull gives one band over
two strakes; noise beyond 2 mm breaks the seams and may add a strake; strays at the bow may take rank 1; on a real wreck
frames close together are often proposed as one wide doubtful frame (`ordinata ?`) and flat sediment as a keelson;
the strakes are bands between the seams read, boxes where seams are missing (a band far wider than its neighbours or
over the longitudinals is a doubtful `fascia B?`): correct them by hand. The relief is a height field: what is not seen
from above in the ship frame (undercuts, the underside of a timber, a timber under another, the outer face of the
planking of a wreck seen from inside) is in neither the proposals nor the solids. Trace every timber touching the one you build in 3D; at 5 mm a 12 m hull (1.9 M cells) takes about 0.75 GB to build in 3D, PLY 160 MB, IFC
20 s and 290 MB, and a 20 m hull (16 M cells) several GB of memory and an IFC of a few GB taking minutes: build the
relief at 1 cm for the 3D exports of a long hull (decimation is planned for a later version); a cancelled or stale build is said, and an export warns. Names may crowd where narrow strakes or several stringers end together.

Limits of the reconstruction. It is a hypothesis built on what is traced: an untraced timber lying on the planking enters
the sections and the faired hull passes over it; a twisted wreck is mirrored twisted; past the last station the faired
hull is a hypothesis of the breadth only, its height that of the last section; a seam without land lines is followed for
at most a metre past its points, then held straight (`uncertain`); an extended strake is cut where it grows wider
than 1.5 times its measured width or leaves the hull's half-breadth (`uncertain`); with several chunks the frames must carry the field's
numbers, and only the active chunk's files are written; without the mirror, where one side is lost and the other
preserved only the faired hull is reconstructed. The reconstruction is drawn and exported only when asked for (a box
in *Drawings*, *Plan of the wreck* and *Export 3D*, `--reconstruction`); when what it was built from changes it is
said to be built again, and in another ship frame the 3D export refuses it.

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
ppg plugin shipwreck drawings PROJECT.ppg [--scale 20] [--out FOLDER] [--reconstruction]
ppg plugin shipwreck timbers PROJECT.ppg [--table FILE]
ppg plugin shipwreck export PROJECT.ppg --out timbers.xlsx                      # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROJECT.ppg
ppg plugin shipwreck clinker detect PROJECT.ppg [--min-step MM] [--min-strake M]   # the lands and the land lines
ppg plugin shipwreck clinker strakes PROJECT.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROJECT.ppg --out FOLDER                        # files for the Clinker Boat Tool
ppg plugin shipwreck clinker import PROJECT.ppg --obj PLANKS.obj [--side-from y|name] [--no-deviation]
ppg plugin shipwreck clinker status PROJECT.ppg                                     # also says when the lands are stale
ppg plugin shipwreck relief build PROJECT.ppg [--cell MM]                           # the relief of the hull from above
ppg plugin shipwreck relief export PROJECT.ppg --kind hillshade --out FILE.tif [--world]
ppg plugin shipwreck propose PROJECT.ppg [--view inside|outside] [--min-step MM] [--opening M] [--top-hat MM]
ppg plugin shipwreck trace list PROJECT.ppg
ppg plugin shipwreck trace accept PROJECT.ppg ID... [--name NAME]                   # or --kind frame [--min-score 0.5]
ppg plugin shipwreck trace reject PROJECT.ppg ID...
ppg plugin shipwreck trace names PROJECT.ppg --table FILE
ppg plugin shipwreck keel PROJECT.ppg --siding MM --moulding MM [--from X --to X]   # or --none
ppg plugin shipwreck solids PROJECT.ppg [--thickness MM]
ppg plugin shipwreck export3d PROJECT.ppg --out FILE.obj|.glb|.ifc [--world] [--reconstruction]
ppg plugin shipwreck plan PROJECT.ppg [--background relief|ortho|none] [--scale N] [--out FOLDER] [--reconstruction]
ppg plugin shipwreck ends PROJECT.ppg [--stem X] [--stern X]                       # or --none
ppg plugin shipwreck frame-of PROJECT.ppg CHUNK [--chunk THIS]                     # the ship frame of another chunk
ppg plugin shipwreck chunks PROJECT.ppg                                            # the chunks of the reconstruction
ppg plugin shipwreck reconstruct PROJECT.ppg [--no-mirror] [--no-extension] [--no-frames] [--no-loft]
ppg plugin shipwreck reconstruct table PROJECT.ppg [--out FOLDER]
```

A table of timbers has a header: `kind`, `side`, `name`, `station`, `notes` and either `x, y` (a point) or `x1, y1, x2, y2`
(a line) in ship coordinates, or `wkt` in the coordinates of the chunk.

## License

GPL-2.0-or-later.
