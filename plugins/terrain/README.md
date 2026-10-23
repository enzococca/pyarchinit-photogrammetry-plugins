# Terrain

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/terrain/)**: the DEM to prepare, the analyses step by step, how to read the results.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 or later): the terrain analyses of the DEM that
[WhiteboxTools](https://www.whiteboxgeo.com/) makes, on the DEM of the chunk and back into the
project. WhiteboxTools (John Lindsay, University of Guelph, MIT) is a program: its binary for this
computer (macOS Apple Silicon and Intel, Windows, Linux) is downloaded once into the plugin's
folder at the first analysis, about 40 MB; `PPG_WBT` names a binary of your own.

* **Hydrology**: the depressions of the DEM breached (least cost) or filled, D8 flow directions
  and accumulation, the stream network of the cells draining at least a given area, the basins.
  Products: the flow accumulation (log10 of the cells), the stream links and the basins as
  rasters; the network as lines and the basins as polygons in `terrain/hydrology.gpkg`, with the
  drained area and length of every link and the area of every basin.
* **Geomorphometry**: relative topographic position (TPI) and deviation from the mean height on a
  window, geomorphons (the ten landforms: flat, peak, ridge, shoulder, spur, slope, hollow,
  footslope, valley, pit) with a search distance, profile and plan curvature, ruggedness index,
  time in daylight over the year (a georeferenced project: the horizon angles of WhiteboxTools and a
  solar almanac of the plugin, open level ground at 1). Any metric can be computed on a coarser cell
  than the DEM's (the landforms of a field at 1 m, not its furrows at 5 cm): the cell goes into the
  name of the product (`geomorphons_30_c1`).
* **Depressions and mounds**: the nested depressions of the DEM by the level-set method (Wu et al.
  2019): every sink of the filled DEM is sliced from its spill level down and, at each level, the
  connected parts under it are the depressions of that level, nested in the ones above - with
  level, depth, area, volume, elongation and the polygon it lies in; the mounds (hills, tumuli)
  from the same on the inverted DEM. Rasters of the depth and of the nesting levels; the polygons
  in `terrain/depressioni.gpkg` / `terrain/rilievi.gpkg`. Written in NumPy and SciPy: no GDAL
  bindings needed.
* **Viewshed** from a point of the DEM with the height of the eye.
* The products in the *Terreno* (*Terrain*) pane of the project dock, named in the language of the
  interface, overlaid on the orthomosaic or DEM with palette and opacity (greys, diverging around
  zero, earth, heat, water, a colour per class), the vectors as layers of the view, exports as
  GeoTIFF and GeoPackage (*File > Esporta*), the rasters and tables in the QGIS project of the
  program. The progress bar follows every tool of WhiteboxTools while it runs; *Remove* asks
  first and keeps a GeoPackage another product still uses.
* `ppg plugin terrain hydrology|geomorph|depressions|viewshed|list|export|remove|install ...`.

```bash
pip install "ppg-terrain @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/terrain"
ppg plugin terrain hydrology scavo.ppg --threshold 20
ppg plugin terrain geomorph scavo.ppg tpi --size 15
ppg plugin terrain depressions scavo.ppg --min-area 0.5 --min-depth 0.1
ppg plugin terrain depressions scavo.ppg --mounds
ppg plugin terrain viewshed scavo.ppg 500120.5 4500812.0 --height 1.7
ppg plugin terrain export scavo.ppg terreno.gpkg
```

The products live in `<chunk data>/terrain/` on the grid and in the frame of the DEM (a local
project stays local: the tools read a copy with a stand-in CRS) and in `chunk.meta["terrain"]`.

Licence: GPL-2.0-or-later. WhiteboxTools is MIT-licensed and downloaded from whiteboxgeo.com.
