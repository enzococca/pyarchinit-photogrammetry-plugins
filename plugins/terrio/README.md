# terrio

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/terrio/)**: the point cloud files and sources, the ground classification, the reference rasters, the exports.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) built on [terrio](https://github.com/opengeos/terrio), geospatial I/O in pure
Rust: no GDAL, PROJ or PDAL to install, one wheel per platform (macOS, Windows x64 and ARM64,
Linux x86_64 and aarch64).

| Where | What |
|---|---|
| *Plugin > terrio > Importa nuvola di punti...* | LAS, LAZ, COPC, EPT, PLY and E57 files (also zipped), on disk or on the web (https, S3, Google Cloud, Azure; remote COPC and EPT are read in place), as the dense cloud of a new chunk (which the window shows as the active one) or of the active one (the current cloud is kept as a version). Colours, intensity (as grey) and classes come with the points; a limit of points, a choice of classes and a reference system for files without one. |
| *Plugin > terrio > Classifica il terreno (SMRF, CSF, PMF)...* | Ground (class 2) and other points (class 1) of the dense cloud with the PDAL-style filters of terrio: Simple Morphological Filter, Cloth Simulation Filter or Progressive Morphological Filter, each with its own defaults (SMRF slope 0.15 and threshold 0.5 m, PMF slope 1.0 and initial distance 0.15 m, CSF threshold 0.5 m); low noise (class 7) found first with ELM; the other points optionally in the ASPRS vegetation classes 3, 4 and 5 by their height above the ground. Only the points in classes 0, 1 and 2 change (and 3, 4, 5 with the vegetation): buildings, walls and the other archaeological classes (64 and up), noise, the classes of an imported lidar and the points labelled by hand stay as they are, and the noise already found is left out of the filters. The vegetation classes take everything above the ground that has no other class - standing walls, sections and spoil heaps too: classify the walls first (class 64) or correct by hand. A DEM built from class 2 is then a DTM. The summary names the classes in the language of the program. |
| *Plugin > terrio > Raster di riferimento dal web...* | A Cloud Optimized GeoTIFF by URL (or a STAC search, `STACIT:...`, or a file) read only over the survey plus a margin, reprojected into the chunk's system and kept as a version of the DEM or of the orthomosaic (*Versioni*), named `Riferimento: <file>` (`Reference: <file>` in English) unless a name is given: the active one does not change. |
| *File > Esporta* | the dense cloud as **E57** or as an **EPT** dataset (a folder), the orthomosaic as **PMTiles** or **MBTiles** (Web Mercator tiles for web maps) |
| *Plugin > terrio > Installa terrio...* | the packaged application has no pip: the plugin downloads the terrio wheel of this computer from PyPI (its SHA-256 checked) into its own folder |

From the command line:

```bash
ppg plugin terrio version
ppg plugin terrio install                                   # into the plugin folder, without pip
ppg plugin terrio import scavo.ppg scansione.e57 --label "laser"        # a new chunk; the active one stays
ppg plugin terrio import scavo.ppg scansione.e57 --label "laser" --activate   # ... and becomes the active one
ppg plugin terrio import scavo.ppg https://host/lidar.copc.laz --chunk rilievo --max-points 5000000
ppg plugin terrio ground scavo.ppg --method csf --resolution 0.5 --noise --vegetation
ppg plugin terrio ground scavo.ppg --method pmf             # PMF's defaults: slope 1.0, initial distance 0.15
ppg plugin terrio reference scavo.ppg https://host/dem_regionale.tif --product dem --margin 50 --resolution 1
ppg plugin terrio export scavo.ppg nuvola.e57
ppg plugin terrio export scavo.ppg nuvola.ept               # a folder with ept.json
ppg plugin terrio export scavo.ppg ortofoto.pmtiles
```

## Installing

* In the program: *Strumenti > Plugin...*, `terrio`, *Installa / aggiorna*, *Riavvia ora*; then
  *Plugin > terrio > Installa terrio* once (about 17 MB from PyPI).
* With pip, in the Python environment of the program (terrio comes with it):

  ```bash
  pip install "ppg-terrio @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/terrio"
  ```

The plugin is written for terrio 0.4.0 and accepts the 0.3.2 and 0.4 releases (`terrio>=0.3.2,<0.5`):
terrio is young and its API may still change before 1.0.

## Notes

* Coordinates: terrio works in the coordinates of the file or of the chunk; the plugin moves the
  points into the internal frame of the chunk (`chunk.transform`) and back. A new chunk made from
  a file takes a round origin as its frame, as the laser scans imported by the program do.
* E57 files are read with the pose of every scan applied and written as one scan with 16-bit
  colours; the program's own E57 import (`io.scans`, pye57) keeps the scans apart, where pye57
  is available.
* The ground filters run on the whole cloud in memory: a cloud of tens of millions of points
  needs a few gigabytes.
* Tests: `pip install -e . pytest` in the environment of the program, then `pytest tests`.

Licence GPL-2.0-or-later. Author: Enzo Cocca.
