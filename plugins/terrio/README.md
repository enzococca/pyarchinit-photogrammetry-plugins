# terrio

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/terrio/)**: the point cloud files and sources, the ground classification, the cleaning, colours and values from rasters, ellipsoidal and geoid heights, the reference rasters (also from Microsoft Planetary Computer), the exports. *[Italiano](README.it.md)*

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) built on [terrio](https://github.com/opengeos/terrio), geospatial I/O in pure
Rust: no GDAL, PROJ or PDAL to install, one wheel per platform (macOS, Windows x64 and ARM64,
Linux x86_64 and aarch64).

| Where | What |
|---|---|
| *Plugin > terrio > Importa nuvola di punti...* | LAS, LAZ, COPC, EPT, PLY and E57 files (also zipped), on disk or on the web (https, S3, Google Cloud, Azure; remote COPC and EPT are read in place), as the dense cloud of a new chunk (which the window shows as the active one) or of the active one (the current cloud is kept as a version). Colours, intensity (as grey) and classes come with the points; a limit of points, a choice of classes and a reference system for files without one. |
| *Plugin > terrio > Classifica il terreno (SMRF, CSF, PMF)...* | Ground (class 2) and other points (class 1) of the dense cloud with the PDAL-style filters of terrio: Simple Morphological Filter, Cloth Simulation Filter or Progressive Morphological Filter, each with its own defaults (SMRF slope 0.15 and threshold 0.5 m, PMF slope 1.0 and initial distance 0.15 m, CSF threshold 0.5 m); low noise (class 7) found first with ELM; the other points optionally in the ASPRS vegetation classes 3, 4 and 5 by their height above the ground. Only the points in classes 0, 1 and 2 change (and 3, 4, 5 with the vegetation): buildings, walls and the other archaeological classes (64 and up), noise, the classes of an imported lidar and the points labelled by hand stay as they are, and the noise already found is left out of the filters. The vegetation classes take everything above the ground that has no other class - standing walls, sections and spoil heaps too: classify the walls first (class 64) or correct by hand. A DEM built from class 2 is then a DTM. The summary names the classes in the language of the program. |
| *Plugin > terrio > Pulisci la nuvola (LOF, DBSCAN)...* | Isolated points by their local outlier factor (LOF above 3 with 20 neighbours) go to class 7 (noise); detached groups - DBSCAN clusters (radius 4 times the point spacing) holding less than 1 % of the cloud and more than 4 radii from any other point: birds, people, floating fragments - are removed or given a class (18 by default). The points are counted and shown in the 3D view (red, orange) before anything changes; the points labelled by hand never change, those already in classes 7 and 18 are left out; the cloud of before is kept as a version. |
| *Plugin > terrio > Colori e valori da un raster...* | The colours of the orthomosaic of the chunk (or of a version, a file, a COG on the web, in any reference system) for the points it covers - a lidar without colours, a cloud of another program - the cloud of before kept as a version; or a value of a band - the temperature of the thermal orthomosaic, an index (NDVI), the DEM, a band of the reflectance - as a dimension of the points (`temperature`, `ndvi`, `height`: a `float` property of the dense cloud PLY, NaN where the raster has no value), exported with the plugin's LAS, LAZ, COPC, EPT, PLY and E57. The raster is read tile by tile with terrio's `colorization`. |
| *Plugin > terrio > Valori dei punti nella vista 3D...* | such a value in colours (turbo, viridis, inferno, grey...) in the 3D view (program 1.3.4 or later), the cloud's colours untouched; or the dimension taken away |
| *Plugin > terrio > Quote ellissoidiche e sul geoide...* | The active DEM (or a kept version) or the dense cloud moved between ellipsoidal heights (GNSS) and heights above a geoid - EGM2008 (`EPSG:3855`), EGM96 (`EPSG:5773`), a national vertical CRS, or a geoid grid file (`.gtx`, `.tif`) - as a new version. The grids come from the PROJ CDN when missing (downloaded into `<plugin data>/grids`, or read in place); a height no grid moved is an error, never a silent copy. |
| *Plugin > terrio > Raster di riferimento dal web...* | A Cloud Optimized GeoTIFF by URL (or a STAC search, `STACIT:...`, or a file), or a collection of **Microsoft Planetary Computer** (Copernicus DEM GLO-30 and GLO-90, ALOS World 3D, NASADEM, USGS 3DEP, Sentinel-2 L2A, NAIP, or any other collection and asset: searched over the survey, the addresses signed by terrio), read only over the survey plus a margin, reprojected into the chunk's system and kept as a version of the DEM or of the orthomosaic (*Versioni*), named `Riferimento: <file>` (`Reference: <file>` in English) unless a name is given: the active one does not change. A DEM's heights can be moved into the survey's system on the way (the catalogue knows that Copernicus gives EGM2008 heights). |
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
ppg plugin terrio reference scavo.ppg --collection cop-dem-glo-30 --margin 100 --heights-to ellipsoid
ppg plugin terrio catalogue                                 # the Planetary Computer collections offered by name
ppg plugin terrio clean scavo.ppg --dry-run                 # count only
ppg plugin terrio clean scavo.ppg --remove                  # isolated -> 7, detached groups removed
ppg plugin terrio colour scavo.ppg --list                   # the rasters of the chunk
ppg plugin terrio colour scavo.ppg orthomosaic              # colours
ppg plugin terrio colour scavo.ppg thermal                  # the temperature as a dimension
ppg plugin terrio values scavo.ppg
ppg plugin terrio heights scavo.ppg --from ellipsoid --to EPSG:3855
ppg plugin terrio heights scavo.ppg --product point_cloud --from ellipsoid --to geoide_nazionale.gtx --activate
ppg plugin terrio export scavo.ppg nuvola.e57
ppg plugin terrio export scavo.ppg nuvola.ept               # a folder with ept.json
ppg plugin terrio export scavo.ppg ortofoto.pmtiles
```

## Installing

* In the program: *Strumenti > Plugin...*, `terrio`, *Installa / aggiorna*, *Riavvia ora*; then
  *Plugin > terrio > Installa terrio* once (about 20 MB from PyPI); the same action updates an older terrio.
* With pip, in the Python environment of the program (terrio comes with it):

  ```bash
  pip install "ppg-terrio @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/terrio"
  ```

The plugin is written for terrio 0.5.0 and pip installs `terrio>=0.5,<0.6`. An older terrio (0.3.2 and
the 0.4 releases, as the plugin folder of the application may hold) still runs the import, the ground
classification, the reference rasters and the exports; the cleaning, the colours and values, the heights and
the Planetary Computer need 0.5 and say so - the window offers to download it. terrio is young and its API
may still change before 1.0.

## Notes

* Coordinates: terrio works in the coordinates of the file or of the chunk; the plugin moves the
  points into the internal frame of the chunk (`chunk.transform`) and back. A new chunk made from
  a file takes a round origin as its frame, as the laser scans imported by the program do.
* E57 files are read with the pose of every scan applied and written as one scan with 16-bit
  colours; the program's own E57 import (`io.scans`, pye57) keeps the scans apart, where pye57
  is available.
* The ground filters run on the whole cloud in memory: a cloud of tens of millions of points
  needs a few gigabytes.
* Cleaning: on photogrammetric clouds of very uneven density, LOF also marks the edges between dense and
  sparse parts (the default threshold, 3, keeps most of them out), and tree canopies without their trunks are
  "detached groups" too: look at the points in the 3D view before applying, and prefer a class to removal.
* The values of the points are not known to the program: when it writes the cloud again (its own
  cleaning, classes, cropping) they go, and are taken again from the raster.
* Planetary Computer: used without an account, with its rate limits; a subscription key in
  `PC_SDK_SUBSCRIPTION_KEY` is sent when set. Sentinel-2 true-colour files are large: terrio fetches about
  200 MB per item for a window, so the catalogue reads one item (the newest under 10 % of clouds).
* Tests: `pip install -e . pytest` in the environment of the program, then `pytest tests`
  (`PPG_TEST_NETWORK=1` adds the test of the Planetary Computer and of the PROJ CDN).

Licence GPL-2.0-or-later. Author: Enzo Cocca.
