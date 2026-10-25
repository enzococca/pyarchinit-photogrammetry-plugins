# pyarchinit-photogrammetry plugins

*[Italiano](README.it.md)*

The plugins of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later) and their index, `index.json`, which the program reads in *Strumenti > Plugin* and
with `ppg plugins available`.

| Plugin | What it does |
|---|---|
| [`geoarchaeology`](plugins/geoarchaeology) - Geoarchaeology | Boreholes, cores and trench logs (CSV of collars and intervals) on the survey: stratigraphic columns, contact surfaces and thicknesses by kriging (variogram fitted by weighted least squares, ordinary or universal kriging, the kriging variance; IDW for a few boreholes), kept under the DEM and in order; sections drawn on the view or through boreholes, with the logs projected; a 3D model of the units with their volumes, clipped to the US drawn; GeoPackage, GeoTIFF, DXF, PLY/OBJ, CSV, in the QGIS project |
| [`geophysics`](plugins/geophysics) - Geophysics | Geophysical grids (magnetometry, resistivity, GPR time slices, EM) from Surfer GRD, XYZ/CSV, GeoTIFF or folders of slices, georeferenced with the markers or the pegs' coordinates, filtered (despike, destripe, high-pass, clip, fill), overlaid on the orthomosaic and the DEM with opacity, exported as GeoTIFF, Surfer GRD and GeoPackage, in the QGIS project |
| [`terrain`](plugins/terrain) - Terrain | Terrain analysis of the DEM with WhiteboxTools (downloaded once): hydrology (stream network and basins), geomorphometry (topographic position, geomorphons, curvatures, ruggedness, daylight), nested depressions and mounds with depth, volume and elongation, viewshed; overlaid on the orthomosaic and DEM, exported as GeoTIFF and GeoPackage, in the QGIS project |
| [`gis-archive`](plugins/gis-archive) - Archivio GIS | The survey of a project - spot heights, contour lines, units drawn, sections, stones, markers, DEM and orthomosaic - into a GIS database outside PyArchInit: GeoPackage, PostgreSQL/PostGIS or SpatiaLite |
| [`powerlines`](plugins/powerlines) - Powerlines | Overhead powerlines in the dense cloud: every conductor as a catenary with its length, sag, clearance from the ground and distance from the nearest obstacle, the poles, the points of wires and poles classified (14, 15); GeoPackage, DXF and CSV exports |
| [`spirula`](plugins/spirula) - Spirula | 3D Gaussian splatting of the aligned photos with [Spirula Studio](https://github.com/harry7557558/spirula-studio) (Vulkan: NVIDIA, AMD, Intel and Apple GPUs): the splat as a product of the chunk, a mesh from it, the viewer |
| [`terrio`](plugins/terrio) - terrio | Point clouds of any format (LAS, LAZ, COPC, EPT, PLY, E57, also from the web) into the project, ground classification with the SMRF, CSF and PMF filters, reference rasters from the web as versions of the DEM or orthomosaic, E57, EPT, PMTiles and MBTiles exports, through [terrio](https://github.com/opengeos/terrio) (pure Rust, no GDAL or PDAL) |
| [`blender-em`](plugins/blender-em) - Blender and EMtools | The 3D proxies of the stratigraphic units and the survey into an Extended Matrix project (`EM/proxies/<unit>.glb`, `RB/`), named after the nodes of the em.json of PyArchInit or EMStudio, which EMtools pairs by name; Blender opened on the project and kept up to date while the proxies change (what was changed in Blender never overwritten); the aligned cameras in Blender with their photos undistorted; what is modelled in Blender back into the chunk as the proxies of the units, with their volume; the selection shared live with [EMtools](https://github.com/zalmoxes-laran/EM-blender-tools) 1.6 through its sync WebSocket, and its command that makes the proxy of a unit |
| [`samgeo`](plugins/samgeo) - Samgeo | What you write found in the orthomosaic or another raster with Grounding DINO + SAM 2.1 or optional SAM 3 (SamGeo3): proposals to review, then stratigraphic units, dense cloud classes, a GeoPackage or a mask. SAM 3 requires Meta-approved access, acceptance of its separate license and the user's Hugging Face token |
| [`geoai`](plugins/geoai) - GeoAI | Train RGB segmentation classes from reviewed archaeological unit drawings, validate on a separate area and apply the model to produce proposals to review, export or convert into units. Shares the AI environment with Samgeo |

## Installing a plugin

* In the program: *Strumenti > Plugin...*, the plugin in the lower list, *Installa / aggiorna*,
  then *Riavvia ora*. Works in the packaged application too, which has no pip.
* From the command line: `ppg plugins install gis-archive`, then `ppg plugins list`.
* With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-gis-archive @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/gis-archive"
  ```

Plugins run with the rights of the program: the index lists only plugins that were reviewed, and
the program installs a wheel only when its SHA-256 matches the one in the index.

## Writing a plugin

The guide is [docs/plugins.md](https://github.com/enzococca/pyarchinit-photogrammetry/blob/main/docs/plugins.md)
in the repository of the program; [`template/`](template) is a plugin to start from (menu actions, work
in the background with the progress bar, a command, an export format, settings, tests).
`ppg plugins new NAME` (or *Strumenti > Plugin... > Nuovo plugin...* in the program) writes the same
project with your name, the parts you choose and its tests, ready for `pip install -e .`.

## Submitting a plugin

A plugin can live in its own repository or here, under `plugins/`.

* **Your repository.** Build a pure-Python wheel (`python -m build --wheel`), attach it to a
  release, and open a pull request adding its entry to `index.json`. The entry can be made from the
  wheel:

  ```bash
  python scripts/make_entry.py https://github.com/you/ppg-thing/releases/download/v0.1.0/ppg_thing-0.1.0-py3-none-any.whl --write
  ```

* **Here.** A pull request adding `plugins/<name>/` (package, `pyproject.toml`, `README.md`,
  tests, `entry.json` with title, description and Italian texts). Once merged, its versions are
  published as GitHub releases tagged `<name>-v<version>`: the workflow `release` builds the wheel,
  attaches it to the release and commits the entry into `index.json` on `main` by itself (texts from
  `entry.json`): the program lists the new version at its next *Aggiorna l'elenco*.

An entry of the index ([schema](schema/index.schema.json)):

```json
{
  "name": "gis-archive",
  "version": "0.1.0",
  "title": "GIS archive",
  "description": "Spot heights, contour lines, units, sections, stones, markers, DEM and orthomosaic into a GeoPackage, PostgreSQL/PostGIS or SpatiaLite database",
  "i18n": {"it": {"title": "Archivio GIS", "description": "Quote, curve di livello, US, ..."}},
  "author": "Enzo Cocca",
  "homepage": "https://github.com/enzococca/pyarchinit-photogrammetry-plugins/tree/main/plugins/gis-archive",
  "license": "GPL-2.0-or-later",
  "api": 1,
  "min_app": "1.1.0",
  "requires": [],
  "wheel": "https://github.com/.../ppg_gis_archive-0.1.0-py3-none-any.whl",
  "sha256": "..."
}
```

* `name` is the name of the entry point in the group `pyarchinit_photogrammetry.plugins`;
* `api` the plugin API it was written for, `min_app` the oldest version of the program it works with;
* `requires` the modules it imports beyond those of the program and cannot fetch by itself (the
  installation is refused without them: the packaged application cannot add modules; terrio, which
  downloads its own wheel from PyPI, says `[]`);
* `i18n` title and description in other languages, shown when the interface speaks them.

Each pull request is checked by the workflow `checks`: the format of the index, the `__version__` of
every plugin of this repository against the version of its `pyproject.toml` (they must be the same:
the program shows the version a plugin reports beside the one of the index), then each wheel is
downloaded (hash, version, entry point, no path outside its folder) and the plugin is started with
the program; the plugins of this repository run their tests, PostGIS included. Locally:

```bash
pip install jsonschema
python scripts/check_index.py --load        # with pyarchinit-photogrammetry installed
```

The plugins must be under a licence compatible with the GPL (they run inside a GPL program); those
of this repository are GPL-2.0-or-later.

## Layout

```
index.json                 the index the program reads
schema/index.schema.json   its format
plugins/<name>/            the plugins maintained here
template/                  a plugin to start from
scripts/check_index.py     checks of the index, the wheels and the plugins
scripts/make_entry.py      the entry of a wheel
.github/workflows/         checks (pull requests), release (wheels of the plugins of this repository)
```

Licence: GPL-2.0-or-later. Author: Enzo Cocca.
