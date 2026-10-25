# Samgeo

*[Italiano](README.it.md)*

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.5.0 or later): what you write - *pietre*, *mattoni*, *tegole e coppi*, or any English phrase -
found in the orthomosaic of the chunk, or in any raster the raster calculator reads. It follows the
recipe of `LangSAM` in [segment-geospatial](https://github.com/opengeos/segment-geospatial)
(Qiusheng Wu): [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (IDEA
Research) finds the boxes of what is written, [SAM 2.1](https://github.com/facebookresearch/sam2)
(Meta) draws the mask in each box. Both run on your computer, with open weights and no account.

What the models find is a **layer of proposals** on the 2D view, to check before it becomes
anything else:

| From a run | What you get |
|---|---|
| **The layer** `samgeo: <name>` | The objects found, each with its number, the text, the phrase that found it, the score (Grounding DINO's confidence) and the area; drawn again on every raster the view shows; a proposal deleted on the view (*Seleziona* and Canc) leaves the run too |
| **US** | The proposals (those selected on the view, those not yet units, or all) become stratigraphic units of the chunk, of the kind chosen (US, USM...), with the text and the score as attributes (`samgeo_prompt`, `samgeo_score`): in the live QGIS link, numbered by *Proponi US e rapporti*, in the US forms of PyArchInit; one entry of *Modifica > Annulla* for all of them. The proposals made units stay in the layer as outlines; when they all are, the layer is hidden |
| **Classes of the dense cloud** | The points inside the proposals (in plan view) get the class `samgeo: <name>` (codes from 150); the points labelled by hand keep theirs; done again, the class follows the proposals left |
| **GeoPackage** | The runs chosen, a layer each (`MULTIPOLYGON`: `n`, `prompt`, `label`, `score`, `quality`, `area_m2`) |
| **The mask** | The objects numbered in a GeoTIFF the raster calculator reads by itself (`samgeo_<run>_mask`) |

A run asks for:

* **what to look for**: a ready entry (*pietre*, *muri*, *mattoni*, *tegole e coppi*, *frammenti
  ceramici*, *ossa*, *tessere di mosaico*, *alberi*, *cespugli*, *persone*, *attrezzi e secchi*,
  *teli e coperture*), given to the models in English, or your own text - English works best;
  several things separated by commas (`stone, brick`);
* **the raster**: the orthomosaic, or any raster of the chunk (a relief visualisation of the DEM, for
  mounds and ditches: experimentally - the models were trained on photographs);
* **where**: the whole raster, a polygon you draw (the program's own drawing tool), or the part shown
  in the 2D view;
* the **least confidence** (0.30), the **smallest** and the **largest area** in m², the
  **resolution** (the raster's, not finer than 1 cm/px - plenty for stones; coarser is faster), the
  **models**: normal (Grounding DINO base, SAM 2.1 small), fast (tiny, tiny), precise (base, large).

## How a run works

1. The area is read at the resolution of the run, in tiles of 1024 px overlapping by 128 px. Three
   bands of 8 bits are read as they are; any other raster (16 bits, reflectances, one band) is
   stretched between the 2nd and the 98th percentile of the area, one band as grey.
2. Each tile goes to the models; each mask becomes a polygon in the coordinates of the chunk.
3. An object cut by the border of a tile is joined with its other part from the neighbouring tile
   (they agree in the strip the two tiles share); two boxes on the same object leave one proposal;
   when the area has more than one tile, a box as large as a whole tile is left out (Grounding DINO
   often answers "the whole image" beside the objects). Then the score and area filters and, with a
   polygon, only what lies inside it.
4. The run is kept in the project, its files in the chunk folder: `samgeo/<run>.gpkg` and
   `samgeo/<run>_mask.tif`.

## Where

* the **Samgeo** pane (beside the workspace; *Vista > Pannelli* opens it again): what to look for,
  *Trova*, the runs of the active chunk (the box shows or hides a layer, a double click zooms to it),
  *US...*, *Classi della nuvola*, *GeoPackage...*, *Inquadra*, *Rimuovi*, *Installa i modelli...*;
* *Plugin > Samgeo* in the menu; the runs under each chunk in the workspace;
* the command line:

```bash
ppg plugin samgeo status
ppg plugin samgeo install                                  # PyTorch and the models (uv)
ppg plugin samgeo segment scavo.ppg pietre
ppg plugin samgeo segment scavo.ppg "roof tile" --area 2480000,4500000,2480004,4500003 --units
ppg plugin samgeo segment scavo.ppg muri --resolution 0.02 --min-area 0.5 --models precise --json
ppg plugin samgeo segment scavo.ppg mound --raster dem --threshold 0.25
ppg plugin samgeo list scavo.ppg
ppg plugin samgeo export scavo.ppg samgeo.gpkg --run pietre
ppg plugin samgeo remove scavo.ppg pietre
```

`--chunk LABEL` works on another chunk than the active one; `--raster` takes the name of a raster in
the raster calculator.

## How the models run

PyTorch never runs in the program's process (its OpenMP runtime and pycolmap's abort the process
when both start): the models run in a process of their own (`ppg_samgeo/worker.py`), started once
for a run and fed one tile at a time; cancelling stops it at once. The worker runs with:

* the environment of the AI plugins, installed by the plugin on request with
  [uv](https://github.com/astral-sh/uv): Python 3.12, PyTorch (the build for your graphics card,
  chosen by `uv --torch-backend auto`: CUDA with an NVIDIA card, Metal on Apple Silicon, else the
  processor), transformers and sam2, about 2 GB, in
  `<app data>/pyarchinit-photogrammetry/geoai-env` (`PPG_GEOAI_ENV` to put it elsewhere). The plugin
  geoai uses the same environment, so PyTorch is there once. The packaged application works this way;
* the Python of the program, when PyTorch, transformers and sam2 are installed there.

The weights of the models (about 1 GB: Grounding DINO base and SAM 2.1 small) are downloaded from
Hugging Face at the first run and kept in its cache (`HF_HOME`). On the processor a tile takes tens
of seconds; a GPU is far faster.

## Limits

* English texts work best; the ready entries are translated by the plugin.
* The models find objects - stones, bricks, tiles, trees, people, tools - far better than the
  contrast between two soils: that is for geoai, which learns from your drawings.
* With an area of a single tile (a small polygon) Grounding DINO may answer with the whole area:
  delete that proposal, or set a largest area.
* SAM 3 (one model that finds every instance of a concept) comes in a next version, for those
  who have access to its weights.

## Installing

* In the program: *Strumenti > Plugin...*, `samgeo`, *Installa / aggiorna*, *Riavvia ora*; then
  *Installa i modelli...* in its pane.
* With pip, in the Python environment of the program:

  ```bash
  pip install "ppg-samgeo @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/samgeo"
  ```

Tests: `pip install -e . pytest` in the environment of the program, then `pytest tests`; they use a
fake engine of the worker (the bright blobs of a synthetic orthophoto are the stones), without
PyTorch.

Licence: GPL-2.0-or-later. Grounding DINO, SAM 2.1, PyTorch and transformers (Apache-2.0 and BSD)
run in their own environment. Author: Enzo Cocca.
