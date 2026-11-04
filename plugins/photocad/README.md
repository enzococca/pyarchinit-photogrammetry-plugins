# PhotoCAD

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 or later)
that sends the **orthophoto**, the **DEM** (as a hillshade) or the **orthophoto of an elevation** to
[PhotoCAD](https://photocad.ai) to be vectorised, and brings the drawing back **in place**: in the coordinates of
the chunk on the plan, in the (u, v) metres of the elevation (and, if asked, in 3D on its surface). The drawing
becomes a layer of the 2D view, a georeferenced DXF, a GeoPackage and an SVG; its closed outlines can become US or
USM.

PhotoCAD is an **external paid service**: the image leaves the computer and every conversion uses one credit
(about $2; the first $10 are free). Before each upload the plugin says so, with the credits and free conversions left and the size of the
image, and waits for a yes. Nothing to install beyond the program: the standard library talks to the API; ezdxf,
rasterio, Pillow and shapely come with the program.

| Where | What |
|---|---|
| *Plugin > PhotoCAD > Vettorializza l'ortofoto...* | The whole orthophoto, the part visible in the view, or an area drawn on it. Options: detailed or simplified drawing, remove the background. |
| *... > Vettorializza il DEM...* | The DEM rendered as a hillshade (light from the north-west), same choices. |
| *... > Vettorializza il prospetto mostrato...* | The orthophoto of the elevation or section shown in the 2D view, in its (u, v); optionally also in 3D on its surface. |
| *... > Converti in US / USM...* | The PhotoCAD polygons selected on the view (or a whole PhotoCAD layer) become stratigraphic units of the active chunk - on the plan, or in the drawings of the elevation shown. Kind (US, USM, ...) and an optional first number; closed polylines count as polygons, open lines are skipped and counted. One entry of *Edit > Undo*. |
| *(automatic)* Stones | PhotoCAD draws stones as open lines: their ends within 2 px are joined, the lines noded and polygonized; faces under 50 px or mostly on white (no data) are left out. The stones go to `polygons.gpkg` and to a second layer **"PhotoCAD: \<name\> (pietre)"**; *Converti in US / USM* takes them by default (`--from lines` for PhotoCAD's own closed outlines). Again with other values: `ppg plugin photocad polygonize PROJECT ID --snap-px 3 --min-area-px 80`. |
| *... > Importa come disegni liberi...* | (program 1.7.2) The features of a conversion as free 3D drawings in the layer *PhotoCAD*: on the plan at the height of the DEM, on an elevation on its surface. |
| *... > Recupera una conversione...* | A conversion already paid for downloaded and georeferenced again, without a new upload: one that finished while the network failed or the download broke, or that was cancelled after PhotoCAD had finished. |
| *... > Esporta una conversione...* / *Rimuovi una conversione...* | A copy as DXF, GeoPackage or SVG; a conversion out of the project with its files (the units made from it stay). |
| *... > Chiave API PhotoCAD...* | The key in the system keychain: *Salva nel portachiavi*, *Dimentica*. |
| *... > Credito PhotoCAD* | The credits and the free conversions left on the account. |

Every conversion is a layer **"PhotoCAD: \<name\>"** of the 2D view (polygons and lines, selectable, in the Layer
pane), on the plan or on the elevation it was made on, drawn again whenever the raster is loaded again.

## The API key

From the environment variable `PHOTOCAD_API_KEY` when set, else from the **system keychain** (service
`pyarchinit-photogrammetry`, user `photocad`: the same family of entries where the program's assistant keeps the
Claude and OpenAI keys). It is never written into the settings, the project, the logs, `conversion.json` or an error
message (masked as `pcad_live_…abcd`). Without a keychain (the `keyring` library comes with the program's
`assistant` extra) only the environment variable works.

## How the drawing comes back in place

The image is cut into **tiles at the raster's native resolution** - at most 2400 px on the long side, about 10 %
overlap, one conversion each (a strip under 600 px is enlarged up to 4 times, thinner than that it is not sent:
PhotoCAD fails on thin strips). Before sending, the confirmation says how many conversions it takes and refuses when
the account has fewer credits and free conversions; `--max-tiles` (6) guards against a long image taking many.

PhotoCAD's drawing does not come in the image's frame: its page is the box of the **subject** it found, in
millimetres at a natural size (whatever `units` asks). So each tile's drawing is downloaded in mm and placed by
**registration**: the page onto the image's content box (one scale, proportions within 2 %), then offset and scale
refined (+-24 px, +-3 %) by bringing the lines onto the image's edges (the strength of the gradient under them,
maximised to a fortieth of a pixel). The residual (median distance of the lines from the image's edges, in pixels)
is kept in `conversion.json`; over 2 px, or under 60 % of the lines near an edge, the drawing is refused and nothing
is placed. The tiles are then merged: from each tile what is nearer its own centre (the copies in the overlaps once).
On an elevation each vertex can also go to the world, at the depth read from the elevation's relief.

Files, in `<chunk data>/photocad/<id>/`: per tile `t0/`, `t1/`...: `image.jpg` (what was sent), `photocad_mm.dxf` (as
PhotoCAD gave it), `geometry.json`, `drawing.svg` with `drawing.svg.json` (where it lies), `drawing.dxf` (the tile
placed); for the whole: `drawing.dxf` (georeferenced, PhotoCAD's layers kept), `drawing.gpkg` (table `photocad`:
polygons and lines with the PhotoCAD layer, length and area; in the CRS of the chunk, or without CRS for an
elevation; `photocad_3d` with Z for an elevation in 3D), `drawing_3d.dxf`, `conversion.json` (source, tiles with
their PhotoCAD ids, idempotency keys and registrations, date, credits used). Each upload carries an
`Idempotency-Key` saved before it is sent: an upload asked again (a 429, a lost connection, `fetch` after a crash)
never charges twice.

## Command line

```bash
ppg plugin photocad key set            # typed without echo, into the keychain
ppg plugin photocad key status|forget
ppg plugin photocad credit
ppg plugin photocad list scavo.ppg     # also lists the elevations: elevation:<id>
ppg plugin photocad convert scavo.ppg --source ortho --out pianta.dxf
ppg plugin photocad convert scavo.ppg --source elevation:3236beab --3d --out prospetto.gpkg
ppg plugin photocad convert scavo.ppg --source dem --bbox 500010,4000020,500060,4000070 --simplified --yes
ppg plugin photocad fetch scavo.ppg --id <id or PhotoCAD id> [--out disegno.dxf]
ppg plugin photocad units scavo.ppg <id> --tipo USM --start 101
ppg plugin photocad export scavo.ppg <id> disegno.svg
ppg plugin photocad remove scavo.ppg <id>
```

`convert` asks for confirmation at the prompt unless `--yes` (or the project was told not to ask again); options
`--chunk NAME`, `--remove-background`, `--pdf`, `--name`, `--max-tiles N`.

## Limits

- The placement rests on the image having edges where the drawing has lines (stones, walls, cuts: what PhotoCAD
  draws); a drawing of a flat, featureless image may be refused.
- A large orthophoto at native resolution takes many tiles (many conversions): send a part (visible area or drawn
  area), or raise `--max-tiles` knowingly.
- 20 conversions a minute; a refused or failed conversion is said in plain words (a failed one is refunded by
  PhotoCAD).

License: GPL-2.0-or-later.
