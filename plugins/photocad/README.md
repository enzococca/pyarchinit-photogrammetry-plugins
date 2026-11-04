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

The image covers a block of whole cells of the raster, reduced by an integer factor so that its long side is at most
2400 px (PhotoCAD reduces to that anyway): each pixel of the image is a known square of the survey and its lower-left
corner a known point. The DXF is asked in metres (`units=m`), without margin (`margin=0`), with the long side of the
image in metres (`longest`): PhotoCAD draws from the lower-left corner of the image, Y up, so moving the drawing by
that corner puts it in the coordinates of the chunk (or the (u, v) of the elevation). On an elevation each vertex can
also go to the world, at the depth read from the elevation's relief.

Files, in `<chunk data>/photocad/<id>/`: `image.jpg` (what was sent), `photocad.dxf` (as PhotoCAD gave it),
`drawing.dxf` (georeferenced, PhotoCAD's layers kept), `drawing.gpkg` (table `photocad`: polygons and lines with the
PhotoCAD layer, length and area; in the CRS of the chunk, or without CRS for an elevation; `photocad_3d` with Z for an
elevation in 3D), `drawing.svg` with `drawing.svg.json` (where it lies: the SVG keeps PhotoCAD's frame, Y down),
`drawing_3d.dxf`, `drawing.pdf` if asked, `conversion.json` (source, bounds, pixel, date, credits used).

A paid conversion is never lost: its PhotoCAD id is written in the project (and in `conversion.json`) right after
the upload, before PhotoCAD finishes; whatever goes wrong afterwards, *Recupera una conversione...* or
`ppg plugin photocad fetch` gets it. Cancelling while PhotoCAD still works deletes the conversion there. An upload is
never sent twice (only a 429 is asked again). The key goes only to PhotoCAD's own address: a file served from another
address (cloud storage) is downloaded without it.

Before placing the drawing the plugin checks that it is in the image's frame: inside the image, the page with the
image's proportions, and the drawing asked in metres matching the same drawing at its natural size (same scale on both
axes, no offset). If not, nothing is placed and the files stay in the folder for a diagnosis.

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
`--chunk NAME`, `--remove-background`, `--pdf`, `--name`.

## Limits

- PhotoCAD's own frame (origin, Y up, `longest` applied to the image, no margin) follows its documentation; the
  plugin was tested against a fake server that behaves as documented, not against photocad.ai.
- The drawing is as good as the image: an orthophoto of a few thousand pixels reduced to 2400 px loses detail - send
  a part (visible area or drawn area) for detail.
- 20 conversions a minute; a refused or failed conversion is said in plain words (a failed one is refunded by
  PhotoCAD).

License: GPL-2.0-or-later.
