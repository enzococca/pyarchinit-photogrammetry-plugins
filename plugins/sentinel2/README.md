# Sentinel-2

*[Italiano](README.it.md)*

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 or later): Sentinel-2 time series over the area of a survey, to look for **crop marks**.
Buried walls and ditches show through the crop above them in some weeks of some years: over a
wall the roots find less water and the crop is weaker (a lower index, a *negative* mark), over a
ditch they find more and the crop is greener (a *positive* mark). At 10 m a pixel, Sentinel-2 sees
the larger ones (enclosures, roads, villas, ditches some metres wide) and, above all, *when* the
fields show them: two satellites pass every five days, year after year.

* **The area**: the footprint of the chunk's DEM or orthomosaic, or a polygon drawn on the view
  with the program's own tool, with a margin around it (300 m by default: the background of a
  crop-mark map needs the fields around the site). A chunk in a local system (LOCAL) has no place
  on the Earth and cannot be searched: the plugin says so (a box of longitudes and latitudes can
  still be given from the command line).
* **The search** in a STAC catalog: [Earth Search](https://earth-search.aws.element84.com/v1) by
  Element 84 by default (the collection `sentinel-2-l2a`: public Cloud-Optimized GeoTIFFs on AWS,
  no account, no key), any STAC API by its URL. The standard library talks to the catalog, so the
  packaged application needs nothing; [pystac-client](https://github.com/stac-utils/pystac-client)
  is used when it is installed. The tiles of one passage of a satellite become one *acquisition*;
  the cloud cover of the tiles filters them.
* **The reading**: only the window of the area and its margin, from the COGs over HTTPS (GDAL's
  `/vsicurl/`, with retries), bands B02, B03, B04, B05, B08, B8A, B11 and the scene classification
  (SCL), put on one grid of 10 m in the UTM zone of the area (the 20 m bands interpolated, the
  classes taken from the nearest pixel, a tile of another zone warped, two tiles of one passage
  joined). The scene classification is read first: clouds, cirrus and their shadows (with 20 m
  around them), snow and every class not kept as clear are masked; an acquisition with less than
  half of the area clear (by default) is left out before its other bands are downloaded. What was
  read is cached in the project: **a second look needs no network**. With the window of a site of
  a few hundred metres, an acquisition is about 16 MB to download (the COGs are read by internal
  tiles of 1024 pixels) and about 10 seconds; the cache keeps about 100 KB of it.
* **The cache**: the acquisitions read can be taken out of it - those selected in the table of the
  pane (*Remove from the cache*, also in its context menu) or all of them (*Empty the cache...*),
  after a question. Their files are deleted, they stay in the search as *to read* and the next
  *Read and compute* reads them again; indices and series are computed again from the others. When
  composites or crop-mark maps were made from them, the question asks whether to remove those too
  or to keep them: kept, they are marked *dates no longer cached* (they are what was computed then,
  but the cache cannot make them again), and the mark goes when the acquisitions are read again.
* **The indices** of every clear date: NDVI = (B08 - B04) / (B08 + B04), vigour of the vegetation
  (10 m); NDRE = (B8A - B05) / (B8A + B05), chlorophyll (red edge, 20 m); NDMI = (B8A - B11) /
  (B8A + B11), water in the leaves (20 m). Clear cells only; one GeoTIFF per index with a band per
  date.
* **The series** over the area: per date and per month (median, 10th, 25th, 75th and 90th
  percentiles, clear part of the area) as CSV, and its chart (PNG): when the fields green up,
  when they dry, when a mark can show.
* **Composites**: the median of an index over a period, cell by cell: a season of a year
  (`2023-summer`; the winter of a year is December before, January and February), a season of every
  year (`summer`), a year, a month (`2023-06`), a span of days (`2023-04-01/2023-06-30`), a date.
* **Crop-mark maps**: the index of a date, or the median of a period, minus its smooth
  background (a Gaussian, sigma 50 m by default, or a median of twice that window), divided by
  the robust spread of what is left (1.4826 times the median absolute deviation): a z-score, near
  0 over even fields, under -2 where the vegetation is weaker than around it (walls, roads,
  compacted ground) and over 2 where it is greener (ditches, pits). The marks are lines a few
  pixels wide. They read as an index does, in the pane and in QGIS alike: red-yellow-green from
  -3 (red, weaker vegetation) to 3 (green, greener); the legend under the view names the two ends
  of whatever palette is chosen.
* **On the view**: an index of a date or a period, natural colour (B04, B03, B02) of a date, a
  composite or a crop-mark map over the orthomosaic or the DEM of the chunk, with the palettes of
  the program's vegetation indices (red-yellow-green, greys, viridis, turbo) or a diverging one,
  and an opacity; every Sentinel-2 pixel stays a square. *Show the whole area* puts the area and
  its margin on an empty frame, to see beyond the orthomosaic. The area and the window read are
  drawn in the layer *Sentinel-2: area*; both follow the view when it shows another raster.
* **Products and exports**: every composite and crop-mark map is a GeoTIFF on the grid of the
  scenes (UTM) and a copy in the chunk's coordinate system, on a grid aligned with its DEM or
  orthomosaic (cells of about 2.5 m, transparent where there is no value); the copies go into the QGIS
  project of the program. *Export everything* writes the stack of every index and date (one
  GeoTIFF), the series (CSV per date and per month), the chart and the products; *File > Esporta*
  has the series (CSV) and the stack (GeoTIFF).
* The **Sentinel-2** pane of the project dock does it all, in the background with the progress
  bar (cancel stops the readings under way); *Plugin > Sentinel-2* has the main actions.
* **In the project's tree** (program 1.3.4 or later): a *Sentinel-2* node under every chunk the
  plugin worked on, with the acquisitions read (date and clear part of the area), the indices by
  date, the series, the composites and the crop-mark maps. A double click shows one over the view
  (natural colour for an acquisition, the index of a date, a product; the chart for the series) and
  the pane follows; the context menu has *Show*, *Hide*, *Export...*, *Open in QGIS*, *Remove* and,
  for the acquisitions, *Remove from the cache*. With program 1.3.3 the pane does the same.

```bash
pip install "ppg-sentinel2 @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/sentinel2"
ppg plugin sentinel2 search scavo.ppg --start 2021-01-01 --end 2024-12-31 --max-cloud 30 --margin 300
ppg plugin sentinel2 fetch scavo.ppg --min-clear 0.5          # read, cache, indices, series
ppg plugin sentinel2 series scavo.ppg --monthly --index NDVI
ppg plugin sentinel2 cropmarks scavo.ppg --index NDVI --period 2022-summer --background 50
ppg plugin sentinel2 composite scavo.ppg --index NDMI --period 2022-06
ppg plugin sentinel2 list scavo.ppg
ppg plugin sentinel2 export scavo.ppg cartella_export/
ppg plugin sentinel2 remove-scene scavo.ppg 20220614_S2B      # out of the cache (read again by fetch)
ppg plugin sentinel2 remove-scene scavo.ppg --all --with-products
# a box of longitudes and latitudes instead of the footprint of the DEM (here a public site)
ppg plugin sentinel2 search scavo.ppg --bbox -1.8334,51.1744,-1.8191,51.1834 --margin 0
```

`fetch --scenes ID,ID` reads only some acquisitions (their ids are in `search` and `list`),
`--clear 4,5,6,7` keeps the unclassified pixels too, `--buffer` sets the metres masked around
clouds and shadows; `compute` recomputes indices and series from the cache, with no network
(`--min-clear` changes the least clear part of the area and saves it, as *Compute again* does with
the value of the pane). `remove-scene ID [ID...]` or `--all` takes acquisitions out of the cache:
the products made from them stay, marked, unless `--with-products`. `--bbox` takes
`west,south,east,north` also with a negative first number, as `--bbox=...` or as four numbers;
`--polygon` a polygon in the chunk's coordinates (`"x,y;x,y;x,y"`). `--catalog URL` searches
another STAC API (`--client stdlib|pystac` chooses the client).

The numbers of the COGs become reflectance with the scale and offset of the catalog: from the
processing baseline 04.00 (January 2022) ESA adds 1000 to the L2A numbers, which the offset of
-0.1 takes away; Earth Search marks the items whose numbers it has already harmonised
(`earthsearch:boa_offset_applied`), and the plugin trusts the numbers over the metadata: an offset
that would make the blue reflectance of the land negative is dropped.

Everything lives in `<chunk data>/sentinel2/` (the search, the cache of the windows read, the
stacks, the series, the products) and in `chunk.meta["sentinel2"]`.

**Data**: Contains modified Copernicus Sentinel data [years of the acquisitions used]. The
notice is written into every GeoTIFF (`ATTRIBUTION`, `TIFFTAG_COPYRIGHT`), at the head of the
CSVs, at the foot of the chart and in its metadata, and in the pane. Sentinel-2 data are free and
open under the Legal Notice on the use of Copernicus Sentinel Data and Service Information; the
Cloud-Optimized GeoTIFFs are those of the
[Sentinel-2 dataset on the Registry of Open Data on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

Licence: GPL-2.0-or-later. Author: Enzo Cocca.
