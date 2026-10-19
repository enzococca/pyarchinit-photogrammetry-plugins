# Archivio GIS - GIS archive

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later): the survey of a project into a GIS database outside PyArchInit - a **GeoPackage**,
a **PostgreSQL/PostGIS** database or a **SpatiaLite** file.

| Table | What | Geometry |
|---|---|---|
| `ppg_quote` | spot heights drawn: height, area, US, unit, precision, survey, date, drawer | point |
| `ppg_curve` | contour lines of the DEM at the interval chosen (suggested from the DEM by default), index lines | line |
| `ppg_us` | stratigraphic units drawn: site, area, US, type, the fields of `pyunitastratigrafiche`, area in m² | multipolygon |
| `ppg_sezioni` | traces of the sections: name, length, drawing | line |
| `ppg_pietre` | stones drawn with SAM: group, area | polygon |
| `ppg_marker` | markers and control points: height, reference, error, control or check | point |
| `ppg_raster` | DEM and orthomosaic: footprint, file, resolution, size, CRS, table of the copy | polygon |

Every row carries `progetto`, `chunk`, `sito` and `archiviato`: archiving the same project and
chunk again replaces its rows and keeps those of the other projects, so one database can collect
a whole excavation. The tables are in the CRS of the chunk.

Rasters are copied into the database when it can hold them - tiles of the GeoPackage (elevations
as a gridded coverage of floats; table names in capitals, as GDAL writes them through rasterio),
tiles of 256 pixels in a PostGIS raster table with its spatial index and the constraints QGIS reads
(`CREATE EXTENSION postgis_raster` once; the tiles are encoded by the plugin, so the GDAL of the
server can stay disabled) - or recorded as a reference only (file and footprint; always so in
SpatiaLite). A raster copy is a table of its own (`ppg_dem_<project>_<chunk>`), made again at every
archive; a table of that name holding something else is never replaced. The copies can be reduced
to a largest side in pixels.

## Use

* *Plugin > Archivio GIS > Salva in un database GIS...*: the database (a file, or server, port,
  database, user, password and schema of PostgreSQL) and what to save. Everything but the password
  is remembered for the next time.
* *File > Esporta > Archivio GIS (GeoPackage)...*: everything into a GeoPackage at once.
* Command line:

  ```bash
  ppg plugin gis-archive scavo.ppg archivio.gpkg
  ppg plugin gis-archive scavo.ppg spatialite:archivio.sqlite --what quote,us,sezioni
  PGPASSWORD=... ppg plugin gis-archive scavo.ppg "postgresql://enzo@localhost:5432/scavi?schema=tell" \
      --interval 0.5 --rasters copy --max-side 8192
  ```

## Italiano

Il rilievo del progetto in un database GIS diverso da PyArchInit (GeoPackage, PostgreSQL/PostGIS,
SpatiaLite): quote, curve di livello dal DEM, US disegnate, sezioni, pietre, marker, DEM e
ortofoto, ognuno nella sua tabella con progetto, chunk e sito. Dal menu *Plugin > Archivio GIS*,
da *File > Esporta* o da terminale (`ppg plugin gis-archive`). La password di PostgreSQL non viene
mai salvata.

Licence: GPL-2.0-or-later. Author: Enzo Cocca.
