# Archivio GIS - GIS archive

**[User guide](https://enzococca.github.io/pyarchinit-photogrammetry/plugins/gis-archive/)**: what the project must hold, the databases, the tables and their columns, archiving again.

A plugin of [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 or later): the survey of a project into a GIS database outside PyArchInit - a **GeoPackage**,
a **PostgreSQL/PostGIS** database or a **SpatiaLite** file.

| Table | What | Geometry |
|---|---|---|
| `ppg_quote` | spot heights drawn: height, area, US, unit, precision, survey, date, drawer | point |
| `ppg_curve` | contour lines of the DEM at the interval chosen (suggested from the DEM by default), index lines | line |
| `ppg_us` | stratigraphic units drawn: site, area, US, type, the fields of `pyunitastratigrafiche`, area in m² | multipolygon |
| `ppg_sezioni` | traces of the sections: name, length, drawing; they belong to the project, not to a chunk: one set of rows per project | line |
| `ppg_pietre` | stones drawn with SAM on the orthomosaic: group, area (stones drawn on the elevation of a wall are in the coordinates of the elevation, not of the map: they are left out and the summary says so) | polygon |
| `ppg_marker` | markers and control points: height, reference, error, control or check | point |
| `ppg_raster` | DEM and orthomosaic: footprint, file, resolution and size of what is archived (of the copy, when it is reduced), CRS, table of the copy | polygon |

Every row carries `progetto` (the file name of the project), `chunk`, `sito` and `archiviato`,
and what says whose it is: `progetto_id`, an id the project keeps (made at its first archive and
saved with it: the command line writes it into the project file), `progetto_file` and `chunk_id`.
Archiving the same project and chunk again replaces its rows and keeps those of every other
project - also of another project with the same file name in another folder; a copy of a project
(*Save as*, a copied folder) is another project. Archiving only some products replaces only their
rows: the orthomosaic alone leaves the row of the DEM. Rows written by version 0.1.0 (without
`progetto_id`) are taken as those of the project of the same file name, unless the rasters they
record lie in the folder of another project file. One database can collect a whole excavation.
The tables are in the CRS of the chunk.

Rasters are copied into the database when it can hold them - tiles of the GeoPackage (elevations
as a gridded coverage of floats; table names in capitals, as GDAL writes them through rasterio),
tiles of 256 pixels in a PostGIS raster table with its spatial index and the constraints QGIS reads
(`CREATE EXTENSION postgis_raster` once; the tiles are encoded by the plugin, so the GDAL of the
server can stay disabled) - or recorded as a reference only (file and footprint; always so in
SpatiaLite). A raster copy is a table of its own (`ppg_dem_<project>_<chunk>`, followed by a part of
the project id when another project with the same file name already has that table), made again
at every archive; a table of that name holding something else is never replaced. The copies can be
reduced to a largest side in pixels. PostgreSQL is reached through the driver the program uses:
psycopg2, or psycopg 3 where psycopg2 is missing (the Windows ARM64 application).

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

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/gis-archive/)**: che cosa deve contenere il progetto, i database, le tabelle e le loro colonne, archiviare di nuovo.

Il rilievo del progetto in un database GIS diverso da PyArchInit (GeoPackage, PostgreSQL/PostGIS,
SpatiaLite): quote, curve di livello dal DEM, US disegnate, sezioni, pietre, marker, DEM e
ortofoto, ognuno nella sua tabella con progetto, chunk e sito. Dal menu *Plugin > Archivio GIS*,
da *File > Esporta* o da terminale (`ppg plugin gis-archive`). La password di PostgreSQL non viene
mai salvata.

Ogni riga ha anche `progetto_id` (un id che il progetto conserva, creato al primo archivio),
`progetto_file` e `chunk_id`: archiviando di nuovo lo stesso progetto e chunk si sostituiscono solo
le sue righe, mai quelle di un altro progetto, nemmeno con lo stesso nome di file; una copia del
progetto (*Salva con nome*) e' un altro progetto. Archiviando solo alcuni prodotti si sostituiscono
solo le loro righe (la sola ortofoto lascia la riga del DEM). Le sezioni sono del progetto e si
scrivono una volta sola; le pietre disegnate sul prospetto di un muro (coordinate del prospetto, non
della mappa) restano fuori e il riepilogo lo dice; `risoluzione` e' quella della copia archiviata.
Le righe scritte dalla versione 0.1.0 vengono riconosciute come quelle del progetto con lo stesso
nome di file, a meno che i loro raster stiano nella cartella di un altro progetto. PostgreSQL con
psycopg2 o, dove manca (l'applicazione Windows ARM64), con psycopg 3.

Licence: GPL-2.0-or-later. Author: Enzo Cocca.
