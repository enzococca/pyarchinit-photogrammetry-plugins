# Plugin di pyarchinit-photogrammetry

*[English](README.md)*

I plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(dalla 1.1) e il loro indice, `index.json`, che il programma legge in *Strumenti > Plugin* e con
`ppg plugins available`.

| Plugin | Cosa fa |
|---|---|
| [`geoarchaeology`](plugins/geoarchaeology) - Geoarcheologia | Sondaggi, carote e sezioni di scavo (CSV di collari e intervalli) sul rilievo: colonne stratigrafiche, superfici di contatto e spessori per kriging (variogramma adattato ai minimi quadrati pesati, kriging ordinario o universale, la varianza di kriging; IDW con pochi sondaggi), sotto il DEM e in ordine; sezioni disegnate sulla vista o per i sondaggi, con i log proiettati; modello 3D delle unità con i volumi, ritagliato sulle US disegnate; GeoPackage, GeoTIFF, DXF, PLY/OBJ, CSV, nel progetto QGIS |
| [`geophysics`](plugins/geophysics) - Geofisica | Griglie geofisiche - magnetometria, resistività, time slice GPR, EM - importate da Surfer GRD, XYZ/CSV, GeoTIFF o cartelle di slice, georeferenziate con i marker o le coordinate dei picchetti, filtrate (despike, destripe, passa-alto, clip, riempimento), sovrapposte a ortofoto e DEM con opacità, esportate in GeoTIFF, Surfer GRD e GeoPackage, nel progetto QGIS |
| [`terrain`](plugins/terrain) - Terreno | Analisi del terreno sul DEM con WhiteboxTools (scaricato una volta): reticolo e bacini, posizione topografica, geomorphons, curvature, rugosità e ore di luce, depressioni e rilievi annidati con profondità e volume, visibilità; sovrapposti a ortofoto e DEM, esportati in GeoTIFF e GeoPackage, nel progetto QGIS |
| [`gis-archive`](plugins/gis-archive) - Archivio GIS | Il rilievo del progetto - quote, curve di livello, US disegnate, sezioni, pietre, marker, DEM e ortofoto - in un database GIS diverso da PyArchInit: GeoPackage, PostgreSQL/PostGIS o SpatiaLite |
| [`powerlines`](plugins/powerlines) - Linee elettriche | Linee elettriche aeree nella nuvola densa: ogni conduttore come catenaria con lunghezza, freccia, franco dal suolo e distanza dall'ostacolo più vicino, i pali, i punti di cavi e pali classificati (14, 15); esportazioni GeoPackage, DXF e CSV |
| [`spirula`](plugins/spirula) - Spirula | Gaussian splatting 3D delle foto allineate con [Spirula Studio](https://github.com/harry7557558/spirula-studio) (Vulkan: GPU NVIDIA, AMD, Intel e Apple): lo splat come prodotto del chunk, una mesh da esso, il visualizzatore |
| [`terrio`](plugins/terrio) - terrio | Nuvole di punti di qualsiasi formato (LAS, LAZ, COPC, EPT, PLY, E57, anche dal web) nel progetto, classificazione del terreno con i filtri SMRF, CSF e PMF, raster di riferimento dal web come versioni del DEM o dell'ortofoto, esportazioni E57, EPT, PMTiles e MBTiles, con [terrio](https://github.com/opengeos/terrio) (Rust puro, senza GDAL o PDAL) |
| [`blender-em`](plugins/blender-em) - Blender ed EMtools | I proxy 3D delle unità stratigrafiche e il rilievo in un progetto Extended Matrix (`EM/proxies/<unità>.glb`, `RB/`), con i nomi dei nodi dell'em.json di PyArchInit o EMStudio, che EMtools abbina per nome; Blender aperto sul progetto e aggiornato dal vivo quando i proxy cambiano (ciò che è stato cambiato in Blender non è mai sovrascritto); le camere allineate in Blender con le foto raddrizzate; ciò che si modella in Blender di nuovo nel chunk come proxy delle unità, con il loro volume; la selezione condivisa dal vivo con [EMtools](https://github.com/zalmoxes-laran/EM-blender-tools) 1.6 tramite il suo WebSocket di sincronizzazione, e il suo comando che crea il proxy di un'unità |
| [`samgeo`](plugins/samgeo) - Samgeo | Quello che scrivi trovato nell'ortofoto o in un altro raster del chunk - pietre, muri, mattoni, tegole e coppi, frammenti ceramici, alberi, persone - con Grounding DINO e SAM 2.1 (la ricetta di LangSAM di segment-geospatial) sul tuo computer: un layer di proposte da controllare nella vista, poi US, classi della nuvola densa o un GeoPackage; la maschera nel calcolatore raster |

## Installare un plugin

* Dal programma: *Strumenti > Plugin...*, il plugin nell'elenco in basso, *Installa / aggiorna*,
  poi *Riavvia ora*. Funziona anche nell'applicazione impacchettata, che non ha pip.
* Da terminale: `ppg plugins install gis-archive`, poi `ppg plugins list`.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-gis-archive @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/gis-archive"
  ```

I plugin girano con gli stessi permessi del programma: l'indice elenca solo plugin controllati, e
il programma installa una wheel solo se il suo SHA-256 è quello scritto nell'indice.

## Scrivere un plugin

La guida è [docs/plugins.md](https://github.com/enzococca/pyarchinit-photogrammetry/blob/main/docs/plugins.md)
nel repository del programma; [`template/`](template) è un plugin da cui partire (azioni di menu,
lavoro in background con la barra di avanzamento, un comando, un formato di esportazione,
impostazioni, test). `ppg plugins new NOME` (o *Strumenti > Plugin... > Nuovo plugin...* nel programma)
scrive lo stesso progetto col tuo nome, le parti che scegli e i suoi test, pronto per `pip install -e .`.

## Proporre un plugin

Un plugin può stare nel suo repository o qui, in `plugins/`.

* **Nel tuo repository.** Costruisci una wheel solo Python (`python -m build --wheel`), allegala a
  una release e apri una pull request che aggiunge la sua voce a `index.json`. La voce si ricava
  dalla wheel:

  ```bash
  python scripts/make_entry.py https://github.com/tu/ppg-cosa/releases/download/v0.1.0/ppg_cosa-0.1.0-py3-none-any.whl --write
  ```

* **Qui.** Una pull request che aggiunge `plugins/<nome>/` (pacchetto, `pyproject.toml`,
  `README.md`, test, `entry.json` con titolo, descrizione e testi italiani). Dopo l'unione, le versioni
  si pubblicano come release di GitHub con il tag `<nome>-v<versione>`: il workflow `release` costruisce
  la wheel, la allega alla release e scrive da solo la voce in `index.json` su `main` (testi da
  `entry.json`): il programma mostra la nuova versione al prossimo *Aggiorna l'elenco*.

I campi di una voce sono descritti nello [schema](schema/index.schema.json) e nel
[README inglese](README.md#submitting-a-plugin): `name` è il nome dell'entry point, `api` la
versione dell'API dei plugin, `min_app` la versione minima del programma, `requires` i moduli che
il plugin importa oltre a quelli del programma e non sa procurarsi da solo (terrio, che scarica la sua
wheel da PyPI, dice `[]`), `i18n` titolo e descrizione in altre lingue (per
esempio `{"it": {"title": "Archivio GIS", "description": "..."}}`).

Ogni pull request passa dal workflow `checks`: formato dell'indice, il `__version__` di ogni plugin di
questo repository confrontato con la versione del suo `pyproject.toml` (devono coincidere: il programma
mostra la versione dichiarata dal plugin accanto a quella dell'indice), poi ogni wheel viene scaricata
(hash, versione, entry point, nessun percorso fuori dalla sua cartella) e il plugin viene avviato
con il programma; i plugin di questo repository eseguono i loro test, PostGIS compreso. In locale:

```bash
pip install jsonschema
python scripts/check_index.py --load        # con pyarchinit-photogrammetry installato
```

I plugin devono avere una licenza compatibile con la GPL (girano dentro un programma GPL); quelli di
questo repository sono GPL-2.0-or-later.

Licenza: GPL-2.0-or-later. Autore: Enzo Cocca.
