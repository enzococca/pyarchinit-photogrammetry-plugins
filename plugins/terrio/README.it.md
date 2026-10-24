# terrio

*[English](README.md)* - **[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/terrio/)**:
i file e le sorgenti delle nuvole di punti, la classificazione del terreno, la pulizia, i colori e i valori dai
raster, le quote ellissoidiche e sul geoide, i raster di riferimento (anche dal Planetary Computer di Microsoft),
le esportazioni.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.1 o
successiva) costruito su [terrio](https://github.com/opengeos/terrio), lettura e scrittura di dati geografici
in Rust puro: niente GDAL, PROJ o PDAL da installare, una wheel per piattaforma (macOS, Windows x64 e ARM64,
Linux x86_64 e aarch64).

| Dove | Che cosa |
|---|---|
| *Plugin > terrio > Importa nuvola di punti...* | File LAS, LAZ, COPC, EPT, PLY ed E57 (anche in uno zip), su disco o sul web (https, S3, Google Cloud, Azure; COPC ed EPT remoti letti sul posto), come nuvola densa di un nuovo chunk (che la finestra rende attivo) o di quello attivo (la nuvola di ora resta fra le versioni). Colori, intensità (in grigio) e classi vengono con i punti; un limite di punti, una scelta di classi e un sistema di riferimento per i file che non lo hanno. |
| *Plugin > terrio > Classifica il terreno (SMRF, CSF, PMF)...* | Terreno (classe 2) e altri punti (classe 1) della nuvola densa con i filtri di terrio alla PDAL: Simple Morphological Filter, Cloth Simulation Filter o Progressive Morphological Filter, ciascuno con i suoi valori predefiniti; prima il rumore basso (classe 7) con ELM; gli altri punti, a scelta, nelle classi di vegetazione ASPRS 3, 4 e 5 per altezza dal suolo. Cambiano solo i punti delle classi 0, 1 e 2 (e 3, 4, 5 con la vegetazione): edifici, muri e altre classi archeologiche (64 e oltre), rumore, classi di un lidar importato e punti classificati a mano restano come sono. |
| *Plugin > terrio > Pulisci la nuvola (LOF, DBSCAN)...* | I punti isolati per il loro fattore di anomalia locale (LOF sopra 3 con 20 vicini) vanno in classe 7 (rumore); i gruppi staccati - gruppi di DBSCAN (raggio 4 volte la distanza fra i punti) con meno dell'1 % della nuvola e a più di 4 raggi da ogni altro punto: uccelli, persone, frammenti sospesi - vengono rimossi o messi in una classe (18 di base). I punti vengono contati e mostrati nella vista 3D (rossi, arancioni) prima di cambiare qualcosa; i punti classificati a mano non cambiano mai, quelli già nelle classi 7 e 18 restano fuori; la nuvola di prima resta fra le versioni. |
| *Plugin > terrio > Colori e valori da un raster...* | I colori dell'ortofoto del chunk (o di una versione, di un file, di un COG sul web, in qualsiasi sistema di riferimento) per i punti che copre - un lidar senza colori, la nuvola di un altro programma - con la nuvola di prima fra le versioni; oppure il valore di una banda - la temperatura dell'ortofoto termica, un indice (NDVI), il DEM, una banda della riflettanza - come dimensione dei punti (`temperature`, `ndvi`, `height`: una proprietà `float` della PLY della nuvola densa, NaN dove il raster non ha valore), esportata con LAS, LAZ, COPC, EPT, PLY ed E57 del plugin. Il raster è letto a tasselli con `colorization` di terrio. |
| *Plugin > terrio > Valori dei punti nella vista 3D...* | uno di questi valori a colori (turbo, viridis, inferno, grey...) nella vista 3D (programma 1.3.4 o successivo), senza toccare i colori della nuvola; oppure la dimensione tolta |
| *Plugin > terrio > Quote ellissoidiche e sul geoide...* | Il DEM attivo (o una versione) o la nuvola densa spostati fra quote ellissoidiche (GNSS) e quote sul geoide - EGM2008 (`EPSG:3855`), EGM96 (`EPSG:5773`), un CRS verticale nazionale, o un file del geoide (`.gtx`, `.tif`) - come nuova versione. Le griglie che mancano vengono dal CDN di PROJ (scaricate in `<dati del plugin>/grids`, o lette sul posto); una quota che nessuna griglia ha spostato è un errore, mai una copia silenziosa. |
| *Plugin > terrio > Raster di riferimento dal web...* | Un Cloud Optimized GeoTIFF da URL (o una ricerca STAC, `STACIT:...`, o un file), oppure una collezione del **Planetary Computer di Microsoft** (Copernicus DEM GLO-30 e GLO-90, ALOS World 3D, NASADEM, USGS 3DEP, Sentinel-2 L2A, NAIP, o qualsiasi altra collezione e asset: cercata sopra il rilievo, con gli indirizzi firmati da terrio), letto solo sopra il rilievo più un margine, riproiettato nel sistema del chunk e tenuto come versione del DEM o dell'ortofoto (*Versioni*), con il nome `Riferimento: <file>` se non ne dai uno: quello attivo non cambia. Le quote di un DEM possono essere portate nel sistema del rilievo (il catalogo sa che Copernicus dà quote EGM2008). |
| *File > Esporta* | la nuvola densa in **E57** o come dataset **EPT** (una cartella), l'ortofoto in **PMTiles** o **MBTiles** (tasselli Web Mercator per le mappe web) |
| *Plugin > terrio > Installa terrio...* | l'applicazione pacchettizzata non ha pip: il plugin scarica la wheel di terrio per questo computer da PyPI (con la sua SHA-256 controllata) nella propria cartella; aggiorna anche un terrio più vecchio |

Dalla riga di comando:

```bash
ppg plugin terrio version                                   # anche la cartella delle griglie
ppg plugin terrio install                                   # nella cartella del plugin, senza pip
ppg plugin terrio import scavo.ppg scansione.e57 --label "laser"
ppg plugin terrio import scavo.ppg https://host/lidar.copc.laz --chunk rilievo --max-points 5000000
ppg plugin terrio ground scavo.ppg --method csf --resolution 0.5 --noise --vegetation
ppg plugin terrio clean scavo.ppg --dry-run                 # solo i conteggi
ppg plugin terrio clean scavo.ppg --remove                  # isolati -> 7, gruppi staccati rimossi
ppg plugin terrio colour scavo.ppg --list                   # i raster del chunk
ppg plugin terrio colour scavo.ppg orthomosaic              # i colori
ppg plugin terrio colour scavo.ppg thermal                  # la temperatura come dimensione
ppg plugin terrio values scavo.ppg
ppg plugin terrio heights scavo.ppg --from ellipsoid --to EPSG:3855
ppg plugin terrio heights scavo.ppg --product point_cloud --from ellipsoid --to geoide_nazionale.gtx --activate
ppg plugin terrio reference scavo.ppg https://host/dem_regionale.tif --product dem --margin 50 --resolution 1
ppg plugin terrio reference scavo.ppg --collection cop-dem-glo-30 --margin 100 --heights-to ellipsoid
ppg plugin terrio catalogue                                 # le collezioni del Planetary Computer per nome
ppg plugin terrio export scavo.ppg nuvola.e57
ppg plugin terrio export scavo.ppg nuvola.laz               # con le dimensioni dei valori
ppg plugin terrio export scavo.ppg ortofoto.pmtiles
```

## Installazione

* Nel programma: *Strumenti > Plugin...*, `terrio`, *Installa / aggiorna*, *Riavvia ora*; poi *Plugin >
  terrio > Installa terrio* una volta (circa 20 MB da PyPI).
* Con pip, nell'ambiente Python del programma (terrio arriva con lui):

  ```bash
  pip install "ppg-terrio @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/terrio"
  ```

Il plugin è scritto per terrio 0.5.0 e pip installa `terrio>=0.5,<0.6`. Un terrio più vecchio (0.3.2 e le
versioni 0.4, come quello che può trovarsi nella cartella del plugin dell'applicazione) fa ancora
l'importazione, la classificazione del terreno, i raster di riferimento e le esportazioni; la pulizia, i colori
e i valori, le quote e il Planetary Computer richiedono la 0.5 e lo dicono - la finestra propone di scaricarla.
terrio è giovane e le sue funzioni possono ancora cambiare prima della 1.0.

## Note

* Coordinate: terrio lavora nelle coordinate del file o del chunk; il plugin porta i punti nel sistema interno
  del chunk (`chunk.transform`) e indietro.
* Pulizia: sulle nuvole fotogrammetriche di densità molto diversa il LOF segna anche i bordi fra parti dense e
  rade (la soglia di base, 3, ne lascia fuori quasi tutti), e le chiome degli alberi senza tronco sono anch'esse
  "gruppi staccati": guarda i punti nella vista 3D prima di applicare, e preferisci una classe alla rimozione.
* Il programma non conosce i valori dei punti: quando riscrive la nuvola (la sua pulizia, le classi, il
  ritaglio) spariscono e vanno presi di nuovo dal raster.
* Planetary Computer: usato senza account, con i suoi limiti di accesso; una chiave in
  `PC_SDK_SUBSCRIPTION_KEY` viene mandata se c'è. I file a colori naturali di Sentinel-2 sono grandi: terrio
  scarica circa 200 MB per immagine per una finestra, così il catalogo ne legge una (la più recente sotto il
  10 % di nuvole).
* I filtri lavorano sulla nuvola intera in memoria: decine di milioni di punti richiedono qualche gigabyte.
* Test: `pip install -e . pytest` nell'ambiente del programma, poi `pytest tests` (`PPG_TEST_NETWORK=1`
  aggiunge il test del Planetary Computer e del CDN di PROJ).

Licenza GPL-2.0-or-later. Autore: Enzo Cocca.
