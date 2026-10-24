# Sentinel-2

*[English](README.md)*

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 o successiva): serie temporali Sentinel-2 sull'area di un rilievo, per cercare i **crop
mark**. Muri e fossati sepolti si vedono nella coltura che li copre in alcune settimane di alcuni
anni: sopra un muro le radici trovano meno acqua e la coltura è più debole (un indice più basso,
un segno *negativo*), sopra un fossato ne trovano di più e la coltura è più verde (un segno
*positivo*). Con 10 m per pixel Sentinel-2 vede i più grandi (recinti, strade, ville, fossati larghi
qualche metro) e soprattutto *quando* i campi li mostrano: due satelliti passano ogni cinque giorni,
anno dopo anno.

* **L'area**: l'impronta del DEM o dell'ortofoto del chunk, oppure un poligono disegnato sulla
  vista con lo strumento del programma, con un margine intorno (300 m di default: lo sfondo di una
  mappa dei crop mark ha bisogno dei campi intorno al sito). Un chunk in un sistema locale (LOCAL)
  non ha un posto sulla Terra e non si può cercare: il plugin lo dice (dalla riga di comando si può
  comunque dare un riquadro in longitudine e latitudine).
* **La ricerca** in un catalogo STAC: [Earth Search](https://earth-search.aws.element84.com/v1) di
  Element 84 di default (la collezione `sentinel-2-l2a`: Cloud-Optimized GeoTIFF pubblici su AWS,
  senza account né chiavi), qualsiasi API STAC dal suo URL. Con il catalogo parla la libreria
  standard, quindi l'applicazione impacchettata non ha bisogno di nulla;
  [pystac-client](https://github.com/stac-utils/pystac-client) si usa quando è installato. I tile di
  un passaggio di un satellite diventano un'*acquisizione*; la copertura nuvolosa dei tile le filtra.
* **La lettura**: solo la finestra dell'area e del suo margine, dai COG via HTTPS (`/vsicurl/` di
  GDAL, con nuovi tentativi), bande B02, B03, B04, B05, B08, B8A, B11 e la classificazione della
  scena (SCL), su una sola griglia da 10 m nella zona UTM dell'area (le bande da 20 m interpolate, le
  classi prese dal pixel più vicino, un tile di un'altra zona riproiettato, due tile dello stesso
  passaggio uniti). La classificazione della scena si legge per prima: nuvole, cirri e le loro ombre
  (con 20 m intorno), neve e ogni classe non tenuta come serena sono mascherate; un'acquisizione con
  meno di metà dell'area serena (di default) resta fuori prima che si scarichino le altre bande. Ciò
  che è stato letto resta in cache nel progetto: **una seconda occhiata non usa la rete**. Con la
  finestra di un sito di qualche centinaio di metri un'acquisizione sono circa 16 MB da scaricare
  (i COG si leggono per tasselli interni da 1024 pixel) e circa 10 secondi; la cache ne tiene circa
  100 KB.
* **La cache**: le acquisizioni lette si possono togliere - quelle selezionate nella tabella del
  pannello (*Rimuovi dalla cache*, anche dal menu del tasto destro) o tutte (*Svuota la cache...*),
  dopo una domanda. I loro file vengono cancellati, restano nella ricerca come *da leggere* e il
  prossimo *Scarica e calcola* le legge di nuovo; indici e serie vengono ricalcolati con le altre.
  Se da quelle acquisizioni sono stati fatti compositi o mappe dei crop mark, la domanda chiede se
  rimuovere anche quelli o tenerli: tenuti, sono segnati *date non più in cache* (sono ciò che è
  stato calcolato allora, ma la cache non può più rifarli), e il segno se ne va quando le
  acquisizioni vengono lette di nuovo.
* **Gli indici** di ogni data serena: NDVI = (B08 - B04) / (B08 + B04), vigore della vegetazione
  (10 m); NDRE = (B8A - B05) / (B8A + B05), clorofilla (bordo del rosso, 20 m); NDMI = (B8A - B11) /
  (B8A + B11), acqua nelle foglie (20 m). Solo le celle serene; un GeoTIFF per indice con una banda
  per data.
* **La serie** sull'area: per data e per mese (mediana, 10°, 25°, 75° e 90° percentile, parte
  serena dell'area) in CSV, e il suo grafico (PNG): quando i campi rinverdiscono, quando seccano,
  quando un segno può comparire.
* **Compositi**: la mediana di un indice in un periodo, cella per cella: una stagione di un anno
  (`2023-summer`; l'inverno di un anno è il dicembre prima, gennaio e febbraio), una stagione di
  ogni anno (`summer`), un anno, un mese (`2023-06`), un intervallo di giorni
  (`2023-04-01/2023-06-30`), una data.
* **Mappe dei crop mark**: l'indice di una data, o la mediana di un periodo, meno il suo sfondo
  liscio (una gaussiana, sigma 50 m di default, o una mediana su una finestra doppia), diviso per la
  dispersione robusta di ciò che resta (1,4826 volte la deviazione assoluta mediana): uno z-score,
  vicino a 0 sui campi uniformi, sotto -2 dove la vegetazione è più debole di quella intorno (muri,
  strade, terreno compattato) e sopra 2 dove è più verde (fossati, fosse). I segni sono linee larghe
  pochi pixel. Si leggono come un indice, nel pannello come in QGIS: rosso-giallo-verde da -3 (rosso,
  vegetazione più debole) a 3 (verde, più vigorosa); la legenda sotto la vista nomina i due estremi
  della palette scelta.
* **Sulla vista**: un indice di una data o di un periodo, i colori naturali (B04, B03, B02) di una
  data, un composito o una mappa dei crop mark sopra l'ortofoto o il DEM del chunk, con le palette
  degli indici di vegetazione del programma (rosso-giallo-verde, grigi, viridis, turbo) o una
  divergente, e un'opacità; ogni pixel di Sentinel-2 resta un quadrato. *Mostra l'area intera* mette
  l'area e il suo margine su una cornice vuota, per vedere oltre l'ortofoto. L'area e la finestra
  letta sono disegnate nel layer *Sentinel-2: area*; entrambe seguono la vista quando mostra un altro
  raster.
* **Prodotti ed esportazioni**: ogni composito e ogni mappa dei crop mark è un GeoTIFF sulla
  griglia delle scene (UTM) e una copia nel sistema di riferimento del chunk, su una griglia
  allineata al suo DEM o alla sua ortofoto (celle di circa 2,5 m, trasparente dove manca il valore); le
  copie vanno nel progetto QGIS del programma. *Esporta tutto* scrive lo stack di ogni indice e data
  (un GeoTIFF), la serie (CSV per data e per mese), il grafico e i prodotti; *File > Esporta* ha la
  serie (CSV) e lo stack (GeoTIFF).
* Il pannello **Sentinel-2** (con il programma 1.3.4 un pannello a sé accanto all'area di lavoro, da *Vista > Pannelli*; prima una scheda del dock del progetto) fa tutto, in background con la barra di
  avanzamento (annulla ferma le letture in corso); *Plugin > Sentinel-2* ha le azioni principali.
* **Nell'albero del progetto** (programma 1.3.4 o successivo): un nodo *Sentinel-2* sotto ogni chunk
  su cui il plugin ha lavorato, con le acquisizioni lette (data e parte serena dell'area), gli indici
  per data, la serie, i compositi e le mappe dei crop mark. Il doppio clic ne mostra uno sulla vista
  (colori naturali per un'acquisizione, l'indice di una data, un prodotto; il grafico per la serie)
  e il pannello lo segue; il menu del tasto destro ha *Mostra*, *Nascondi*, *Esporta...*, *Apri in
  QGIS*, *Rimuovi* e, per le acquisizioni, *Rimuovi dalla cache*. Con il programma 1.3.3 fa tutto il
  pannello.

```bash
pip install "ppg-sentinel2 @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/sentinel2"
ppg plugin sentinel2 search scavo.ppg --start 2021-01-01 --end 2024-12-31 --max-cloud 30 --margin 300
ppg plugin sentinel2 fetch scavo.ppg --min-clear 0.5          # lettura, cache, indici, serie
ppg plugin sentinel2 series scavo.ppg --monthly --index NDVI
ppg plugin sentinel2 cropmarks scavo.ppg --index NDVI --period 2022-summer --background 50
ppg plugin sentinel2 composite scavo.ppg --index NDMI --period 2022-06
ppg plugin sentinel2 list scavo.ppg
ppg plugin sentinel2 export scavo.ppg cartella_export/
ppg plugin sentinel2 remove-scene scavo.ppg 20220614_S2B      # fuori dalla cache (fetch la rilegge)
ppg plugin sentinel2 remove-scene scavo.ppg --all --with-products
# un riquadro in longitudine e latitudine invece dell'impronta del DEM (qui un sito pubblico)
ppg plugin sentinel2 search scavo.ppg --bbox -1.8334,51.1744,-1.8191,51.1834 --margin 0
```

`fetch --scenes ID,ID` legge solo alcune acquisizioni (i loro id sono in `search` e `list`),
`--clear 4,5,6,7` tiene anche i pixel non classificati, `--buffer` fissa i metri mascherati intorno
a nuvole e ombre; `compute` ricalcola indici e serie dalla cache, senza rete (`--min-clear` cambia
la parte serena minima dell'area e la salva, come fa *Ricalcola* con il valore del pannello).
`remove-scene ID [ID...]` o `--all` toglie acquisizioni dalla cache: i prodotti fatti da esse
restano, segnati, a meno di `--with-products`. `--bbox` accetta `ovest,sud,est,nord` anche con il
primo numero negativo, come `--bbox=...` o come quattro numeri; `--polygon` un poligono nelle
coordinate del chunk (`"x,y;x,y;x,y"`). `--catalog URL` cerca in un'altra API STAC
(`--client stdlib|pystac` sceglie il client).

I numeri dei COG diventano riflettanza con la scala e l'offset del catalogo: dalla processing
baseline 04.00 (gennaio 2022) l'ESA aggiunge 1000 ai numeri L2A, che l'offset di -0,1 toglie; Earth
Search segna gli item i cui numeri ha già armonizzato (`earthsearch:boa_offset_applied`), e il
plugin crede ai numeri più che ai metadati: un offset che renderebbe negativa la riflettanza nel blu
del terreno viene lasciato cadere.

Tutto sta in `<dati del chunk>/sentinel2/` (la ricerca, la cache delle finestre lette, gli stack,
la serie, i prodotti) e in `chunk.meta["sentinel2"]`.

**Dati**: Contains modified Copernicus Sentinel data [anni delle acquisizioni usate] (contiene dati
Copernicus Sentinel modificati). La dicitura è scritta in ogni GeoTIFF (`ATTRIBUTION`,
`TIFFTAG_COPYRIGHT`), in testa ai CSV, ai piedi del grafico e nei suoi metadati, e nel pannello. I
dati Sentinel-2 sono liberi e aperti secondo la nota legale sull'uso dei dati e delle informazioni
dei servizi Copernicus Sentinel; i Cloud-Optimized GeoTIFF sono quelli del
[dataset Sentinel-2 del Registry of Open Data on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

Licenza: GPL-2.0-or-later. Autore: Enzo Cocca.
