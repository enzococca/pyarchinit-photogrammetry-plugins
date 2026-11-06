# Geofisica

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/geophysics/)**: i file da esportare, come posare una griglia sui picchetti, i filtri, come leggere i risultati.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 o successiva): le griglie di una prospezione geofisica - magnetometria, resistività, time
slice GPR, elettromagnetica - accanto ai prodotti fotogrammetrici dello stesso terreno.

* **Import**: Surfer GRD (ASCII e binario, Surfer 6 e 7), ASCII XYZ / CSV (`x y valore`, le
  esportazioni di Geoplot, Bartington, Terrasurveyor; virgole decimali con punti e virgola,
  tabulazioni o spazi fra le colonne; i valori fittizi 2047.5, -9999, 32767, -32768, il vuoto di
  Surfer e uno tuo come assenza di lettura; letture più fitte lungo le traverse che fra le traverse
  danno una griglia della distanza più fine con le righe fra due traverse interpolate, oppure le
  letture mediate in una cella tua, le celle centrate sulle letture), GeoTIFF e qualsiasi raster,
  una cartella di raster o immagini come slice di una prospezione GPR (nell'ordine naturale dei
  nomi). Ogni griglia diventa un GeoTIFF float32 nel progetto (`<dati del chunk>/geophysics/`).
* **Georeferenziazione** di una griglia rilevata in metri locali: gli angoli della griglia come
  marker posati sull'ortofoto, oppure un'origine e un azimut, oppure coppie digitate di coordinate
  locali e mondo (una similitudine ai minimi quadrati, con il residuo di ogni punto); posata di
  nuovo, una griglia parte dai suoi metri locali: la nuova posa sostituisce la vecchia.
* **Filtri**: despike, destripe (traversa a media zero), clip, passa-basso (sigma 1 cella),
  passa-alto (il residuo di un lisciamento gaussiano di sigma 10 celle: le anomalie piccole sopra la
  geologia), riempimento dei vuoti, ricampionamento - in una copia o sul posto, ogni filtro
  registrato; gli stessi valori predefiniti nella finestra e dalla riga di comando.
* **Sovrapposizione** all'ortofoto o al DEM della vista raster (oppure, se il chunk non ha né l'una
  né l'altro, su un riquadro vuoto della misura della griglia), con la banda (la slice di una
  prospezione GPR), la palette (grigi, grigi invertiti, divergente, terra, calore) e l'opacità; resta
  sopra il raster che la vista mostra dopo, fino a *Nascondi*.
* **Anomalie**: *Disegna anomalia* disegna un poligono sulla vista con il disegno del programma,
  prestato al plugin (trascina un riquadro, oppure clic sui vertici e doppio clic o Invio per
  chiudere), con la griglia, la banda e una nota, nel layer *Anomalie geofisiche* (ridisegnato su ogni
  raster che la vista mostra); conservate nel progetto, si cancellano con lo strumento *Seleziona* del
  programma.
* **Esportazioni**: GeoTIFF (tutte le bande) e Surfer GRD (la banda mostrata; una griglia ruotata
  ricampionata con il nord in alto) di una griglia, un GeoPackage con i contorni delle griglie e le
  anomalie (*File > Esporta*), le griglie nel progetto QGIS del programma con la loro scala di
  colori.
* `ppg plugin geophysics import|list|georef|filter|export PROJECT.ppg ...` dalla riga di comando.

## Tomografia di resistività elettrica (ERT, 0.2.0)

La scheda *ERT* del pannello porta le letture di un georesistivimetro a una sezione di resistività
nel rilievo, invertita con [pyGIMLi](https://www.pygimli.org) (Apache-2.0):

1. **Letture**: una tabella CSV/TXT (`a, b, m, n` e `rhoa`, oppure `r`, oppure `u` e `i`; `err` se
   c'è; anche le colonne `Spa.1-4`, `Rho`, `Dev`, `Vp`, `In` del Syscal; elettrodi numerati da 1 o
   posizioni in metri), il formato dati unificato di pyGIMLi (`.ohm`, `.dat`) e, tramite pyGIMLi,
   gli altri file degli strumenti (RES2DINV, ABEM, Syscal); una tabella degli elettrodi facoltativa
   (`x, z` lungo la linea oppure `x, y, z` del mondo). *Profilo sintetico* produce letture di un
   modello a due strati con un blocco.
2. **Linea nel rilievo**: disegnata sull'ortofoto o sul DEM, da due marker o con coordinate digitate
   (due punti o più: una linea spezzata), oppure dalle coordinate degli elettrodi nel file quando sono
   nel sistema del chunk; la distanza del primo elettrodo e la spaziatura, misurata lungo il terreno
   (il cavo steso sul pendio) o in pianta; le quote degli elettrodi dal DEM del chunk, dal file o piatte.
3. **Pseudosezione e filtri**: le letture come punti nella vista 2D (si selezionano e si cancellano
   quelle cattive); filtri su intervallo, errore, reciproci e picchi.
4. **Inversione**: il gestore ERT di pyGIMLi su una mesh che segue la topografia (fattori
   geometrici calcolati su di essa), lambda, peso verticale, modello a blocchi, dati robusti,
   errori relativo e in tensione; χ² e RMS; le parti che le letture non risolvono oscurate con la
   copertura o con l'indice di profondità di investigazione di Oldenburg e Li.
5. **Sezione**: nella vista 2D nel suo riferimento (distanza lungo la linea per quota, colori
   logaritmici con la legenda in ohm m), nella vista 3D come sezione verticale sotto la linea con il
   terreno tagliato da una parte, i proxy delle US del programma, le unità e i carotaggi del plugin
   geoarcheologia; la linea e gli
   elettrodi sulla pianta.
6. **Esportazioni**: la sezione in GeoTIFF (riferimento u, v, con `.frame.json`), le celle della
   mesh in CSV, i pixel in XYZ con coordinate del mondo, figure PNG/SVG/PDF della sezione e della
   pseudosezione, la linea e gli elettrodi in GeoPackage, i paradati (parametri, adattamento,
   versioni) in JSON, le letture tenute in `.ohm`.

pyGIMLi non gira mai nel processo del programma: *Installa pyGIMLi* crea con uv un ambiente a parte
nella cartella dati del plugin (circa 80 MB, più circa 60 MB per Python se uv non lo ha già; `PPG_GEOPHYSICS_ENV` lo sposta) e ogni lettura o
inversione gira lì, in un processo di lavoro. pyGIMLi 1.6 ha pacchetti per Windows x64, Linux x86_64
e macOS 26 o successivo (Apple Silicon e Intel), non per Windows o Linux su ARM.

```bash
ppg plugin geophysics ert install
ppg plugin geophysics ert import scavo.ppg linea1.csv --name "Linea 1" --spacing 1
ppg plugin geophysics ert place scavo.ppg "Linea 1" --markers E1,E48 --heights dem
ppg plugin geophysics ert filter scavo.ppg "Linea 1" --spikes 8 --max-error 5
ppg plugin geophysics ert invert scavo.ppg "Linea 1" --lam 20 --doi oldenburg
ppg plugin geophysics ert export scavo.ppg "Linea 1" esportazioni/linea1
```

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import scavo.ppg prospezione.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef scavo.ppg "Area 1" --origin 500100.0,4500800.0 --azimuth 95
ppg plugin geophysics filter scavo.ppg "Area 1" destripe --copy
ppg plugin geophysics export scavo.ppg "Area 1 destripe" area1.tif
```

Licenza: GPL-2.0-or-later.
