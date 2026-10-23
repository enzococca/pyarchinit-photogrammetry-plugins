# Geofisica

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/geophysics/)**: i file da esportare, come posare una griglia sui picchetti, i filtri, come leggere i risultati.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 o successiva): le griglie di una prospezione geofisica - magnetometria, resistività, time
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
* **Sovrapposizione** all'ortofoto o al DEM della vista raster (oppure, se la vista è vuota, su un
  riquadro della misura della griglia), con la banda (la slice di una prospezione GPR), la palette
  (grigi, grigi invertiti, divergente, terra, calore) e l'opacità.
* **Anomalie**: *Disegna anomalia* disegna un poligono sulla vista (clic sui vertici, doppio clic o
  Invio per chiudere) con la griglia, la banda e una nota, nel layer *Anomalie geofisiche*;
  conservate nel progetto, si cancellano con lo strumento *Seleziona* del programma.
* **Esportazioni**: GeoTIFF (tutte le bande) e Surfer GRD (la banda mostrata; una griglia ruotata
  ricampionata con il nord in alto) di una griglia, un GeoPackage con i contorni delle griglie e le
  anomalie (*File > Esporta*), le griglie nel progetto QGIS del programma con la loro scala di
  colori.
* `ppg plugin geophysics import|list|georef|filter|export PROJECT.ppg ...` dalla riga di comando.

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import scavo.ppg prospezione.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef scavo.ppg "Area 1" --origin 500100.0,4500800.0 --azimuth 95
ppg plugin geophysics filter scavo.ppg "Area 1" destripe --copy
ppg plugin geophysics export scavo.ppg "Area 1 destripe" area1.tif
```

Licenza: GPL-2.0-or-later.
