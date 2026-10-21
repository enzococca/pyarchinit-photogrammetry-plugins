# Geofisica

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 o successiva): le griglie di una prospezione geofisica - magnetometria, resistività, time
slice GPR, elettromagnetica - accanto ai prodotti fotogrammetrici dello stesso terreno.

* **Import**: Surfer GRD (ASCII e binario, Surfer 6 e 7), ASCII XYZ / CSV (`x y valore`, le
  esportazioni di Geoplot, Bartington, Terrasurveyor), GeoTIFF e qualsiasi raster, una cartella di
  raster o immagini come slice di una prospezione GPR. Ogni griglia diventa un GeoTIFF float32 nel
  progetto (`<dati del chunk>/geophysics/`).
* **Georeferenziazione** di una griglia rilevata in metri locali: gli angoli della griglia come
  marker posati sull'ortofoto, oppure un'origine e un azimut, oppure coppie digitate di coordinate
  locali e mondo (una similitudine ai minimi quadrati, con il residuo di ogni punto).
* **Filtri**: despike, destripe (traversa a media zero), clip, passa-basso, passa-alto (il residuo
  di un lisciamento gaussiano: le anomalie piccole sopra la geologia), riempimento dei vuoti,
  ricampionamento - in una copia o sul posto, ogni filtro registrato.
* **Sovrapposizione** all'ortofoto o al DEM della vista raster, con la banda (la slice di una
  prospezione GPR), la palette (grigi, grigi invertiti, divergente, terra, calore) e l'opacità; le
  anomalie disegnate con lo strumento poligono del programma nel layer *Anomalie geofisiche*.
* **Esportazioni**: GeoTIFF e Surfer GRD di una griglia, un GeoPackage con i contorni delle griglie
  e le anomalie (*File > Esporta*), le griglie nel progetto QGIS del programma con la loro scala di
  colori.
* `ppg plugin geophysics import|list|georef|filter|export ...` dalla riga di comando.

```bash
pip install "ppg-geophysics @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geophysics"
ppg plugin geophysics import prospezione.grd --kind magnetometry --name "Area 1"
ppg plugin geophysics georef "Area 1" --origin 438100.0,4553800.0 --azimuth 95
ppg plugin geophysics filter "Area 1" destripe --copy
ppg plugin geophysics export "Area 1 destripe" area1.tif
```

Licenza: GPL-2.0-or-later.
