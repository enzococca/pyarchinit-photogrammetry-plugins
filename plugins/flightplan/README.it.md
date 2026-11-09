# Piani di volo

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.4 o successivo):
i piani di volo del drone, progettati prima di andare sul campo e controllati dopo il volo.

- **Progetto**: l'area (disegnata sulla vista 2D, da un layer, da un file KML/GeoJSON/GeoPackage o da coordinate, con
  un margine), la camera (un elenco di camere da drone, oppure una dall'EXIF di una foto, dal sensore calibrato del
  chunk o scritta a mano: i valori dei sensori non si inventano mai), un obiettivo (ortofoto, DEM, modello 3D completo,
  ispezione ravvicinata, ricognizione veloce, GSD a mano) o una quota, sovrapposizioni, velocità, direzione, doppia
  griglia, obliqui, scatto per punto o a intervallo. Il piano: blocchi e tratte che stanno in una batteria, scatti,
  minuti, batterie, GSD, avvisi.
- **Volo che segue il terreno** sul DEM del progetto (le tegole pubbliche del terreno di AWS dove finisce, o al suo
  posto): ogni punto con la sua quota sul decollo, i decolli delle tratte corretti con il loro suolo.
- **Copertura prevista** sul DEM del chunk: foto per cella e GSD per cella, celle con copertura bassa, una mappa RGBA
  sulla vista 2D. Senza occlusioni.
- **Target GNSS** da un CSV (gradi, UTM, qualsiasi EPSG) o dai marker del chunk, come discese con una scala di foto.
- **Esportazioni** per i droni e per il GIS, una scheda del volo in Excel e HTML, una mappa per tratta sopra l'ortofoto
  del chunk, la cartella dei waypoint di DJI Fly riempita con le tratte.
- **Dopo il volo**: gli scatti previsti contro le foto del chunk (abbinati, mancanti, in più, ripetuti, scarti,
  differenza di quota dall'XMP di DJI) e la copertura data dalle foto allineate contro quella prevista.

Tutto è nel pannello **Piani di volo** (agganciato a destra, nove passi con il loro stato), in un gruppo della colonna
degli strumenti della vista 2D, in *File > Esporta*, nell'albero dell'area di lavoro e sulla riga di comando. La guida è
sul [sito del programma](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/flightplan/).

Il pianificatore - griglie a serpentina, anelli, percorsi punto per punto, tratte che stanno in una batteria, i profili
delle camere, il KMZ DJI e la copia in DJI Fly - viene dal pianificatore di Photo3D dello stesso autore. Un piano JSON di
quel pianificatore si carica così com'è.

> **Spazio aereo, permessi, ostacoli e meteo sono del pilota.** Il plugin calcola una rotta; controllare che il volo
> sia consentito e sicuro spetta al pilota. Guarda la rotta nell'app di volo prima di decollare.

## Le quote nei file

Tutti i file per i droni hanno quote **sopra il punto di decollo**, una per waypoint (un piano che segue il terreno
tiene le sue quote punto per punto in tutti i formati per i droni), e ogni file lo dichiara:

| Formato | Riferimento scritto nel file |
|---|---|
| KMZ WPML DJI (DJI Pilot 2, FlightHub 2, DJI Fly) | `heightMode` / `executeHeightMode` `relativeToStartPoint`, `coordinateMode` `WGS84`, `executeHeight` per punto |
| QGroundControl `.plan` | item frame 3 (`MAV_FRAME_GLOBAL_RELATIVE_ALT`), `AltitudeMode` 1, `globalPlanAltitudeMode` 1 |
| Mission Planner `.waypoints` | frame 3; riga home al decollo con quota 0 |
| Parrot `.mavlink` | frame 3 |
| Litchi CSV | `altitudemode` 0 (sul decollo) |
| KML | `altitudeMode` `absolute` = suolo al decollo + quota sul decollo; quota del decollo e sua sorgente scritte nel file (0 con una nota se non si conosce) |
| GeoJSON | z = quota sul decollo, `riferimento_quote` = `decollo` |
| GeoPackage (sistema del chunk) | `quota_decollo_m` sul decollo; Z = suolo al decollo + quella, nel sistema di quote del DEM; colonna `riferimento_quote` |

Il drone non legge mai KML, GeoJSON o GeoPackage.

## Provato e non provato

- **Provato sul drone vero**: il KMZ di DJI Fly sul DJI Mini 4 Pro (il file del pianificatore).
- **Non ancora importati nelle app vere**: il KMZ WPML in DJI Pilot 2 e FlightHub 2, Litchi, QGroundControl, Mission
  Planner, Parrot e il KML delle aree per le app DJI (sperimentale). In particolare le quote punto per punto di
  QGroundControl, Mission Planner, Parrot e Litchi da un piano che segue il terreno non sono state provate in quelle
  app. I file seguono la specifica di ciascun formato e sono controllati dai test: **provali nell'app vera prima di
  volare.**
- **Non ancora provata su un telecomando vero**: la copia in DJI Fly su Windows con MTP e PowerShell.
- **Non ancora provata su un volo vero**: la copertura ottenuta dopo il volo (provata su dati sintetici).

## Limiti

- Nessuna occlusione nella copertura, prevista o ottenuta (muri e alberi non nascondono il suolo); le celle fuori dal
  DEM sono piane al decollo; le celle fuori dalla ricostruzione non si misurano.
- Senza un'ortofoto nel progetto le mappe delle tratte sono su sfondo grigio (nessuna tegola di mappe di terzi).
- Un chunk in un sistema locale: area solo da un file o in gradi, niente disegno sulla vista, niente copertura.
- Le riprese ravvicinate (una trincea, una parete) non sono nella 0.1.0.
- Il confronto e la copertura diventano «Da rifare» quando cambiano il piano o le foto, dopo un nuovo allineamento, un
  DEM nuovo o posizioni nuove.

## Riga di comando

```
ppg plugin flightplan cameras
ppg plugin flightplan design PROGETTO.ppg --bbox O,S,E,N --goal orto --value 2 [--terrain] [--takeoff LAT,LON]
ppg plugin flightplan show PROGETTO.ppg
ppg plugin flightplan terrain PROGETTO.ppg [--detail fine] [--off]
ppg plugin flightplan targets PROGETTO.ppg --csv target.csv [--epsg N]      # oppure --markers
ppg plugin flightplan coverage PROGETTO.ppg [--cell M] [--low N]
ppg plugin flightplan export PROGETTO.ppg --format kmz-zip --out volo.zip
ppg plugin flightplan djifly PROGETTO.ppg --zip pronte.zip --missions UUID,UUID
ppg plugin flightplan compare PROGETTO.ppg [--plan-id ID] [--max-distance M]
ppg plugin flightplan list PROGETTO.ppg                                     # use ID, remove ID
ppg plugin flightplan design --coords "lat,lon;lat,lon;lat,lon" --plan piano.json   # senza progetto
```

Formati: `kmz`, `kmz-zip`, `djifly-zip`, `plan`, `waypoints`, `mavlink`, `litchi`, `kml`, `geojson`, `gpkg`,
`area-kml`, `sheet-xlsx`, `sheet-html`, `maps-zip`. I piani di un progetto restano nel suo chunk; copertura, confronto e
copie di DJI Fly in `<dati del chunk>/flightplan/`. Ogni esportazione ricalcola prima le quote dei blocchi che seguono
il terreno e sono cambiati nel frattempo. La rete serve solo per le tegole pubbliche del terreno, a richiesta.

Licenza: GPL-2.0-or-later.
